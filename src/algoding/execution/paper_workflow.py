from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient

from algoding.data.historical import AlpacaHistoricalClient
from algoding.execution.models import OrderIntent
from algoding.execution.paper import PaperOrderRouter
from algoding.execution.sync import PaperBrokerSync
from algoding.execution.storage import OrderStore
from algoding.portfolio.risk import RiskEngine
from algoding.research.momentum import rank_by_simple_momentum
from algoding.settings import Settings


@dataclass
class RebalancePlan:
    rankings: list[dict[str, float | str]]
    targets: list[str]
    current_positions: dict[str, float]
    orders: list[OrderIntent]


class MomentumPaperTrader:
    def __init__(self, settings: Settings) -> None:
        if not settings.has_alpaca_credentials:
            raise RuntimeError("Alpaca credentials are required in .env for paper trading.")
        self._settings = settings
        self._history = AlpacaHistoricalClient(settings)
        self._trading_client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=True,
        )
        self._router = PaperOrderRouter(settings)
        self._store = OrderStore(settings)
        self._risk_engine = RiskEngine()

    def get_account_summary(self) -> dict[str, object]:
        account = self._trading_client.get_account()
        positions = self._trading_client.get_all_positions()
        return {
            "account_number": account.account_number,
            "status": str(account.status),
            "equity": account.equity,
            "cash": account.cash,
            "buying_power": account.buying_power,
            "paper": True,
            "positions": [
                {
                    "symbol": position.symbol,
                    "qty": position.qty,
                    "market_value": position.market_value,
                    "unrealized_pl": position.unrealized_pl,
                }
                for position in positions
            ],
        }

    def record_account_snapshot(self, snapshot_name: str = "basket_daily") -> dict[str, object]:
        summary = self.get_account_summary()
        snapshot = {
            "snapshot_name": snapshot_name,
            "equity": float(summary["equity"]) if summary.get("equity") is not None else None,
            "cash": float(summary["cash"]) if summary.get("cash") is not None else None,
            "buying_power": float(summary["buying_power"]) if summary.get("buying_power") is not None else None,
            "positions_count": len(summary["positions"]),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "account": summary,
        }
        self._store.ensure_schema()
        self._store.record_account_snapshot(snapshot)
        return snapshot

    def list_account_snapshots(self, limit: int = 10) -> list[dict[str, object]]:
        self._store.ensure_schema()
        return self._store.list_account_snapshots(limit)

    def run_daily_cycle(self, symbols: list[str], submit: bool) -> dict[str, object]:
        plan = self.build_plan(symbols)
        orders = self.execute_plan(plan, submit=submit)
        sync = PaperBrokerSync(self._settings).run(days=14)
        snapshot = self.record_account_snapshot("basket_daily")
        return {
            "symbols": symbols,
            "submitted": submit,
            "targets": plan.targets,
            "current_positions": plan.current_positions,
            "rankings": plan.rankings,
            "orders": orders,
            "sync": sync,
            "snapshot": snapshot,
        }

    def build_plan(self, symbols: list[str]) -> RebalancePlan:
        closes = self._history.get_recent_daily_closes(symbols, self._settings.paper_lookback_bars)
        rankings = rank_by_simple_momentum(closes, self._settings.paper_lookback_bars)
        targets = [
            str(item["symbol"])
            for item in rankings[: self._settings.paper_max_positions]
            if float(item["momentum"]) > 0
        ]
        positions = self._trading_client.get_all_positions()
        current_positions = {
            position.symbol: float(position.qty_available or position.qty) for position in positions
        }
        orders: list[OrderIntent] = []

        managed_symbols = set(symbols)
        for symbol, qty in current_positions.items():
            if symbol in managed_symbols and symbol not in targets and qty > 0:
                intent = OrderIntent.sample(symbol=symbol, side="sell")
                intent.quantity = qty
                orders.append(intent)

        for symbol in targets:
            if current_positions.get(symbol, 0) > 0:
                continue
            intent = OrderIntent.sample(symbol=symbol, side="buy")
            intent.quantity = 0
            intent.notional = self._settings.paper_position_notional
            orders.append(intent)

        return RebalancePlan(
            rankings=rankings,
            targets=targets,
            current_positions=current_positions,
            orders=orders,
        )

    def execute_plan(self, plan: RebalancePlan, submit: bool) -> list[dict[str, object]]:
        self._store.ensure_schema()
        results: list[dict[str, object]] = []
        for intent in plan.orders:
            self._risk_engine.validate(intent)
            if submit:
                payload = self._router.submit_order(intent)
                status = "submitted" if payload.get("submitted") else "prepared"
                broker_order_id = str(payload.get("broker_order_id", ""))
            else:
                payload = {
                    "mode": "paper",
                    "submitted": False,
                    "symbol": intent.symbol,
                    "side": intent.side,
                    "quantity": intent.quantity,
                    "notional": intent.notional,
                    "strategy_name": intent.strategy_name,
                    "client_order_id": intent.client_order_id,
                }
                status = "planned"
                broker_order_id = ""
            self._store.record_order(intent, payload, status=status, broker_order_id=broker_order_id)
            results.append(payload)
        return results
