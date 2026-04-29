from __future__ import annotations

from dataclasses import dataclass

from alpaca.trading.client import TradingClient

from algoding.data.historical import AlpacaHistoricalClient
from algoding.execution.models import OrderIntent
from algoding.execution.paper import PaperOrderRouter
from algoding.execution.storage import OrderStore
from algoding.portfolio.risk import RiskEngine
from algoding.research.xle_strategies import evaluate_registered_xle_strategies
from algoding.settings import Settings


@dataclass
class XleRunResult:
    evaluations: list[dict[str, object]]
    best: dict[str, object]
    current_position_qty: float
    orders: list[OrderIntent]


class XleStrategyLab:
    def __init__(self, settings: Settings) -> None:
        if not settings.has_alpaca_credentials:
            raise RuntimeError("Alpaca credentials are required in .env for XLE strategy lab.")
        self._settings = settings
        self._history = AlpacaHistoricalClient(settings)
        self._store = OrderStore(settings)
        self._router = PaperOrderRouter(settings)
        self._risk_engine = RiskEngine()
        self._trading_client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=True,
        )

    def compare(self, symbol: str = "XLE", bars: int = 90) -> list[dict[str, object]]:
        bar_map = self._history.get_recent_daily_bars([symbol], bars)
        prices = [float(bar.close) for bar in bar_map.get(symbol, [])]
        evaluations = evaluate_registered_xle_strategies(prices)
        self._store.ensure_schema()
        for summary in evaluations:
            self._store.record_strategy_run(summary)
        return evaluations

    def latest_report(self, symbol: str = "XLE", limit: int = 10) -> list[dict[str, object]]:
        self._store.ensure_schema()
        return self._store.list_latest_strategy_runs(symbol, limit)

    def build_best_strategy_run(self, symbol: str = "XLE", bars: int = 90) -> XleRunResult:
        evaluations = self.compare(symbol=symbol, bars=bars)
        best = evaluations[0]
        current_qty = 0.0
        try:
            position = self._trading_client.get_open_position(symbol)
            current_qty = float(position.qty)
        except Exception:
            current_qty = 0.0

        orders: list[OrderIntent] = []
        if best["latest_signal"] == "long" and current_qty <= 0:
            intent = OrderIntent.sample(symbol=symbol, side="buy")
            intent.strategy_name = str(best["strategy_name"])
            intent.quantity = 0
            intent.notional = self._settings.paper_position_notional
            orders.append(intent)
        elif best["latest_signal"] == "flat" and current_qty > 0:
            intent = OrderIntent.sample(symbol=symbol, side="sell")
            intent.strategy_name = str(best["strategy_name"])
            intent.quantity = current_qty
            intent.notional = None
            orders.append(intent)

        return XleRunResult(
            evaluations=evaluations,
            best=best,
            current_position_qty=current_qty,
            orders=orders,
        )

    def execute_best_strategy(self, run: XleRunResult, submit: bool) -> list[dict[str, object]]:
        self._store.ensure_schema()
        results: list[dict[str, object]] = []
        for intent in run.orders:
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
