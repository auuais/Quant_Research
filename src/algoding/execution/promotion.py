from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from alpaca.trading.client import TradingClient

from algoding.execution.models import OrderIntent
from algoding.execution.paper import PaperOrderRouter
from algoding.execution.storage import OrderStore
from algoding.execution.sync import PaperBrokerSync
from algoding.portfolio.risk import RiskEngine
from algoding.settings import Settings


@dataclass
class PromotedCandidate:
    market: str
    symbol: str
    strategy_name: str
    notional: float
    active: bool
    promoted_at: str
    high_water_mark: float | None = None
    last_exit_reason: str | None = None
    stop_loss_pct: float | None = None
    trailing_stop_pct: float | None = None


class PromotionRegistry:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or Path("config/promoted_candidates.json")

    def list_candidates(self) -> list[PromotedCandidate]:
        if not self._path.exists():
            return []
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        return [PromotedCandidate(**item) for item in payload]

    def upsert_candidate(self, candidate: PromotedCandidate) -> PromotedCandidate:
        candidates = self.list_candidates()
        updated: list[PromotedCandidate] = []
        replaced = False
        for existing in candidates:
            if existing.symbol.upper() == candidate.symbol.upper():
                updated.append(candidate)
                replaced = True
            else:
                updated.append(existing)
        if not replaced:
            updated.append(candidate)
        self._save(updated)
        return candidate

    def replace_all(self, candidates: list[PromotedCandidate]) -> None:
        self._save(candidates)

    def _save(self, candidates: list[PromotedCandidate]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(candidate) for candidate in candidates]
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class PromotedPaperTrader:
    _UNSUPPORTED_MARKETS = {"fx"}

    def __init__(self, settings: Settings) -> None:
        if not settings.has_alpaca_credentials:
            raise RuntimeError("Alpaca credentials are required in .env for promoted paper trading.")
        self._settings = settings
        self._registry = PromotionRegistry()
        self._store = OrderStore(settings)
        self._router = PaperOrderRouter(settings)
        self._risk_engine = RiskEngine()
        self._trading_client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=True,
        )

    def promote(self, market: str, symbol: str, strategy_name: str, notional: float) -> dict[str, object]:
        normalized_market = market.strip().lower()
        normalized_symbol = symbol.strip().upper()
        if normalized_market in self._UNSUPPORTED_MARKETS:
            raise ValueError(f"{normalized_market} is research-only in the current broker setup.")

        self._store.ensure_schema()
        books = self._store.get_virtual_strategy_books(normalized_symbol)
        strategy_book = next((book for book in books if str(book["strategy_name"]) == strategy_name), None)
        if strategy_book is None:
            raise ValueError(
                f"No internal research book found for symbol={normalized_symbol} strategy={strategy_name}."
            )

        candidate = PromotedCandidate(
            market=normalized_market,
            symbol=normalized_symbol,
            strategy_name=strategy_name,
            notional=notional,
            active=True,
            promoted_at=datetime.now(timezone.utc).isoformat(),
            stop_loss_pct=self._risk_engine._limits.stop_loss_pct,
            trailing_stop_pct=self._risk_engine._limits.trailing_stop_pct,
        )
        self._registry.upsert_candidate(candidate)
        return {
            "promoted": asdict(candidate),
            "book": strategy_book,
        }

    def list_promoted(self) -> list[dict[str, object]]:
        return [asdict(candidate) for candidate in self._registry.list_candidates()]

    def run_promoted(self, submit: bool) -> dict[str, object]:
        self._store.ensure_schema()
        candidates = [candidate for candidate in self._registry.list_candidates() if candidate.active]
        for candidate in candidates:
            if candidate.stop_loss_pct is None:
                candidate.stop_loss_pct = self._risk_engine._limits.stop_loss_pct
            if candidate.trailing_stop_pct is None:
                candidate.trailing_stop_pct = self._risk_engine._limits.trailing_stop_pct
        account = self._trading_client.get_account()
        positions = self._trading_client.get_all_positions()
        account_equity = float(account.equity)
        gross_exposure_ratio = self._gross_exposure_ratio(account_equity, positions)
        open_positions_count = len(positions)
        self._risk_engine.check_daily_loss_halt(account_equity, self._reference_equity_for_today())
        results: list[dict[str, object]] = []
        updated_candidates: list[PromotedCandidate] = []
        for candidate in candidates:
            result = self._run_candidate(
                candidate,
                submit=submit,
                account_equity=account_equity,
                gross_exposure_ratio=gross_exposure_ratio,
                open_positions_count=open_positions_count,
            )
            updated_candidates.append(result["candidate"])
            results.append(result["result"])
        self._registry.replace_all(updated_candidates)
        sync = PaperBrokerSync(self._settings).run(days=14)
        snapshot = self._record_promoted_snapshot()
        return {
            "submitted": submit,
            "candidates": [asdict(candidate) for candidate in updated_candidates],
            "results": results,
            "sync": sync,
            "snapshot": snapshot,
        }

    def _run_candidate(
        self,
        candidate: PromotedCandidate,
        submit: bool,
        account_equity: float,
        gross_exposure_ratio: float,
        open_positions_count: int,
    ) -> dict[str, object]:
        books = self._store.get_virtual_strategy_books(candidate.symbol)
        strategy_book = next(
            (book for book in books if str(book["strategy_name"]) == candidate.strategy_name),
            None,
        )
        if strategy_book is None:
            return {
                "candidate": candidate,
                "result": {
                    "symbol": candidate.symbol,
                    "strategy_name": candidate.strategy_name,
                    "status": "missing_book",
                    "orders": [],
                },
            }

        position_state = self._current_position_state(candidate)
        intent: OrderIntent | None = None
        last_signal = str(strategy_book.get("last_signal", "flat"))
        stop_decision = self._risk_engine.evaluate_long_stops(position_state)
        candidate.high_water_mark = stop_decision.high_water_mark

        if position_state.qty > 0 and stop_decision.should_exit:
            intent = OrderIntent.sample(symbol=candidate.symbol, side="sell")
            intent.strategy_name = candidate.strategy_name
            intent.quantity = position_state.qty
            intent.notional = None
            intent.reason = str(stop_decision.reason)
            candidate.last_exit_reason = str(stop_decision.reason)
        elif last_signal == "flat" and position_state.qty > 0:
            intent = OrderIntent.sample(symbol=candidate.symbol, side="sell")
            intent.strategy_name = candidate.strategy_name
            intent.quantity = position_state.qty
            intent.notional = None
            intent.reason = "signal_exit"
            candidate.last_exit_reason = "signal_exit"
        elif last_signal == "long" and position_state.qty <= 0:
            intent = OrderIntent.sample(symbol=candidate.symbol, side="buy")
            intent.strategy_name = candidate.strategy_name
            intent.quantity = 0
            intent.notional = candidate.notional
            intent.reason = "signal_entry"
            self._risk_engine.validate_promoted_entry(
                intent=intent,
                account_equity=account_equity,
                gross_exposure_ratio=gross_exposure_ratio,
                open_positions_count=open_positions_count,
                current_position_qty=position_state.qty,
            )
            candidate.last_exit_reason = None

        if intent is None:
            return {
                "candidate": candidate,
                "result": {
                    "symbol": candidate.symbol,
                    "strategy_name": candidate.strategy_name,
                    "status": "no_action",
                    "current_position_qty": position_state.qty,
                    "last_signal": last_signal,
                    "risk": {
                        "high_water_mark": stop_decision.high_water_mark,
                        "fixed_stop_price": stop_decision.fixed_stop_price,
                        "trailing_stop_price": stop_decision.trailing_stop_price,
                        "effective_stop_price": stop_decision.effective_stop_price,
                    },
                    "orders": [],
                },
            }

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
                "reason": intent.reason,
            }
            status = "planned"
            broker_order_id = ""
        self._store.record_order(intent, payload, status=status, broker_order_id=broker_order_id)
        if intent.side.lower() == "sell":
            candidate.high_water_mark = None
        return {
            "candidate": candidate,
            "result": {
                "symbol": candidate.symbol,
                "strategy_name": candidate.strategy_name,
                "status": status,
                "current_position_qty": position_state.qty,
                "last_signal": last_signal,
                "risk": {
                    "high_water_mark": stop_decision.high_water_mark,
                    "fixed_stop_price": stop_decision.fixed_stop_price,
                    "trailing_stop_price": stop_decision.trailing_stop_price,
                    "effective_stop_price": stop_decision.effective_stop_price,
                },
                "orders": [payload],
            },
        }

    def _current_position_state(self, candidate: PromotedCandidate):
        try:
            position = self._trading_client.get_open_position(candidate.symbol)
            from algoding.portfolio.risk import PositionRiskState

            return PositionRiskState(
                symbol=candidate.symbol,
                qty=float(position.qty),
                avg_entry_price=float(position.avg_entry_price) if position.avg_entry_price is not None else None,
                current_price=float(position.current_price) if position.current_price is not None else None,
                market_value=float(position.market_value) if position.market_value is not None else 0.0,
                high_water_mark=candidate.high_water_mark,
            )
        except Exception:
            from algoding.portfolio.risk import PositionRiskState

            return PositionRiskState(
                symbol=candidate.symbol,
                qty=0.0,
                avg_entry_price=None,
                current_price=None,
                market_value=0.0,
                high_water_mark=candidate.high_water_mark,
            )

    def _gross_exposure_ratio(self, account_equity: float, positions: list[object]) -> float:
        if account_equity <= 0:
            return 0.0
        total_market_value = 0.0
        for position in positions:
            try:
                total_market_value += abs(float(position.market_value))
            except Exception:
                continue
        return total_market_value / account_equity

    def _reference_equity_for_today(self) -> float | None:
        snapshots = self._store.list_account_snapshots(limit=20)
        todays = [snapshot for snapshot in snapshots if self._risk_engine.is_snapshot_from_today(snapshot["created_at"])]
        if not todays:
            return None
        oldest = todays[-1]
        equity = oldest.get("equity")
        return float(equity) if equity is not None else None

    def _record_promoted_snapshot(self) -> dict[str, object]:
        account = self._trading_client.get_account()
        positions = self._trading_client.get_all_positions()
        snapshot = {
            "snapshot_name": "promoted_cycle",
            "equity": float(account.equity) if account.equity is not None else None,
            "cash": float(account.cash) if account.cash is not None else None,
            "buying_power": float(account.buying_power) if account.buying_power is not None else None,
            "positions_count": len(positions),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "account": {
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
            },
        }
        self._store.record_account_snapshot(snapshot)
        return snapshot
