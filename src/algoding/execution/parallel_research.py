from __future__ import annotations

from dataclasses import dataclass

from algoding.data.fx import FrankfurterFxHistoricalClient
from algoding.data.historical import AlpacaHistoricalClient
from algoding.execution.storage import OrderStore
from algoding.research.parallel_strategies import (
    StrategyDefinition,
    build_market_strategies,
    evaluate_strategy,
)
from algoding.settings import Settings


@dataclass(frozen=True)
class MarketPreset:
    name: str
    symbols: list[str]
    source: str


MARKET_PRESETS: dict[str, MarketPreset] = {
    "gold": MarketPreset(name="gold", symbols=["GLD"], source="stocks"),
    "stocks": MarketPreset(name="stocks", symbols=["AAPL", "MSFT", "NVDA", "AMZN", "META"], source="stocks"),
    "fx": MarketPreset(name="fx", symbols=["EURUSD", "USDJPY", "GBPUSD"], source="fx"),
}


class ParallelResearchLedger:
    def __init__(self, settings: Settings) -> None:
        if not settings.has_alpaca_credentials:
            raise RuntimeError("Alpaca credentials are required in .env for parallel research ledger.")
        self._settings = settings
        self._stock_history = AlpacaHistoricalClient(settings)
        self._fx_history = FrankfurterFxHistoricalClient()
        self._store = OrderStore(settings)

    def run_market(self, market: str, bars: int = 90) -> dict[str, object]:
        preset = self._resolve_preset(market)
        self._store.ensure_schema()
        price_series = self._load_price_series(preset, bars)

        symbol_results: list[dict[str, object]] = []
        for symbol, symbol_bars in price_series.items():
            prices = [point["close"] for point in symbol_bars]
            current_price = prices[-1]
            timestamp = str(symbol_bars[-1]["timestamp"])
            existing_books = self._store.get_virtual_strategy_books(symbol)
            existing_by_name = {book["strategy_name"]: book for book in existing_books}
            strategies = build_market_strategies(symbol.lower())

            evaluations: list[dict[str, object]] = []
            books: list[dict[str, object]] = []
            trades: list[dict[str, object]] = []

            for index, definition in enumerate(strategies):
                evaluation = evaluate_strategy(prices, definition, symbol=symbol)
                target_notional = self._target_notional_for_strategy(definition, index)
                result = self._update_book(
                    definition=definition,
                    evaluation=evaluation,
                    existing_book=existing_by_name.get(definition.name),
                    current_price=current_price,
                    timestamp=timestamp,
                    symbol=symbol,
                    target_notional=target_notional,
                )
                evaluations.append(evaluation)
                books.append(result["book"])
                if result["trade"] is not None:
                    trades.append(result["trade"])
                snapshot = dict(result["book"])
                snapshot["created_at"] = timestamp
                self._store.record_virtual_strategy_snapshot(snapshot)

            books.sort(key=lambda item: float(item["total_pnl"]), reverse=True)
            evaluations.sort(key=lambda item: float(item["score"]), reverse=True)
            symbol_results.append(
                {
                    "symbol": symbol,
                    "current_price": current_price,
                    "timestamp": timestamp,
                    "evaluations": evaluations,
                    "books": books,
                    "trades": trades,
                }
            )

        return {
            "market": preset.name,
            "symbols": preset.symbols,
            "results": symbol_results,
        }

    def report_market(self, market: str) -> dict[str, object]:
        preset = self._resolve_preset(market)
        self._store.ensure_schema()
        reports: list[dict[str, object]] = []
        aggregate_leaderboard: list[dict[str, object]] = []
        for symbol in preset.symbols:
            books = self._store.get_virtual_strategy_books(symbol)
            trades = self._store.list_virtual_strategy_trades(symbol, limit=20)
            leaderboard = self._store.get_virtual_strategy_leaderboard(symbol)
            books.sort(key=lambda item: float(item["total_pnl"]), reverse=True)
            reports.append({"symbol": symbol, "leaderboard": leaderboard, "books": books, "recent_trades": trades})
            for row in leaderboard:
                aggregate_leaderboard.append(dict(row))
        aggregate_leaderboard.sort(
            key=lambda item: (float(item["total_pnl"]), float(item.get("score") or 0.0)), reverse=True
        )
        return {
            "market": preset.name,
            "symbols": preset.symbols,
            "aggregate_leaderboard": aggregate_leaderboard,
            "reports": reports,
        }

    def history(self, symbol: str, strategy_name: str | None = None, limit: int = 50) -> dict[str, object]:
        self._store.ensure_schema()
        snapshots = self._store.list_virtual_strategy_snapshots(
            symbol=symbol.upper(), strategy_name=strategy_name, limit=limit
        )
        return {"symbol": symbol.upper(), "strategy_name": strategy_name, "snapshots": snapshots}

    def _resolve_preset(self, market: str) -> MarketPreset:
        normalized = market.strip().lower()
        if normalized not in MARKET_PRESETS:
            supported = ", ".join(sorted(MARKET_PRESETS))
            raise ValueError(f"Unsupported market preset: {market}. Supported values: {supported}")
        return MARKET_PRESETS[normalized]

    def _load_price_series(self, preset: MarketPreset, bars: int) -> dict[str, list[dict[str, object]]]:
        if preset.source == "stocks":
            bars_by_symbol = self._stock_history.get_recent_daily_bars(preset.symbols, bars)
            return {
                symbol: [
                    {"close": float(bar.close), "timestamp": bar.timestamp.isoformat()}
                    for bar in symbol_bars
                ]
                for symbol, symbol_bars in bars_by_symbol.items()
                if symbol_bars
            }

        fx_bars = self._fx_history.get_recent_daily_bars(preset.symbols, bars)
        return {
            symbol: [{"close": float(bar.close), "timestamp": bar.timestamp} for bar in symbol_bars]
            for symbol, symbol_bars in fx_bars.items()
            if symbol_bars
        }

    def _target_notional_for_strategy(self, definition: StrategyDefinition, index: int) -> float:
        overrides = {
            "momentum_daily": 500.0,
            "mean_reversion_daily": 300.0,
            "breakout_daily": 200.0,
            "momentum_aggressive": 300.0,
        }
        for suffix, target in overrides.items():
            if definition.name.endswith(suffix):
                return target
        return 250.0 + (index * 50.0)

    def _update_book(
        self,
        definition: StrategyDefinition,
        evaluation: dict[str, object],
        existing_book: dict[str, object] | None,
        current_price: float,
        timestamp: str,
        symbol: str,
        target_notional: float,
    ) -> dict[str, object]:
        if existing_book is None:
            position_qty = 0.0
            avg_entry_price = None
            realized_pnl = 0.0
            previous_signal = "flat"
        else:
            position_qty = float(existing_book.get("position_qty", 0.0))
            avg_entry_price = existing_book.get("avg_entry_price")
            avg_entry_price = float(avg_entry_price) if avg_entry_price not in (None, "") else None
            realized_pnl = float(existing_book.get("realized_pnl", 0.0))
            previous_signal = str(existing_book.get("last_signal", "flat"))

        signal = str(evaluation["latest_signal"])
        trade: dict[str, object] | None = None

        if signal == "long" and position_qty <= 0:
            position_qty = target_notional / current_price if current_price > 0 else 0.0
            avg_entry_price = current_price
            trade = {
                "strategy_name": definition.name,
                "symbol": symbol,
                "side": "buy",
                "notional": target_notional,
                "quantity": position_qty,
                "price": current_price,
                "signal": signal,
                "created_at": timestamp,
            }
        elif signal == "flat" and position_qty > 0 and avg_entry_price is not None:
            realized_pnl += (current_price - avg_entry_price) * position_qty
            trade = {
                "strategy_name": definition.name,
                "symbol": symbol,
                "side": "sell",
                "notional": position_qty * current_price,
                "quantity": position_qty,
                "price": current_price,
                "signal": signal,
                "created_at": timestamp,
            }
            position_qty = 0.0
            avg_entry_price = None

        unrealized_pnl = 0.0
        if position_qty > 0 and avg_entry_price is not None:
            unrealized_pnl = (current_price - avg_entry_price) * position_qty

        book = {
            "strategy_name": definition.name,
            "symbol": symbol,
            "target_notional": target_notional,
            "position_qty": round(position_qty, 9),
            "avg_entry_price": round(avg_entry_price, 6) if avg_entry_price is not None else None,
            "realized_pnl": round(realized_pnl, 6),
            "unrealized_pnl": round(unrealized_pnl, 6),
            "total_pnl": round(realized_pnl + unrealized_pnl, 6),
            "last_signal": signal,
            "previous_signal": previous_signal,
            "last_price": round(current_price, 6),
            "updated_at": timestamp,
            "score": evaluation["score"],
            "win_rate": evaluation["win_rate"],
            "historical_total_return": evaluation["total_return"],
            "historical_max_drawdown": evaluation["max_drawdown"],
            "historical_trades": evaluation["trades"],
        }

        self._store.upsert_virtual_strategy_book(book)
        if trade is not None:
            self._store.record_virtual_strategy_trade(trade)
        return {"book": book, "trade": trade}
