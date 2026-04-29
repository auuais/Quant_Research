from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from math import floor
from pathlib import Path

from alpaca.trading.client import TradingClient

from algoding.data.historical import AlpacaHistoricalClient
from algoding.execution.models import OrderIntent
from algoding.execution.paper import PaperOrderRouter
from algoding.execution.storage import OrderStore
from algoding.execution.sync import PaperBrokerSync
from algoding.execution.deepseek_directional_research import DeepseekDirectionalResearchLab
from algoding.portfolio.risk import PositionRiskState, RiskLimits, RiskEngine
from algoding.research.deepseek_directional_model import DirectionalPrediction, DirectionalTradeDefinition, build_directional_trade_trace
from algoding.research.historical_backtest import BacktestBar
from algoding.settings import Settings


TOP_V35_SYMBOLS = ["TSLA", "AVGO", "NVDA"]
V35_PORTFOLIO_PATH = Path("config/v35_paper_portfolio.json")


@dataclass
class V35PaperSymbolState:
    symbol: str
    high_water_mark: float | None = None
    last_exit_reason: str | None = None
    broker_trailing_stop_order_id: str | None = None
    last_exit_at: str | None = None


@dataclass
class V35PaperPortfolio:
    version: str
    strategy_name: str
    symbols: list[str]
    target_capital: float
    capital_ratio: float
    per_symbol_weight: float
    started_at: str
    ends_at: str
    states: list[V35PaperSymbolState]


class V35PaperTrader:
    def __init__(self, settings: Settings) -> None:
        if not settings.has_alpaca_credentials:
            raise RuntimeError("Alpaca credentials are required in .env for V3-5 paper trading.")
        self._settings = settings
        self._history = AlpacaHistoricalClient(settings)
        self._store = OrderStore(settings)
        self._router = PaperOrderRouter(settings)
        self._risk_limits = RiskLimits.from_file().with_overrides(stop_loss_pct=0.05, trailing_stop_pct=0.01)
        self._risk_engine = RiskEngine(self._risk_limits)
        self._trading_client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=True,
        )
        self._base_model_path = r"E:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B"
        self._adapter_path = "reports/research/deepseek_directional_v3/deepseek_directional_lora"
        self._cache_path = Path("reports/research/deepseek_directional_v3/finetuned_directional_scores.jsonl")

    def ensure_portfolio(self, target_capital: float = 60000.0) -> V35PaperPortfolio:
        portfolio = self._load_portfolio()
        if portfolio is not None:
            return portfolio
        account = self._trading_client.get_account()
        equity = float(account.equity)
        ratio = target_capital / equity if equity > 0 else 0.0
        started_at = datetime.now(timezone.utc)
        portfolio = V35PaperPortfolio(
            version="V3-5",
            strategy_name="v3_5_paper_top3",
            symbols=TOP_V35_SYMBOLS.copy(),
            target_capital=target_capital,
            capital_ratio=ratio,
            per_symbol_weight=1 / len(TOP_V35_SYMBOLS),
            started_at=started_at.isoformat(),
            ends_at=(started_at + timedelta(days=90)).isoformat(),
            states=[V35PaperSymbolState(symbol=symbol) for symbol in TOP_V35_SYMBOLS],
        )
        self._save_portfolio(portfolio)
        return portfolio

    def run_cycle(self, submit: bool, target_capital: float = 60000.0) -> dict[str, object]:
        self._store.ensure_schema()
        portfolio = self.ensure_portfolio(target_capital=target_capital)
        account = self._trading_client.get_account()
        account_equity = float(account.equity)
        target_total_notional = account_equity * portfolio.capital_ratio
        target_per_symbol = target_total_notional * portfolio.per_symbol_weight
        predictions_by_symbol, test_bars_by_symbol = self._latest_predictions(portfolio.symbols)
        positions = {
            position.symbol: position
            for position in self._trading_client.get_all_positions()
            if position.symbol in set(portfolio.symbols)
        }
        open_orders = {
            order.symbol: order
            for order in self._router.list_open_orders(symbols=portfolio.symbols)
            if str(getattr(order, "side", "")).lower().endswith("sell")
            and str(getattr(order, "type", "")).lower() in {"trailing_stop", "stop"}
        }

        results: list[dict[str, object]] = []
        state_by_symbol = {state.symbol: state for state in portfolio.states}
        for symbol in portfolio.symbols:
            state = state_by_symbol[symbol]
            test_bars = test_bars_by_symbol[symbol]
            predictions = predictions_by_symbol[symbol]
            trace = build_directional_trade_trace(
                bars=test_bars,
                predictions_by_day=predictions,
                definition=DirectionalTradeDefinition(
                    name=f"{symbol.lower()}_v3_5_paper",
                    minimum_strength="medium",
                    require_relative_confirmation=False,
                    require_momentum_confirmation=False,
                ),
                risk_limits=self._risk_limits,
                no_same_day_reentry=True,
            )
            latest_signal = str(trace["summary"]["latest_signal"])
            current_position = positions.get(symbol)
            position_state = PositionRiskState(
                symbol=symbol,
                qty=float(current_position.qty) if current_position is not None else 0.0,
                avg_entry_price=float(current_position.avg_entry_price) if current_position and current_position.avg_entry_price else None,
                current_price=float(current_position.current_price) if current_position and current_position.current_price else (test_bars[-1].close if test_bars else None),
                market_value=float(current_position.market_value) if current_position and current_position.market_value else 0.0,
                high_water_mark=state.high_water_mark,
            )
            stop_decision = self._risk_engine.evaluate_long_stops(position_state)
            state.high_water_mark = stop_decision.high_water_mark

            intent: OrderIntent | None = None
            if position_state.qty > 0 and stop_decision.should_exit:
                intent = OrderIntent.sample(symbol=symbol, side="sell")
                intent.strategy_name = portfolio.strategy_name
                intent.quantity = position_state.qty
                intent.notional = None
                intent.reason = str(stop_decision.reason)
                state.last_exit_reason = str(stop_decision.reason)
                state.last_exit_at = datetime.now(timezone.utc).isoformat()
            elif latest_signal == "flat" and position_state.qty > 0:
                intent = OrderIntent.sample(symbol=symbol, side="sell")
                intent.strategy_name = portfolio.strategy_name
                intent.quantity = position_state.qty
                intent.notional = None
                intent.reason = "signal_exit"
                state.last_exit_reason = "signal_exit"
                state.last_exit_at = datetime.now(timezone.utc).isoformat()
            elif latest_signal == "long" and position_state.qty <= 0:
                entry_price = float(test_bars[-1].close) if test_bars else None
                entry_qty = _whole_share_quantity(target_per_symbol, entry_price)
                if entry_qty <= 0:
                    entry_qty = 0
                intent = OrderIntent.sample(symbol=symbol, side="buy")
                intent.strategy_name = portfolio.strategy_name
                intent.quantity = entry_qty
                intent.notional = None
                intent.reason = "signal_entry"
                state.last_exit_reason = None

            order_payloads: list[dict[str, object]] = []
            status = "no_action"
            if intent is not None:
                self._risk_engine.validate(intent)
                trailing_order = open_orders.get(symbol)
                if trailing_order is not None and submit:
                    self._router.cancel_order(str(getattr(trailing_order, "id", "")))
                    state.broker_trailing_stop_order_id = None
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
                order_payloads.append(payload)
                if intent.side.lower() == "sell":
                    state.high_water_mark = None
                    state.broker_trailing_stop_order_id = None
            elif position_state.qty > 0:
                trailing_order = open_orders.get(symbol)
                fractional = abs(position_state.qty - round(position_state.qty)) > 1e-9
                expected_trail_percent = round(self._risk_limits.trailing_stop_pct * 100, 4)
                expected_stop_price = stop_decision.fixed_stop_price
                existing_type = str(getattr(trailing_order, "type", "")).lower() if trailing_order is not None else None
                existing_trail_percent = None
                existing_stop_price = None
                if trailing_order is not None and getattr(trailing_order, "trail_percent", None) is not None:
                    existing_trail_percent = float(getattr(trailing_order, "trail_percent"))
                if trailing_order is not None and getattr(trailing_order, "stop_price", None) is not None:
                    existing_stop_price = float(getattr(trailing_order, "stop_price"))
                if fractional:
                    needs_trailing_order = (
                        trailing_order is None
                        or existing_type != "stop"
                        or expected_stop_price is None
                        or existing_stop_price != expected_stop_price
                    )
                else:
                    needs_trailing_order = (
                        trailing_order is None
                        or existing_type != "trailing_stop"
                        or existing_trail_percent != expected_trail_percent
                    )
                if submit and needs_trailing_order:
                    if trailing_order is not None:
                        self._router.cancel_order(str(getattr(trailing_order, "id", "")))
                    if fractional:
                        payload = self._router.submit_stop_order(
                            symbol=symbol,
                            quantity=position_state.qty,
                            stop_price=float(expected_stop_price),
                            strategy_name=f"{portfolio.strategy_name}_broker_protect",
                        )
                    else:
                        payload = self._router.submit_trailing_stop_order(
                            symbol=symbol,
                            quantity=position_state.qty,
                            trail_percent=expected_trail_percent,
                            strategy_name=f"{portfolio.strategy_name}_broker_protect",
                        )
                    order_payloads.append(payload)
                    state.broker_trailing_stop_order_id = str(payload.get("broker_order_id", ""))
                    status = "protected"
                elif trailing_order is not None:
                    state.broker_trailing_stop_order_id = str(getattr(trailing_order, "id", ""))

            results.append(
                {
                    "symbol": symbol,
                    "latest_signal": latest_signal,
                    "status": status,
                    "target_notional": round(target_per_symbol, 2),
                    "current_position_qty": position_state.qty,
                    "risk": {
                        "high_water_mark": stop_decision.high_water_mark,
                        "fixed_stop_price": stop_decision.fixed_stop_price,
                        "trailing_stop_price": stop_decision.trailing_stop_price,
                        "effective_stop_price": stop_decision.effective_stop_price,
                    },
                    "orders": order_payloads,
                }
            )

        self._save_portfolio(portfolio)
        sync = PaperBrokerSync(self._settings).run(days=14)
        snapshot = self._record_snapshot(snapshot_name="v35_daily_cycle")
        return {
            "submitted": submit,
            "portfolio": asdict(portfolio),
            "target_total_notional": round(target_total_notional, 2),
            "target_per_symbol": round(target_per_symbol, 2),
            "results": results,
            "sync": sync,
            "snapshot": snapshot,
        }

    def weekly_report(self, output_dir: str = "reports/paper/v35") -> dict[str, object]:
        portfolio = self.ensure_portfolio()
        account = self._trading_client.get_account()
        positions = self._trading_client.get_all_positions()
        snapshots = [
            snapshot
            for snapshot in self._store.list_account_snapshots(limit=200)
            if str(snapshot.get("snapshot_name", "")).startswith("v35_")
        ]
        current_equity = float(account.equity)
        start_equity = float(snapshots[-1]["equity"]) if snapshots else current_equity
        weekly_reference = None
        now = datetime.now(timezone.utc)
        for snapshot in snapshots:
            created_at = datetime.fromisoformat(str(snapshot["created_at"]))
            if (now - created_at).days >= 7:
                weekly_reference = float(snapshot["equity"])
                break
        if weekly_reference is None:
            weekly_reference = start_equity

        weekly_return = ((current_equity / weekly_reference) - 1) if weekly_reference > 0 else 0.0
        since_start_return = ((current_equity / start_equity) - 1) if start_equity > 0 else 0.0
        report = {
            "version": portfolio.version,
            "strategy_name": portfolio.strategy_name,
            "symbols": portfolio.symbols,
            "generated_at": now.isoformat(),
            "period": {
                "started_at": portfolio.started_at,
                "ends_at": portfolio.ends_at,
            },
            "equity": current_equity,
            "cash": float(account.cash),
            "weekly_return": weekly_return,
            "since_start_return": since_start_return,
            "positions": [
                {
                    "symbol": position.symbol,
                    "qty": float(position.qty),
                    "market_value": float(position.market_value),
                    "unrealized_pl": float(position.unrealized_pl) if position.unrealized_pl else 0.0,
                }
                for position in positions
                if position.symbol in set(portfolio.symbols)
            ],
            "snapshot_count": len(snapshots),
        }
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        stamp = now.strftime("%Y%m%d")
        json_path = output_path / f"v35_weekly_report_{stamp}.json"
        md_path = output_path / f"v35_weekly_report_{stamp}.md"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        md_path.write_text(self._build_weekly_markdown(report), encoding="utf-8")
        return report

    def _latest_predictions(
        self,
        symbols: list[str],
        max_bars: int = 1400,
    ) -> tuple[dict[str, dict[str, DirectionalPrediction]], dict[str, list[BacktestBar]]]:
        bars_by_symbol_raw = self._history.get_recent_bars(symbols + ["SPY"], max_bars, timeframe="day")
        bars_by_symbol = {
            symbol: [
                BacktestBar(
                    timestamp=bar.timestamp.isoformat(),
                    open=float(bar.open),
                    close=float(bar.close),
                    high=float(bar.high),
                    low=float(bar.low),
                    volume=float(getattr(bar, "volume", 0.0) or 0.0),
                )
                for bar in values
            ]
            for symbol, values in bars_by_symbol_raw.items()
        }
        split_helper = DeepseekDirectionalResearchLab(self._settings)
        split_by_day = split_helper._build_date_splits(bars_by_symbol["SPY"])
        cached_predictions = _load_directional_cache(self._cache_path, symbols=symbols)
        predictions_by_symbol: dict[str, dict[str, DirectionalPrediction]] = {}
        test_bars_by_symbol: dict[str, list[BacktestBar]] = {}
        for symbol in symbols:
            bars = bars_by_symbol[symbol]
            test_bars = [
                bar
                for bar in bars
                if split_by_day.get(datetime.fromisoformat(bar.timestamp).date().isoformat()) == "test"
            ]
            predictions_by_symbol[symbol] = _fill_predictions_for_test_bars(
                test_bars=test_bars,
                cached_predictions=cached_predictions.get(symbol, {}),
                symbol=symbol,
            )
            test_bars_by_symbol[symbol] = test_bars
        return predictions_by_symbol, test_bars_by_symbol

    def _record_snapshot(self, snapshot_name: str) -> dict[str, object]:
        account = self._trading_client.get_account()
        positions = self._trading_client.get_all_positions()
        snapshot = {
            "snapshot_name": snapshot_name,
            "equity": float(account.equity) if account.equity is not None else None,
            "cash": float(account.cash) if account.cash is not None else None,
            "buying_power": float(account.buying_power) if account.buying_power is not None else None,
            "positions_count": len([position for position in positions if position.symbol in set(TOP_V35_SYMBOLS)]),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "account": {
                "account_number": account.account_number,
                "positions": [
                    {
                        "symbol": position.symbol,
                        "qty": float(position.qty),
                        "market_value": float(position.market_value),
                        "unrealized_pl": float(position.unrealized_pl) if position.unrealized_pl else 0.0,
                    }
                    for position in positions
                    if position.symbol in set(TOP_V35_SYMBOLS)
                ],
            },
        }
        self._store.record_account_snapshot(snapshot)
        return snapshot

    def _load_portfolio(self) -> V35PaperPortfolio | None:
        if not V35_PORTFOLIO_PATH.exists():
            return None
        payload = json.loads(V35_PORTFOLIO_PATH.read_text(encoding="utf-8"))
        return V35PaperPortfolio(
            version=str(payload["version"]),
            strategy_name=str(payload["strategy_name"]),
            symbols=[str(value) for value in payload["symbols"]],
            target_capital=float(payload["target_capital"]),
            capital_ratio=float(payload["capital_ratio"]),
            per_symbol_weight=float(payload["per_symbol_weight"]),
            started_at=str(payload["started_at"]),
            ends_at=str(payload["ends_at"]),
            states=[
                V35PaperSymbolState(
                    symbol=str(state["symbol"]),
                    high_water_mark=float(state["high_water_mark"]) if state.get("high_water_mark") is not None else None,
                    last_exit_reason=str(state["last_exit_reason"]) if state.get("last_exit_reason") is not None else None,
                    broker_trailing_stop_order_id=str(state["broker_trailing_stop_order_id"]) if state.get("broker_trailing_stop_order_id") is not None else None,
                    last_exit_at=str(state["last_exit_at"]) if state.get("last_exit_at") is not None else None,
                )
                for state in payload["states"]
            ],
        )

    def _save_portfolio(self, portfolio: V35PaperPortfolio) -> None:
        V35_PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
        V35_PORTFOLIO_PATH.write_text(json.dumps(asdict(portfolio), indent=2), encoding="utf-8")

    @staticmethod
    def _build_weekly_markdown(report: dict[str, object]) -> str:
        lines = [
            "# V3-5 Weekly Paper Report",
            "",
            f"- Generated: `{report['generated_at']}`",
            f"- Symbols: `{', '.join(report['symbols'])}`",
            f"- Equity: `${report['equity']:.2f}`",
            f"- Cash: `${report['cash']:.2f}`",
            f"- Weekly return: `{report['weekly_return']:.2%}`",
            f"- Since start return: `{report['since_start_return']:.2%}`",
            "",
            "## Positions",
            "",
            "| Symbol | Qty | Market value | Unrealized P/L |",
            "|---|---:|---:|---:|",
        ]
        for position in report["positions"]:
            lines.append(
                f"| `{position['symbol']}` | `{position['qty']}` | `${position['market_value']:.2f}` | `${position['unrealized_pl']:.2f}` |"
            )
        return "\n".join(lines) + "\n"


def _load_directional_cache(path: Path, *, symbols: list[str]) -> dict[str, dict[str, DirectionalPrediction]]:
    allowed = set(symbols)
    results: dict[str, dict[str, DirectionalPrediction]] = {symbol: {} for symbol in symbols}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        symbol = str(payload.get("symbol", "")).upper()
        if symbol not in allowed:
            continue
        results[symbol][str(payload["trading_day"])] = DirectionalPrediction(
            symbol=symbol,
            trading_day=str(payload["trading_day"]),
            direction=str(payload["direction"]),
            strength=str(payload["strength"]),
            relative_to_spy=str(payload["relative_to_spy"]),
            event_type=str(payload.get("event_type", "other")),
            session=str(payload.get("session", "mixed")),
            raw_completion=str(payload.get("raw_completion", "")),
        )
    return results


def _fill_predictions_for_test_bars(
    *,
    test_bars: list[BacktestBar],
    cached_predictions: dict[str, DirectionalPrediction],
    symbol: str,
) -> dict[str, DirectionalPrediction]:
    resolved: dict[str, DirectionalPrediction] = {}
    latest: DirectionalPrediction | None = None
    for bar in test_bars:
        trading_day = datetime.fromisoformat(bar.timestamp).date().isoformat()
        prediction = cached_predictions.get(trading_day)
        if prediction is not None:
            latest = prediction
            resolved[trading_day] = prediction
        elif latest is not None:
            resolved[trading_day] = DirectionalPrediction(
                symbol=symbol,
                trading_day=trading_day,
                direction=latest.direction,
                strength=latest.strength,
                relative_to_spy=latest.relative_to_spy,
                event_type=latest.event_type,
                session=latest.session,
                raw_completion=latest.raw_completion,
            )
    return resolved


def _whole_share_quantity(target_notional: float, price: float | None) -> int:
    if price is None or price <= 0:
        return 0
    return max(0, int(floor(target_notional / price)))
