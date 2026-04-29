from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from algoding.data.news import AlpacaNewsHistoricalClient
from algoding.execution.models import OrderIntent
from algoding.execution.sync import PaperBrokerSync
from algoding.execution.v35_paper import V35PaperTrader, V35PaperPortfolio, V35PaperSymbolState, _whole_share_quantity
from algoding.portfolio.risk import PositionRiskState
from algoding.research.deepseek_directional_model import DirectionalPrediction
from algoding.research.llm_sentiment import DailySentimentScore, LocalLlmNewsSentimentEngine


class V35IntradayTrader(V35PaperTrader):
    def __init__(self, settings) -> None:
        super().__init__(settings)
        self._news_client = AlpacaNewsHistoricalClient(settings)
        sentiment_settings = type(settings)()
        sentiment_settings.alpaca_api_key = settings.alpaca_api_key
        sentiment_settings.alpaca_secret_key = settings.alpaca_secret_key
        sentiment_settings.llm_news_model_name = r"E:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B"
        sentiment_settings.llm_news_cache_dir = Path("cache/llm_news_deepseek_intraday")
        sentiment_settings.llm_news_force_gpu = settings.llm_news_force_gpu
        sentiment_settings.llm_news_quantization = settings.llm_news_quantization
        self._sentiment_engine = LocalLlmNewsSentimentEngine(sentiment_settings)

    def run_intraday_cycle(
        self,
        *,
        submit: bool,
        target_capital: float = 60000.0,
        lookback_bars: int = 180,
    ) -> dict[str, object]:
        portfolio = self.ensure_portfolio(target_capital=target_capital)
        clock = self._trading_client.get_clock()
        if not clock.is_open:
            return {
                "submitted": submit,
                "market_open": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "next_open": str(clock.next_open) if getattr(clock, "next_open", None) else None,
                "next_close": str(clock.next_close) if getattr(clock, "next_close", None) else None,
                "results": [],
            }

        self._store.ensure_schema()
        account = self._trading_client.get_account()
        account_equity = float(account.equity)
        target_total_notional = account_equity * portfolio.capital_ratio
        target_per_symbol = target_total_notional * portfolio.per_symbol_weight

        daily_predictions = self._latest_daily_predictions(portfolio.symbols)
        minute_bars_by_symbol = self._history.get_recent_minute_bars(portfolio.symbols, lookback_bars)
        intraday_sentiment = self._latest_intraday_sentiment(portfolio.symbols)
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
        now = datetime.now(timezone.utc)
        for symbol in portfolio.symbols:
            state = state_by_symbol[symbol]
            minute_bars = minute_bars_by_symbol.get(symbol, [])
            latest_prediction = daily_predictions.get(symbol)
            latest_sentiment = intraday_sentiment.get(symbol)
            model_long = _prediction_is_long(latest_prediction)
            sentiment_long = _sentiment_is_bullish(latest_sentiment)
            current_position = positions.get(symbol)
            current_price = (
                float(current_position.current_price)
                if current_position is not None and current_position.current_price is not None
                else _latest_close(minute_bars)
            )
            position_state = PositionRiskState(
                symbol=symbol,
                qty=float(current_position.qty) if current_position is not None else 0.0,
                avg_entry_price=float(current_position.avg_entry_price) if current_position and current_position.avg_entry_price else None,
                current_price=current_price,
                market_value=float(current_position.market_value) if current_position and current_position.market_value else 0.0,
                high_water_mark=state.high_water_mark,
            )
            stop_decision = self._risk_engine.evaluate_long_stops(position_state)
            state.high_water_mark = stop_decision.high_water_mark

            same_day_reentry_block = _same_day_reentry_blocked(state, now)
            should_exit_for_signal = (
                position_state.qty > 0
                and (not model_long or not sentiment_long)
            )
            should_enter = (
                position_state.qty <= 0
                and model_long
                and sentiment_long
                and not same_day_reentry_block
            )

            trailing_order = open_orders.get(symbol)
            status = "no_action"
            order_payloads: list[dict[str, object]] = []
            intent: OrderIntent | None = None

            if position_state.qty > 0 and stop_decision.should_exit:
                intent = OrderIntent.sample(symbol=symbol, side="sell")
                intent.strategy_name = f"{portfolio.strategy_name}_intraday"
                intent.quantity = position_state.qty
                intent.notional = None
                intent.reason = str(stop_decision.reason)
                state.last_exit_reason = str(stop_decision.reason)
                state.last_exit_at = now.isoformat()
            elif should_exit_for_signal:
                intent = OrderIntent.sample(symbol=symbol, side="sell")
                intent.strategy_name = f"{portfolio.strategy_name}_intraday"
                intent.quantity = position_state.qty
                intent.notional = None
                intent.reason = "intraday_signal_exit"
                state.last_exit_reason = "intraday_signal_exit"
                state.last_exit_at = now.isoformat()
            elif should_enter:
                entry_price = current_price
                entry_qty = _whole_share_quantity(target_per_symbol, entry_price)
                if entry_qty <= 0:
                    entry_qty = 0
                intent = OrderIntent.sample(symbol=symbol, side="buy")
                intent.strategy_name = f"{portfolio.strategy_name}_intraday"
                intent.quantity = entry_qty
                intent.notional = None
                intent.reason = "intraday_signal_entry"
                state.last_exit_reason = None

            if intent is not None:
                self._risk_engine.validate(intent)
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
                protection = self._ensure_intraday_protection(
                    symbol=symbol,
                    position_qty=position_state.qty,
                    stop_price=stop_decision.fixed_stop_price,
                    trailing_order=trailing_order,
                    strategy_name=f"{portfolio.strategy_name}_intraday_protect",
                    submit=submit,
                )
                if protection is not None:
                    order_payloads.append(protection)
                    state.broker_trailing_stop_order_id = str(protection.get("broker_order_id", ""))
                    status = "protected" if submit else "protection_planned"
                elif trailing_order is not None:
                    state.broker_trailing_stop_order_id = str(getattr(trailing_order, "id", ""))

            results.append(
                {
                    "symbol": symbol,
                    "prediction": asdict(latest_prediction) if latest_prediction is not None else None,
                    "sentiment": _sentiment_payload(latest_sentiment),
                    "model_long": model_long,
                    "sentiment_long": sentiment_long,
                    "same_day_reentry_block": same_day_reentry_block,
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
        snapshot = self._record_snapshot(snapshot_name="v35_intraday_cycle")
        return {
            "submitted": submit,
            "market_open": True,
            "timestamp": now.isoformat(),
            "portfolio": asdict(portfolio),
            "target_total_notional": round(target_total_notional, 2),
            "target_per_symbol": round(target_per_symbol, 2),
            "results": results,
            "sync": sync,
            "snapshot": snapshot,
        }

    def run_active_hours(
        self,
        *,
        submit: bool,
        target_capital: float = 60000.0,
        lookback_bars: int = 180,
        max_cycles: int = 1,
        sleep_seconds: int = 60,
        output_dir: str = "reports/paper/v35_intraday",
    ) -> dict[str, object]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        cycles: list[dict[str, object]] = []
        for index in range(max_cycles):
            cycle = self.run_intraday_cycle(
                submit=submit,
                target_capital=target_capital,
                lookback_bars=lookback_bars,
            )
            cycles.append(cycle)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            (output_path / f"v35_intraday_cycle_{stamp}.json").write_text(
                json.dumps(cycle, indent=2),
                encoding="utf-8",
            )
            if not cycle.get("market_open"):
                break
            if index < max_cycles - 1:
                time.sleep(max(1, sleep_seconds))
        return {
            "submitted": submit,
            "cycles": cycles,
            "count": len(cycles),
        }

    def _latest_daily_predictions(self, symbols: list[str]) -> dict[str, DirectionalPrediction | None]:
        predictions_by_symbol, _ = self._latest_predictions(symbols)
        latest_predictions: dict[str, DirectionalPrediction | None] = {}
        for symbol in symbols:
            if not predictions_by_symbol[symbol]:
                latest_predictions[symbol] = None
                continue
            latest_key = max(predictions_by_symbol[symbol])
            latest_predictions[symbol] = predictions_by_symbol[symbol][latest_key]
        return latest_predictions

    def _latest_intraday_sentiment(self, symbols: list[str]) -> dict[str, DailySentimentScore | None]:
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=3)
        market_tz = ZoneInfo("America/New_York")
        trading_day = now.astimezone(market_tz).date()
        results: dict[str, DailySentimentScore | None] = {}
        for symbol in symbols:
            articles = self._news_client.get_news_articles(
                symbol=symbol,
                start=start,
                end=now,
                include_content=False,
                limit=50,
            )
            today_articles = [
                article
                for article in articles
                if datetime.fromisoformat(article.created_at).astimezone(market_tz).date() == trading_day
            ]
            bundles = self._sentiment_engine.build_trading_day_bundles(
                symbol=symbol,
                articles=today_articles,
                trading_days=[trading_day],
            )
            if not bundles:
                results[symbol] = None
                continue
            scores = self._sentiment_engine.score_bundles(bundles, batch_size=4)
            results[symbol] = scores.get(trading_day.isoformat())
        return results

    def _ensure_intraday_protection(
        self,
        *,
        symbol: str,
        position_qty: float,
        stop_price: float | None,
        trailing_order: object | None,
        strategy_name: str,
        submit: bool,
    ) -> dict[str, object] | None:
        fractional = abs(position_qty - round(position_qty)) > 1e-9
        existing_type = str(getattr(trailing_order, "type", "")).lower() if trailing_order is not None else None
        expires_at = getattr(trailing_order, "expires_at", None) if trailing_order is not None else None
        expires_soon = False
        if expires_at is not None:
            try:
                expires_dt = datetime.fromisoformat(str(expires_at))
                expires_soon = expires_dt <= (datetime.now(timezone.utc) + timedelta(days=2))
            except ValueError:
                expires_soon = False
        if fractional:
            expected_stop_price = round(float(stop_price), 2) if stop_price is not None else None
            existing_stop_price = (
                float(getattr(trailing_order, "stop_price"))
                if trailing_order is not None and getattr(trailing_order, "stop_price", None) is not None
                else None
            )
            needs = (
                        trailing_order is None
                        or existing_type != "stop"
                        or expected_stop_price is None
                        or existing_stop_price != expected_stop_price
                        or expires_soon
                    )
            if not needs:
                return None
            if trailing_order is not None and submit:
                self._router.cancel_order(str(getattr(trailing_order, "id", "")))
            return self._router.submit_stop_order(
                symbol=symbol,
                quantity=position_qty,
                stop_price=float(expected_stop_price),
                strategy_name=strategy_name,
            ) if submit else {
                "mode": "paper",
                "submitted": False,
                "symbol": symbol,
                "side": "sell",
                "quantity": position_qty,
                "strategy_name": strategy_name,
                "type": "stop",
                "stop_price": expected_stop_price,
            }
        expected_trail_percent = round(self._risk_limits.trailing_stop_pct * 100, 4)
        existing_trail_percent = (
            float(getattr(trailing_order, "trail_percent"))
            if trailing_order is not None and getattr(trailing_order, "trail_percent", None) is not None
            else None
        )
        needs = (
            trailing_order is None
            or existing_type != "trailing_stop"
            or existing_trail_percent != expected_trail_percent
            or expires_soon
        )
        if not needs:
            return None
        if trailing_order is not None and submit:
            self._router.cancel_order(str(getattr(trailing_order, "id", "")))
        return self._router.submit_trailing_stop_order(
            symbol=symbol,
            quantity=position_qty,
            trail_percent=expected_trail_percent,
            strategy_name=strategy_name,
        ) if submit else {
            "mode": "paper",
            "submitted": False,
            "symbol": symbol,
            "side": "sell",
            "quantity": position_qty,
            "strategy_name": strategy_name,
            "type": "trailing_stop",
            "trail_percent": expected_trail_percent,
        }


def _same_day_reentry_blocked(state: V35PaperSymbolState, now: datetime) -> bool:
    if not state.last_exit_at:
        return False
    try:
        last_exit_at = datetime.fromisoformat(state.last_exit_at)
    except ValueError:
        return False
    return last_exit_at.date() == now.date()


def _prediction_is_long(prediction: DirectionalPrediction | None) -> bool:
    if prediction is None:
        return False
    return prediction.direction == "bullish" and prediction.strength in {"medium", "high"}


def _sentiment_is_bullish(sentiment: DailySentimentScore | None) -> bool:
    if sentiment is None:
        return False
    return sentiment.label == "bullish" and float(sentiment.score) > 0


def _sentiment_payload(sentiment: DailySentimentScore | None) -> dict[str, object] | None:
    if sentiment is None:
        return None
    return {
        "symbol": sentiment.symbol,
        "trading_day": sentiment.trading_day,
        "label": sentiment.label,
        "score": sentiment.score,
        "article_ids": list(sentiment.article_ids),
        "model_name": sentiment.model_name,
    }


def _latest_close(minute_bars: list[object]) -> float | None:
    if not minute_bars:
        return None
    latest = minute_bars[-1]
    return float(getattr(latest, "close", 0.0))
