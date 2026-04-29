from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from math import sqrt

from algoding.portfolio.risk import RiskLimits
from algoding.research.historical_backtest import BacktestBar, _infer_periods_per_year, _max_drawdown
from algoding.research.llm_sentiment import DailySentimentScore


@dataclass(frozen=True)
class LlmOverlayDefinition:
    name: str
    lookback: int
    threshold: float
    entry_sentiment_min: float | None = None
    exit_sentiment_max: float | None = None
    break_even_trigger_pct: float | None = None
    time_stop_bars: int | None = None
    reentry_cooldown_bars: int = 0
    sentiment_freshness_days: int = 1
    no_trade_weak_neutral_cluster: bool = False
    weak_sentiment_abs_threshold: float = 0.1
    weak_neutral_cluster_ratio: float = 0.67
    profit_lock_trigger_pct: float | None = None
    profit_lock_trailing_stop_pct: float | None = None
    partial_take_profit_fraction: float | None = None
    partial_take_profit_trigger_pct: float | None = None


def build_llm_overlay_strategies(
    prefix: str,
    *,
    entry_sentiment_min: float = 0.05,
    exit_sentiment_max: float = 0.02,
) -> list[LlmOverlayDefinition]:
    return [
        LlmOverlayDefinition(name=f"{prefix}_momentum_baseline", lookback=20, threshold=0.02),
        LlmOverlayDefinition(
            name=f"{prefix}_momentum_llm_entry_filter",
            lookback=20,
            threshold=0.02,
            entry_sentiment_min=entry_sentiment_min,
        ),
        LlmOverlayDefinition(
            name=f"{prefix}_momentum_llm_exit_filter",
            lookback=20,
            threshold=0.02,
            exit_sentiment_max=exit_sentiment_max,
        ),
        LlmOverlayDefinition(
            name=f"{prefix}_momentum_llm_combo",
            lookback=20,
            threshold=0.02,
            entry_sentiment_min=entry_sentiment_min,
            exit_sentiment_max=exit_sentiment_max,
        ),
    ]


def build_llm_overlay_trace(
    *,
    bars: list[BacktestBar],
    sentiment_scores: list[DailySentimentScore],
    definition: LlmOverlayDefinition,
    symbol: str,
    market: str,
    window_name: str,
    risk_limits: RiskLimits,
) -> dict[str, object]:
    if len(bars) < definition.lookback + 2:
        raise ValueError(f"Not enough bars to backtest {definition.name} on {symbol} for {window_name}.")
    if len(sentiment_scores) != len(bars):
        raise ValueError("Sentiment scores must align one-to-one with bars.")

    prices = [bar.close for bar in bars]
    equity = 1.0
    equity_curve = [equity]
    in_position = False
    cash_equity = 1.0
    position_units = 0.0
    entry_price: float | None = None
    high_water_mark: float | None = None
    active_trailing_stop_pct = risk_limits.trailing_stop_pct
    profit_lock_active = False
    break_even_active = False
    partial_taken = False
    trade_start_equity: float | None = None
    position_bars = 0
    last_exit_index: int | None = None
    trade_returns: list[float] = []
    exit_reasons: Counter[str] = Counter()
    position_days = 0
    latest_signal = "flat"
    trace_points: list[dict[str, object]] = []
    events: list[dict[str, object]] = []

    for index, bar in enumerate(bars):
        lagged_sentiment = _smoothed_lagged_sentiment(
            sentiment_scores,
            index=index,
            freshness_days=definition.sentiment_freshness_days,
        )
        weak_cluster = _is_weak_neutral_cluster(
            sentiment_scores,
            index=index,
            freshness_days=definition.sentiment_freshness_days,
            weak_abs_threshold=definition.weak_sentiment_abs_threshold,
            required_ratio=definition.weak_neutral_cluster_ratio,
        )
        signal = _momentum_signal(prices[: index + 1], definition)
        latest_signal = "long" if signal == 1 else "flat"
        indicator_value = _momentum_indicator(prices[: index + 1], definition)
        fixed_stop = None
        trailing_stop = None
        effective_stop = None

        if index >= definition.lookback and in_position and entry_price is not None:
            prev_close = bars[index - 1].close
            prior_high_water = max(high_water_mark or entry_price, entry_price)
            base_fixed_stop = entry_price * (1 - risk_limits.stop_loss_pct)
            fixed_stop = max(base_fixed_stop, entry_price if break_even_active else base_fixed_stop)
            trailing_stop = prior_high_water * (1 - active_trailing_stop_pct)
            effective_stop = max(fixed_stop, trailing_stop)

            if bar.low <= effective_stop:
                cash_equity += position_units * effective_stop
                position_units = 0.0
                equity = cash_equity
                if trade_start_equity is not None and trade_start_equity > 0:
                    trade_returns.append((cash_equity / trade_start_equity) - 1)
                position_days += 1
                equity_curve.append(equity)
                exit_reason = "stop_loss" if effective_stop == fixed_stop else "trailing_stop"
                exit_reasons[exit_reason] += 1
                events.append(
                    {
                        "timestamp": bar.timestamp,
                        "event": "sell",
                        "price": round(effective_stop, 6),
                        "reason": exit_reason,
                        "fraction": 1.0,
                    }
                )
                in_position = False
                entry_price = None
                high_water_mark = None
                active_trailing_stop_pct = risk_limits.trailing_stop_pct
                profit_lock_active = False
                break_even_active = False
                partial_taken = False
                trade_start_equity = None
                position_bars = 0
                last_exit_index = index
            else:
                high_water_mark = max(prior_high_water, bar.high, bar.close)
                if (
                    not break_even_active
                    and definition.break_even_trigger_pct is not None
                    and bar.close >= entry_price * (1 + definition.break_even_trigger_pct)
                ):
                    break_even_active = True
                if (
                    not profit_lock_active
                    and definition.profit_lock_trigger_pct is not None
                    and bar.close >= entry_price * (1 + definition.profit_lock_trigger_pct)
                ):
                    profit_lock_active = True
                    active_trailing_stop_pct = (
                        definition.profit_lock_trailing_stop_pct
                        if definition.profit_lock_trailing_stop_pct is not None
                        else active_trailing_stop_pct
                    )
                trailing_stop = high_water_mark * (1 - active_trailing_stop_pct)
                fixed_stop = max(base_fixed_stop, entry_price if break_even_active else base_fixed_stop)
                effective_stop = max(fixed_stop, trailing_stop)
                position_days += 1
                position_bars += 1

                if (
                    not partial_taken
                    and definition.partial_take_profit_fraction is not None
                    and definition.partial_take_profit_trigger_pct is not None
                    and bar.close >= entry_price * (1 + definition.partial_take_profit_trigger_pct)
                ):
                    sell_fraction = min(max(definition.partial_take_profit_fraction, 0.0), 1.0)
                    if sell_fraction > 0:
                        sell_units = position_units * sell_fraction
                        cash_equity += sell_units * bar.close
                        position_units -= sell_units
                        partial_taken = True
                        exit_reasons["partial_take_profit"] += 1
                        events.append(
                            {
                                "timestamp": bar.timestamp,
                                "event": "sell",
                                "price": round(bar.close, 6),
                                "reason": "partial_take_profit",
                                "fraction": round(sell_fraction, 6),
                            }
                        )

                forced_exit = definition.exit_sentiment_max is not None and lagged_sentiment <= definition.exit_sentiment_max
                time_stop_hit = (
                    definition.time_stop_bars is not None
                    and position_bars >= definition.time_stop_bars
                )
                if signal == 0 or forced_exit or time_stop_hit:
                    exit_reason = "llm_exit" if forced_exit and signal == 1 else "signal_exit"
                    if time_stop_hit:
                        exit_reason = "time_stop"
                    cash_equity += position_units * bar.close
                    position_units = 0.0
                    if trade_start_equity is not None and trade_start_equity > 0:
                        trade_returns.append((cash_equity / trade_start_equity) - 1)
                    exit_reasons[exit_reason] += 1
                    events.append(
                        {
                            "timestamp": bar.timestamp,
                            "event": "sell",
                            "price": round(bar.close, 6),
                            "reason": exit_reason,
                            "fraction": 1.0,
                        }
                    )
                    in_position = False
                    entry_price = None
                    high_water_mark = None
                    active_trailing_stop_pct = risk_limits.trailing_stop_pct
                    profit_lock_active = False
                    break_even_active = False
                    partial_taken = False
                    trade_start_equity = None
                    position_bars = 0
                    last_exit_index = index
                equity = cash_equity + (position_units * bar.close)
                equity_curve.append(equity)

        entry_allowed = (
            definition.entry_sentiment_min is None or lagged_sentiment >= definition.entry_sentiment_min
        )
        cooldown_blocked = (
            last_exit_index is not None
            and index <= last_exit_index + definition.reentry_cooldown_bars
        )
        cluster_blocked = definition.no_trade_weak_neutral_cluster and weak_cluster
        if (
            index >= definition.lookback
            and not in_position
            and signal == 1
            and entry_allowed
            and not cooldown_blocked
            and not cluster_blocked
        ):
            in_position = True
            trade_start_equity = cash_equity
            entry_price = bar.close
            high_water_mark = max(bar.high, bar.close)
            active_trailing_stop_pct = risk_limits.trailing_stop_pct
            profit_lock_active = False
            break_even_active = False
            partial_taken = False
            position_bars = 0
            position_units = cash_equity / bar.close if bar.close > 0 else 0.0
            cash_equity = 0.0
            fixed_stop = entry_price * (1 - risk_limits.stop_loss_pct)
            trailing_stop = high_water_mark * (1 - active_trailing_stop_pct)
            effective_stop = max(fixed_stop, trailing_stop)
            events.append(
                {
                    "timestamp": bar.timestamp,
                    "event": "buy",
                    "price": round(bar.close, 6),
                    "reason": "llm_entry" if definition.entry_sentiment_min is not None else "signal_entry",
                    "fraction": 1.0,
                }
            )
            equity = cash_equity + (position_units * bar.close)
        else:
            equity = cash_equity + (position_units * bar.close)

        trace_points.append(
            {
                "timestamp": bar.timestamp,
                "close": round(bar.close, 6),
                "signal": "long" if signal == 1 else "flat",
                "indicator_value": round(indicator_value, 6) if indicator_value is not None else None,
                "threshold_value": definition.threshold,
                "in_position": in_position,
                "equity": round(equity, 6),
                "position_fraction": round((position_units * bar.close) / equity, 6) if equity > 0 else 0.0,
                "lagged_sentiment_score": lagged_sentiment,
                "high_water_mark": round(high_water_mark, 6) if high_water_mark is not None else None,
                "fixed_stop_price": round(fixed_stop, 6) if fixed_stop is not None else None,
                "trailing_stop_price": round(trailing_stop, 6) if trailing_stop is not None else None,
                "effective_stop_price": round(effective_stop, 6) if effective_stop is not None else None,
                "profit_lock_active": profit_lock_active,
                "break_even_active": break_even_active,
                "active_trailing_stop_pct": round(active_trailing_stop_pct, 6),
                "partial_take_profit_taken": partial_taken,
                "cooldown_blocked": cooldown_blocked,
                "weak_neutral_cluster": weak_cluster,
            }
        )

    positive = [value for value in trade_returns if value > 0]
    negative = [value for value in trade_returns if value < 0]
    win_rate = len(positive) / len(trade_returns) if trade_returns else 0.0
    total_return = equity_curve[-1] - 1
    max_drawdown = _max_drawdown(equity_curve)
    exposure_ratio = position_days / max(1, (len(bars) - definition.lookback))
    periods_per_year = _infer_periods_per_year(bars)
    annualized_return = (
        ((1 + total_return) ** (periods_per_year / max(1, len(bars) - 1))) - 1 if total_return > -1 else -1.0
    )
    periodic_returns = [(equity_curve[i] / equity_curve[i - 1]) - 1 for i in range(1, len(equity_curve))]
    variance = sum(value * value for value in periodic_returns) / max(1, len(periodic_returns))
    annualized_volatility = sqrt(variance) * sqrt(periods_per_year) if periodic_returns else 0.0
    profit_factor = (
        sum(positive) / abs(sum(negative))
        if negative and abs(sum(negative)) > 0
        else float(len(positive)) if positive else 0.0
    )
    score = total_return + (max_drawdown * 0.5) + (win_rate * 0.1) + (min(profit_factor, 3.0) * 0.02)

    summary = {
        "strategy_name": definition.name,
        "strategy_family": "llm_news_overlay",
        "symbol": symbol,
        "market": market,
        "window_name": window_name,
        "run_type": "historical_llm_news",
        "lookback": definition.lookback,
        "threshold": definition.threshold,
        "latest_signal": latest_signal,
        "total_return": round(total_return, 6),
        "annualized_return": round(annualized_return, 6),
        "annualized_volatility": round(annualized_volatility, 6),
        "max_drawdown": round(max_drawdown, 6),
        "trades": len(trade_returns),
        "win_rate": round(win_rate, 6),
        "profit_factor": round(profit_factor, 6),
        "exposure_ratio": round(exposure_ratio, 6),
        "score": round(score, 6),
        "bars_tested": len(bars),
        "periods_per_year": round(periods_per_year, 2),
        "risk": {
            "stop_loss_pct": risk_limits.stop_loss_pct,
            "trailing_stop_pct": risk_limits.trailing_stop_pct,
            "entry_sentiment_min": definition.entry_sentiment_min,
            "exit_sentiment_max": definition.exit_sentiment_max,
            "break_even_trigger_pct": definition.break_even_trigger_pct,
            "time_stop_bars": definition.time_stop_bars,
            "reentry_cooldown_bars": definition.reentry_cooldown_bars,
            "sentiment_freshness_days": definition.sentiment_freshness_days,
            "no_trade_weak_neutral_cluster": definition.no_trade_weak_neutral_cluster,
            "profit_lock_trigger_pct": definition.profit_lock_trigger_pct,
            "profit_lock_trailing_stop_pct": definition.profit_lock_trailing_stop_pct,
            "partial_take_profit_fraction": definition.partial_take_profit_fraction,
            "partial_take_profit_trigger_pct": definition.partial_take_profit_trigger_pct,
        },
        "exit_reasons": dict(exit_reasons),
    }
    return {"summary": summary, "events": events, "trace_points": trace_points}


def _momentum_signal(prices: list[float], definition: LlmOverlayDefinition) -> int:
    if len(prices) <= definition.lookback:
        return 0
    recent = prices[-definition.lookback :]
    current = recent[-1]
    average = sum(recent) / len(recent)
    return 1 if current >= average * (1 + definition.threshold) else 0


def _momentum_indicator(prices: list[float], definition: LlmOverlayDefinition) -> float | None:
    if len(prices) <= definition.lookback:
        return None
    recent = prices[-definition.lookback :]
    current = recent[-1]
    average = sum(recent) / len(recent)
    return (current / average) - 1 if average > 0 else None


def _smoothed_lagged_sentiment(
    sentiment_scores: list[DailySentimentScore],
    *,
    index: int,
    freshness_days: int,
) -> float:
    if index <= 0:
        return 0.0
    lookback = max(1, freshness_days)
    start = max(0, index - lookback)
    window = sentiment_scores[start:index]
    if not window:
        return 0.0
    return sum(item.score for item in window) / len(window)


def _is_weak_neutral_cluster(
    sentiment_scores: list[DailySentimentScore],
    *,
    index: int,
    freshness_days: int,
    weak_abs_threshold: float,
    required_ratio: float,
) -> bool:
    if index <= 0:
        return False
    lookback = max(1, freshness_days)
    start = max(0, index - lookback)
    window = sentiment_scores[start:index]
    if not window:
        return False
    weak_count = sum(
        1
        for item in window
        if item.label == "neutral" or abs(float(item.score)) <= weak_abs_threshold
    )
    return (weak_count / len(window)) >= required_ratio
