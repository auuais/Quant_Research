from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from zoneinfo import ZoneInfo

from algoding.portfolio.risk import RiskLimits
from algoding.research.parallel_strategies import StrategyDefinition


@dataclass(frozen=True)
class BacktestBar:
    timestamp: str
    close: float
    high: float
    low: float
    open: float | None = None
    volume: float | None = None


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        drawdown = (value / peak) - 1 if peak > 0 else 0.0
        max_dd = min(max_dd, drawdown)
    return max_dd


def _infer_periods_per_year(bars: list[BacktestBar]) -> float:
    dates = {bar.timestamp.split("T", 1)[0] for bar in bars}
    trading_days = max(1, len(dates))
    bars_per_day = len(bars) / trading_days
    return max(252.0, bars_per_day * 252.0)


def _latest_signal(prices: list[float], definition: StrategyDefinition) -> int:
    if len(prices) <= definition.lookback:
        return 0

    recent = prices[-definition.lookback :]
    current = recent[-1]
    average = sum(recent) / len(recent)
    max_recent = max(recent[:-1]) if len(recent) > 1 else current

    if definition.signal_type == "momentum":
        return 1 if current >= average * (1 + definition.threshold) else 0
    if definition.signal_type == "mean_reversion":
        return 1 if current <= average * (1 - definition.threshold) else 0
    if definition.signal_type == "breakout":
        return 1 if current >= max_recent else 0
    return 0


def _indicator_value(prices: list[float], definition: StrategyDefinition) -> float | None:
    if len(prices) <= definition.lookback:
        return None

    recent = prices[-definition.lookback :]
    current = recent[-1]
    average = sum(recent) / len(recent)
    max_recent = max(recent[:-1]) if len(recent) > 1 else current

    if definition.signal_type == "momentum":
        return (current / average) - 1 if average > 0 else None
    if definition.signal_type == "mean_reversion":
        return ((average - current) / average) if average > 0 else None
    if definition.signal_type == "breakout":
        return (current / max_recent) - 1 if max_recent > 0 else None
    return None


def _trading_day_key(timestamp: str) -> str:
    return datetime.fromisoformat(timestamp).astimezone(ZoneInfo("America/New_York")).date().isoformat()


def build_backtest_trace(
    bars: list[BacktestBar],
    definition: StrategyDefinition,
    symbol: str,
    market: str,
    window_name: str,
    risk_limits: RiskLimits,
    no_same_day_reentry: bool = False,
) -> dict[str, object]:
    if len(bars) < definition.lookback + 2:
        raise ValueError(f"Not enough bars to backtest {definition.name} on {symbol} for {window_name}.")

    prices = [bar.close for bar in bars]
    equity = 1.0
    equity_curve = [equity]
    in_position = False
    entry_price: float | None = None
    high_water_mark: float | None = None
    trade_returns: list[float] = []
    exit_reasons: Counter[str] = Counter()
    position_days = 0
    latest_signal = "flat"
    trace_points: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    blocked_reentry_day: str | None = None

    for index, bar in enumerate(bars):
        trading_day = _trading_day_key(bar.timestamp)
        signal = _latest_signal(prices[: index + 1], definition) if index >= definition.lookback else 0
        latest_signal = "long" if signal == 1 else "flat"
        indicator_value = _indicator_value(prices[: index + 1], definition)
        threshold_value = definition.threshold if definition.signal_type != "breakout" else 0.0
        fixed_stop = None
        trailing_stop = None
        effective_stop = None
        exited_this_bar = False

        if index >= definition.lookback and in_position and entry_price is not None:
            prev_close = bars[index - 1].close
            prior_high_water = max(high_water_mark or entry_price, entry_price)
            fixed_stop = entry_price * (1 - risk_limits.stop_loss_pct)
            trailing_stop = prior_high_water * (1 - risk_limits.trailing_stop_pct)
            effective_stop = max(value for value in [fixed_stop, trailing_stop] if value is not None)

            if bar.low <= effective_stop:
                daily_return = (effective_stop / prev_close) - 1 if prev_close > 0 else 0.0
                equity *= 1 + daily_return
                trade_returns.append((effective_stop / entry_price) - 1 if entry_price > 0 else 0.0)
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
                    }
                )
                in_position = False
                entry_price = None
                high_water_mark = None
                exited_this_bar = True
                if no_same_day_reentry:
                    blocked_reentry_day = trading_day
            else:
                high_water_mark = max(prior_high_water, bar.high, bar.close)
                trailing_stop = high_water_mark * (1 - risk_limits.trailing_stop_pct)
                effective_stop = max(value for value in [fixed_stop, trailing_stop] if value is not None)
                daily_return = (bar.close / prev_close) - 1 if prev_close > 0 else 0.0
                equity *= 1 + daily_return
                position_days += 1
                if signal == 0:
                    trade_returns.append((bar.close / entry_price) - 1 if entry_price > 0 else 0.0)
                    exit_reasons["signal_exit"] += 1
                    events.append(
                        {
                            "timestamp": bar.timestamp,
                            "event": "sell",
                            "price": round(bar.close, 6),
                            "reason": "signal_exit",
                        }
                    )
                    in_position = False
                    entry_price = None
                    high_water_mark = None
                    exited_this_bar = True
                    if no_same_day_reentry:
                        blocked_reentry_day = trading_day
                equity_curve.append(equity)

        can_reenter_today = not (no_same_day_reentry and blocked_reentry_day == trading_day)
        if index >= definition.lookback and not in_position and signal == 1 and not exited_this_bar and can_reenter_today:
            in_position = True
            entry_price = bar.close
            high_water_mark = max(bar.high, bar.close)
            fixed_stop = entry_price * (1 - risk_limits.stop_loss_pct)
            trailing_stop = high_water_mark * (1 - risk_limits.trailing_stop_pct)
            effective_stop = max(value for value in [fixed_stop, trailing_stop] if value is not None)
            events.append(
                {
                    "timestamp": bar.timestamp,
                    "event": "buy",
                    "price": round(bar.close, 6),
                    "reason": "signal_entry",
                }
            )

        trace_points.append(
            {
                "timestamp": bar.timestamp,
                "close": round(bar.close, 6),
                "signal": "long" if signal == 1 else "flat",
                "indicator_value": round(indicator_value, 6) if indicator_value is not None else None,
                "threshold_value": threshold_value,
                "in_position": in_position,
                "equity": round(equity, 6),
                "high_water_mark": round(high_water_mark, 6) if high_water_mark is not None else None,
                "fixed_stop_price": round(fixed_stop, 6) if fixed_stop is not None else None,
                "trailing_stop_price": round(trailing_stop, 6) if trailing_stop is not None else None,
                "effective_stop_price": round(effective_stop, 6) if effective_stop is not None else None,
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
        "strategy_family": definition.signal_type,
        "symbol": symbol,
        "market": market,
        "window_name": window_name,
        "run_type": "historical_bruteforce",
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
            "max_daily_loss_bps": risk_limits.max_daily_loss_bps,
            "max_gross_exposure": risk_limits.max_gross_exposure,
            "no_same_day_reentry": no_same_day_reentry,
        },
        "exit_reasons": dict(exit_reasons),
    }
    return {
        "summary": summary,
        "events": events,
        "trace_points": trace_points,
    }


def run_risk_managed_backtest(
    bars: list[BacktestBar],
    definition: StrategyDefinition,
    symbol: str,
    market: str,
    window_name: str,
    risk_limits: RiskLimits,
    no_same_day_reentry: bool = False,
) -> dict[str, object]:
    return build_backtest_trace(
        bars=bars,
        definition=definition,
        symbol=symbol,
        market=market,
        window_name=window_name,
        risk_limits=risk_limits,
        no_same_day_reentry=no_same_day_reentry,
    )["summary"]
