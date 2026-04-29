from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyDefinition:
    name: str
    lookback: int
    signal_type: str
    threshold: float = 0.0


def build_market_strategies(strategy_prefix: str) -> list[StrategyDefinition]:
    return [
        StrategyDefinition(
            name=f"{strategy_prefix}_momentum_daily", lookback=20, signal_type="momentum", threshold=0.02
        ),
        StrategyDefinition(
            name=f"{strategy_prefix}_mean_reversion_daily",
            lookback=5,
            signal_type="mean_reversion",
            threshold=0.03,
        ),
        StrategyDefinition(name=f"{strategy_prefix}_breakout_daily", lookback=15, signal_type="breakout"),
        StrategyDefinition(
            name=f"{strategy_prefix}_momentum_aggressive",
            lookback=7,
            signal_type="momentum",
            threshold=0.01,
        ),
    ]


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        drawdown = (value / peak) - 1 if peak > 0 else 0.0
        max_dd = min(max_dd, drawdown)
    return max_dd


def _latest_signal(prices: list[float], definition: StrategyDefinition) -> int:
    if len(prices) <= definition.lookback:
        return 0

    recent = prices[-definition.lookback :]
    current = recent[-1]
    average = sum(recent) / len(recent)
    max_recent = max(recent[:-1]) if len(recent) > 1 else current
    min_recent = min(recent[:-1]) if len(recent) > 1 else current

    if definition.signal_type == "momentum":
        return 1 if current >= average * (1 + definition.threshold) else 0
    if definition.signal_type == "mean_reversion":
        return 1 if current <= average * (1 - definition.threshold) else 0
    if definition.signal_type == "breakout":
        return 1 if current >= max_recent else 0
    if definition.signal_type == "breakdown_exit":
        return 0 if current <= min_recent else 1
    return 0


def evaluate_strategy(prices: list[float], definition: StrategyDefinition, symbol: str) -> dict[str, object]:
    if len(prices) < definition.lookback + 2:
        raise ValueError(f"Not enough bars to evaluate {definition.name}.")

    signals: list[int] = []
    trade_returns: list[float] = []
    equity = 1.0
    equity_curve = [equity]

    for index in range(definition.lookback, len(prices) - 1):
        signal = _latest_signal(prices[: index + 1], definition)
        next_return = (prices[index + 1] / prices[index]) - 1
        if signal == 1:
            trade_returns.append(next_return)
        equity *= 1 + (next_return if signal == 1 else 0.0)
        equity_curve.append(equity)
        signals.append(signal)

    latest = _latest_signal(prices, definition)
    trades = sum(1 for index in range(1, len(signals)) if signals[index] != signals[index - 1] and signals[index] == 1)
    positive = sum(1 for value in trade_returns if value > 0)
    win_rate = positive / len(trade_returns) if trade_returns else 0.0
    total_return = equity_curve[-1] - 1
    max_drawdown = _max_drawdown(equity_curve)
    score = total_return + max_drawdown * 0.5 + win_rate * 0.1

    return {
        "strategy_name": definition.name,
        "symbol": symbol,
        "run_type": "historical_compare",
        "total_return": round(total_return, 6),
        "max_drawdown": round(max_drawdown, 6),
        "trades": trades,
        "win_rate": round(win_rate, 6),
        "latest_signal": "long" if latest == 1 else "flat",
        "score": round(score, 6),
        "lookback": definition.lookback,
        "signal_type": definition.signal_type,
        "threshold": definition.threshold,
    }
