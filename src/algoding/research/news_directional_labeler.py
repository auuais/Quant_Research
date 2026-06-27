from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import pstdev

from algoding.research.historical_backtest import BacktestBar
from algoding.research.news_labeler import BundleMetadata


@dataclass(frozen=True)
class DirectionalNewsExample:
    symbol: str
    trading_day: str
    prompt: str
    completion: str
    direction: str
    strength: str
    relative_to_spy: str
    event_type: str
    session_label: str
    article_count: int
    forward_return_5d: float
    forward_excess_return_5d: float
    realized_vol_20d: float
    threshold: float


def build_directional_examples(
    *,
    symbol: str,
    bundles: list[BundleMetadata],
    bars: list[BacktestBar],
    benchmark_bars: list[BacktestBar] | None,
    horizon_bars: int = 5,
    minimum_threshold: float = 0.01,
    volatility_multiplier: float = 0.5,
    label_mode: str = "band",
) -> list[DirectionalNewsExample]:
    closes_by_day = {datetime.fromisoformat(bar.timestamp).date().isoformat(): float(bar.close) for bar in bars}
    benchmark_by_day = (
        {datetime.fromisoformat(bar.timestamp).date().isoformat(): float(bar.close) for bar in benchmark_bars}
        if benchmark_bars
        else {}
    )
    bars_by_day = [datetime.fromisoformat(bar.timestamp).date().isoformat() for bar in bars]
    returns = _daily_returns([float(bar.close) for bar in bars])
    daily_return_by_day = {bars_by_day[index]: returns[index] for index in range(len(returns))}

    examples: list[DirectionalNewsExample] = []
    for bundle in bundles:
        day_index = bars_by_day.index(bundle.trading_day) if bundle.trading_day in bars_by_day else -1
        if day_index < 20 or day_index + horizon_bars >= len(bars_by_day):
            continue
        start_close = closes_by_day[bundle.trading_day]
        end_close = closes_by_day[bars_by_day[day_index + horizon_bars]]
        forward_return = (end_close / start_close) - 1.0 if start_close > 0 else 0.0
        if benchmark_by_day and bundle.trading_day in benchmark_by_day and bars_by_day[day_index + horizon_bars] in benchmark_by_day:
            bench_start = benchmark_by_day[bundle.trading_day]
            bench_end = benchmark_by_day[bars_by_day[day_index + horizon_bars]]
            benchmark_return = (bench_end / bench_start) - 1.0 if bench_start > 0 else 0.0
        else:
            benchmark_return = 0.0
        excess_return = forward_return - benchmark_return
        realized_vol = pstdev(
            daily_return_by_day[bars_by_day[idx]]
            for idx in range(max(0, day_index - 19), day_index + 1)
            if bars_by_day[idx] in daily_return_by_day
        )
        scaled_vol = realized_vol * (horizon_bars ** 0.5)
        threshold = max(minimum_threshold, scaled_vol * volatility_multiplier)
        # Strength is taken from the realized-move magnitude either way; only the direction
        # rule differs. Triple-barrier labels by the first barrier touched along the path
        # (Lopez de Prado), which is closer to how a stop/target trade actually resolves.
        _, strength = _direction_from_return(forward_return=forward_return, threshold=threshold)
        if label_mode == "triple_barrier":
            direction = _triple_barrier_direction(
                bars=bars, day_index=day_index, horizon=horizon_bars, entry=start_close, threshold=threshold
            )
        else:
            direction, strength = _direction_from_return(forward_return=forward_return, threshold=threshold)
        relative_to_spy = _relative_to_market(excess_return=excess_return, threshold=threshold)
        prompt = build_directional_prompt(bundle)
        completion = build_directional_completion(
            direction=direction,
            strength=strength,
            relative_to_spy=relative_to_spy,
            event_type=bundle.event_type,
            session_label=bundle.session_label,
            horizon_days=horizon_bars,
        )
        examples.append(
            DirectionalNewsExample(
                symbol=symbol,
                trading_day=bundle.trading_day,
                prompt=prompt,
                completion=completion,
                direction=direction,
                strength=strength,
                relative_to_spy=relative_to_spy,
                event_type=bundle.event_type,
                session_label=bundle.session_label,
                article_count=bundle.headline_count,
                forward_return_5d=round(forward_return, 6),
                forward_excess_return_5d=round(excess_return, 6),
                realized_vol_20d=round(realized_vol, 6),
                threshold=round(threshold, 6),
            )
        )
    return examples


def build_directional_prompt(bundle: BundleMetadata) -> str:
    return (
        "You are a financial news directional forecaster. Read the stock-specific news bundle and return compact JSON "
        "with keys direction, strength, relative_to_spy, event_type, session, horizon. "
        "direction must be bullish, bearish, or neutral. "
        "strength must be low, medium, or high. "
        "relative_to_spy must be outperform, underperform, or inline. "
        "event_type must be one of earnings, guidance, analyst, mna, regulatory, litigation, product, macro, other. "
        "session must be pre_market, intraday, post_close, or mixed. "
        "horizon must be 5d.\n\n"
        f"Symbol: {bundle.symbol}\n"
        f"Trading day: {bundle.trading_day}\n"
        f"Session context: {bundle.session_label}\n"
        f"Headline count: {bundle.headline_count}\n"
        f"Headlines:\n{bundle.text}\n\n"
        "JSON:"
    )


def build_directional_completion(
    *,
    direction: str,
    strength: str,
    relative_to_spy: str,
    event_type: str,
    session_label: str,
    horizon_days: int,
) -> str:
    return (
        "{"
        f"\"direction\":\"{direction}\","
        f"\"strength\":\"{strength}\","
        f"\"relative_to_spy\":\"{relative_to_spy}\","
        f"\"event_type\":\"{event_type}\","
        f"\"session\":\"{session_label}\","
        f"\"horizon\":\"{horizon_days}d\""
        "}"
    )


def _triple_barrier_direction(
    *, bars: list[BacktestBar], day_index: int, horizon: int, entry: float, threshold: float
) -> str:
    if entry <= 0:
        return "neutral"
    upper = entry * (1 + threshold)
    lower = entry * (1 - threshold)
    for offset in range(1, horizon + 1):
        bar = bars[day_index + offset]
        touch_up = float(bar.high) >= upper
        touch_down = float(bar.low) <= lower
        if touch_up and touch_down:
            return "bullish" if float(bar.close) >= entry else "bearish"
        if touch_up:
            return "bullish"
        if touch_down:
            return "bearish"
    return "neutral"


def _direction_from_return(*, forward_return: float, threshold: float) -> tuple[str, str]:
    magnitude = abs(forward_return)
    if forward_return >= threshold:
        direction = "bullish"
    elif forward_return <= -threshold:
        direction = "bearish"
    else:
        direction = "neutral"
    if magnitude >= threshold * 2.5:
        strength = "high"
    elif magnitude >= threshold * 1.5:
        strength = "medium"
    else:
        strength = "low"
    return direction, strength


def _relative_to_market(*, excess_return: float, threshold: float) -> str:
    if excess_return >= threshold:
        return "outperform"
    if excess_return <= -threshold:
        return "underperform"
    return "inline"


def _daily_returns(prices: list[float]) -> list[float]:
    returns = [0.0]
    for index in range(1, len(prices)):
        previous = prices[index - 1]
        returns.append((prices[index] / previous) - 1.0 if previous > 0 else 0.0)
    return returns
