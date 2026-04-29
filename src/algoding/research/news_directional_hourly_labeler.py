from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import pstdev
from typing import Iterable
from zoneinfo import ZoneInfo

from algoding.data.news import NewsArticle
from algoding.research.historical_backtest import BacktestBar
from algoding.research.news_labeler import EVENT_KEYWORDS


@dataclass(frozen=True)
class HourlyBundleMetadata:
    symbol: str
    trading_key: str
    article_ids: tuple[int, ...]
    headline_count: int
    text: str
    session_label: str
    event_type: str


@dataclass(frozen=True)
class HourlyDirectionalNewsExample:
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
    forward_return_5h: float
    forward_excess_return_5h: float
    realized_vol_20h: float
    threshold: float


def build_hourly_bundle_metadata(
    *,
    symbol: str,
    articles: Iterable[NewsArticle],
    bar_timestamps: list[datetime],
    max_headlines_per_bar: int = 5,
    timezone_name: str = "America/New_York",
) -> list[HourlyBundleMetadata]:
    if not bar_timestamps:
        return []
    tz = ZoneInfo(timezone_name)
    sorted_bars = sorted(bar_timestamps)
    grouped: dict[str, list[NewsArticle]] = {}
    for article in sorted(articles, key=lambda item: item.created_at):
        created = datetime.fromisoformat(article.created_at).astimezone(tz)
        effective_ts = _next_bar_timestamp(created=created, bar_timestamps=sorted_bars, tz=tz)
        if effective_ts is None:
            continue
        grouped.setdefault(effective_ts.isoformat(), []).append(article)

    bundles: list[HourlyBundleMetadata] = []
    for trading_key, items in sorted(grouped.items()):
        selected = items[:max_headlines_per_bar]
        segments: list[str] = []
        sessions: list[str] = []
        for index, item in enumerate(selected, start=1):
            created = datetime.fromisoformat(item.created_at).astimezone(tz)
            sessions.append(_session_label(created))
            headline = " ".join(item.headline.strip().split())
            summary = " ".join(item.summary.strip().split())
            if summary:
                segments.append(f"{index}. {headline} Summary: {summary}")
            else:
                segments.append(f"{index}. {headline}")
        text = "\n".join(segments)
        bundles.append(
            HourlyBundleMetadata(
                symbol=symbol,
                trading_key=trading_key,
                article_ids=tuple(article.article_id for article in selected),
                headline_count=len(selected),
                text=text,
                session_label=_majority_session(sessions),
                event_type=_infer_event_type(text),
            )
        )
    return bundles


def build_hourly_directional_examples(
    *,
    symbol: str,
    bundles: list[HourlyBundleMetadata],
    bars: list[BacktestBar],
    benchmark_bars: list[BacktestBar] | None,
    horizon_bars: int = 5,
    minimum_threshold: float = 0.0025,
    volatility_multiplier: float = 0.5,
) -> list[HourlyDirectionalNewsExample]:
    closes_by_key = {bar.timestamp: float(bar.close) for bar in bars}
    keys = [bar.timestamp for bar in bars]
    benchmark_by_key = {bar.timestamp: float(bar.close) for bar in benchmark_bars} if benchmark_bars else {}
    returns = _period_returns([float(bar.close) for bar in bars])
    returns_by_key = {keys[index]: returns[index] for index in range(len(returns))}

    examples: list[HourlyDirectionalNewsExample] = []
    for bundle in bundles:
        bar_index = keys.index(bundle.trading_key) if bundle.trading_key in keys else -1
        if bar_index < 20 or bar_index + horizon_bars >= len(keys):
            continue
        start_close = closes_by_key[bundle.trading_key]
        end_close = closes_by_key[keys[bar_index + horizon_bars]]
        forward_return = (end_close / start_close) - 1.0 if start_close > 0 else 0.0
        if benchmark_by_key and bundle.trading_key in benchmark_by_key and keys[bar_index + horizon_bars] in benchmark_by_key:
            bench_start = benchmark_by_key[bundle.trading_key]
            bench_end = benchmark_by_key[keys[bar_index + horizon_bars]]
            benchmark_return = (bench_end / bench_start) - 1.0 if bench_start > 0 else 0.0
        else:
            benchmark_return = 0.0
        excess_return = forward_return - benchmark_return
        realized_vol = pstdev(
            returns_by_key[keys[idx]]
            for idx in range(max(0, bar_index - 19), bar_index + 1)
            if keys[idx] in returns_by_key
        )
        scaled_vol = realized_vol * (horizon_bars ** 0.5)
        threshold = max(minimum_threshold, scaled_vol * volatility_multiplier)
        direction, strength = _direction_from_return(forward_return=forward_return, threshold=threshold)
        relative_to_spy = _relative_to_market(excess_return=excess_return, threshold=threshold)
        prompt = build_hourly_directional_prompt(bundle)
        completion = build_hourly_directional_completion(
            direction=direction,
            strength=strength,
            relative_to_spy=relative_to_spy,
            event_type=bundle.event_type,
            session_label=bundle.session_label,
            horizon_hours=horizon_bars,
        )
        examples.append(
            HourlyDirectionalNewsExample(
                symbol=symbol,
                trading_day=bundle.trading_key,
                prompt=prompt,
                completion=completion,
                direction=direction,
                strength=strength,
                relative_to_spy=relative_to_spy,
                event_type=bundle.event_type,
                session_label=bundle.session_label,
                article_count=bundle.headline_count,
                forward_return_5h=round(forward_return, 6),
                forward_excess_return_5h=round(excess_return, 6),
                realized_vol_20h=round(realized_vol, 6),
                threshold=round(threshold, 6),
            )
        )
    return examples


def build_hourly_directional_prompt(bundle: HourlyBundleMetadata) -> str:
    return (
        "You are a financial news directional forecaster. Read the stock-specific news bundle and return compact JSON "
        "with keys direction, strength, relative_to_spy, event_type, session, horizon. "
        "direction must be bullish, bearish, or neutral. "
        "strength must be low, medium, or high. "
        "relative_to_spy must be outperform, underperform, or inline. "
        "event_type must be one of earnings, guidance, analyst, mna, regulatory, litigation, product, macro, other. "
        "session must be pre_market, intraday, post_close, or mixed. "
        "horizon must be 5h.\n\n"
        f"Symbol: {bundle.symbol}\n"
        f"Trading timestamp: {bundle.trading_key}\n"
        f"Session context: {bundle.session_label}\n"
        f"Headline count: {bundle.headline_count}\n"
        f"Headlines:\n{bundle.text}\n\n"
        "JSON:"
    )


def build_hourly_directional_completion(
    *,
    direction: str,
    strength: str,
    relative_to_spy: str,
    event_type: str,
    session_label: str,
    horizon_hours: int,
) -> str:
    return (
        "{"
        f"\"direction\":\"{direction}\","
        f"\"strength\":\"{strength}\","
        f"\"relative_to_spy\":\"{relative_to_spy}\","
        f"\"event_type\":\"{event_type}\","
        f"\"session\":\"{session_label}\","
        f"\"horizon\":\"{horizon_hours}h\""
        "}"
    )


def _next_bar_timestamp(*, created: datetime, bar_timestamps: list[datetime], tz: ZoneInfo) -> datetime | None:
    for timestamp in bar_timestamps:
        local_bar = timestamp.astimezone(tz)
        if local_bar >= created:
            return timestamp
    return None


def _session_label(created: datetime) -> str:
    if created.hour < 9 or (created.hour == 9 and created.minute < 30):
        return "pre_market"
    if created.hour >= 16:
        return "post_close"
    return "intraday"


def _majority_session(sessions: list[str]) -> str:
    if not sessions:
        return "mixed"
    counts: dict[str, int] = {}
    for value in sessions:
        counts[value] = counts.get(value, 0) + 1
    winner, winner_count = max(counts.items(), key=lambda item: item[1])
    tied = [name for name, count in counts.items() if count == winner_count]
    return winner if len(tied) == 1 else "mixed"


def _infer_event_type(text: str) -> str:
    normalized = text.lower()
    for event_type, keywords in EVENT_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return event_type
    return "other"


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


def _period_returns(prices: list[float]) -> list[float]:
    returns = [0.0]
    for index in range(1, len(prices)):
        previous = prices[index - 1]
        returns.append((prices[index] / previous) - 1.0 if previous > 0 else 0.0)
    return returns
