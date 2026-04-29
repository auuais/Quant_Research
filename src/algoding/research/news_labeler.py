from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time
from statistics import pstdev
from typing import Iterable
from zoneinfo import ZoneInfo

from algoding.data.news import NewsArticle
from algoding.research.historical_backtest import BacktestBar


EVENT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("earnings", ("earnings", "revenue", "eps", "profit", "sales")),
    ("guidance", ("guidance", "forecast", "outlook", "raises outlook", "cuts outlook")),
    ("analyst", ("analyst", "price target", "upgrade", "downgrade", "rating")),
    ("mna", ("acquire", "acquisition", "merger", "buyout", "stake")),
    ("regulatory", ("regulator", "regulatory", "antitrust", "probe", "investigation", "approval")),
    ("litigation", ("lawsuit", "court", "settlement", "sues", "legal")),
    ("product", ("launch", "product", "chip", "platform", "service", "ai")),
    ("macro", ("inflation", "fed", "rates", "tariff", "economy", "opec", "oil")),
]


@dataclass(frozen=True)
class BundleMetadata:
    symbol: str
    trading_day: str
    article_ids: tuple[int, ...]
    headline_count: int
    text: str
    session_label: str
    session_pre_market_count: int
    session_intraday_count: int
    session_post_close_count: int
    event_type: str


@dataclass(frozen=True)
class LabeledNewsExample:
    symbol: str
    trading_day: str
    prompt: str
    completion: str
    label: str
    strength: str
    event_type: str
    session_label: str
    article_count: int
    forward_return_5d: float
    forward_excess_return_5d: float
    realized_vol_20d: float
    threshold: float


def build_bundle_metadata(
    *,
    symbol: str,
    articles: Iterable[NewsArticle],
    trading_days: list[date],
    max_headlines_per_day: int = 5,
    timezone_name: str = "America/New_York",
    market_open_hour: int = 9,
    market_open_minute: int = 30,
    market_close_hour: int = 16,
) -> list[BundleMetadata]:
    tz = ZoneInfo(timezone_name)
    if not trading_days:
        return []
    trading_days_sorted = sorted(trading_days)
    grouped: dict[str, list[NewsArticle]] = {}
    for article in sorted(articles, key=lambda item: item.created_at):
        created = datetime.fromisoformat(article.created_at).astimezone(tz)
        effective_day = _effective_trading_day(
            created=created,
            trading_days=trading_days_sorted,
            market_close_hour=market_close_hour,
        )
        if effective_day is None:
            continue
        grouped.setdefault(effective_day.isoformat(), []).append(article)

    bundles: list[BundleMetadata] = []
    market_open = time(hour=market_open_hour, minute=market_open_minute)
    market_close = time(hour=market_close_hour, minute=0)
    for trading_day, items in sorted(grouped.items()):
        selected = items[:max_headlines_per_day]
        pre_market = 0
        intraday = 0
        post_close = 0
        segments: list[str] = []
        for index, item in enumerate(selected, start=1):
            created = datetime.fromisoformat(item.created_at).astimezone(tz)
            effective = date.fromisoformat(trading_day)
            session = _session_label_for_article(
                created=created,
                effective_day=effective,
                market_open=market_open,
                market_close=market_close,
            )
            if session == "pre_market":
                pre_market += 1
            elif session == "post_close":
                post_close += 1
            else:
                intraday += 1
            headline = re.sub(r"\s+", " ", item.headline.strip())
            summary = re.sub(r"\s+", " ", item.summary.strip())
            if summary:
                segments.append(f"{index}. {headline} Summary: {summary}")
            else:
                segments.append(f"{index}. {headline}")
        text = "\n".join(segments)
        bundles.append(
            BundleMetadata(
                symbol=symbol,
                trading_day=trading_day,
                article_ids=tuple(article.article_id for article in selected),
                headline_count=len(selected),
                text=text,
                session_label=_majority_session(pre_market, intraday, post_close),
                session_pre_market_count=pre_market,
                session_intraday_count=intraday,
                session_post_close_count=post_close,
                event_type=_infer_event_type(text),
            )
        )
    return bundles


def build_labeled_examples(
    *,
    symbol: str,
    bundles: list[BundleMetadata],
    bars: list[BacktestBar],
    benchmark_bars: list[BacktestBar] | None,
    horizon_bars: int = 5,
    minimum_threshold: float = 0.01,
    volatility_multiplier: float = 0.5,
) -> list[LabeledNewsExample]:
    closes_by_day = {datetime.fromisoformat(bar.timestamp).date().isoformat(): float(bar.close) for bar in bars}
    benchmark_by_day = (
        {datetime.fromisoformat(bar.timestamp).date().isoformat(): float(bar.close) for bar in benchmark_bars}
        if benchmark_bars
        else {}
    )
    bars_by_day = [datetime.fromisoformat(bar.timestamp).date().isoformat() for bar in bars]
    returns = _daily_returns([float(bar.close) for bar in bars])
    daily_return_by_day = {bars_by_day[index]: returns[index] for index in range(len(returns))}

    examples: list[LabeledNewsExample] = []
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
        scaled_vol = realized_vol * (horizon_bars**0.5)
        threshold = max(minimum_threshold, scaled_vol * volatility_multiplier)
        label, strength = _label_from_excess_return(excess_return=excess_return, threshold=threshold)
        prompt = build_structured_prompt(bundle)
        completion = build_structured_completion(
            label=label,
            strength=strength,
            event_type=bundle.event_type,
            session_label=bundle.session_label,
            horizon_days=horizon_bars,
        )
        examples.append(
            LabeledNewsExample(
                symbol=symbol,
                trading_day=bundle.trading_day,
                prompt=prompt,
                completion=completion,
                label=label,
                strength=strength,
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


def build_structured_prompt(bundle: BundleMetadata) -> str:
    return (
        "You are a financial news analyst. Read the stock-specific news bundle and return compact JSON "
        "with keys label, strength, event_type, session, horizon. "
        "label must be bullish, bearish, or neutral. "
        "strength must be low, medium, or high. "
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


def build_structured_completion(
    *,
    label: str,
    strength: str,
    event_type: str,
    session_label: str,
    horizon_days: int,
) -> str:
    return (
        "{"
        f"\"label\":\"{label}\","
        f"\"strength\":\"{strength}\","
        f"\"event_type\":\"{event_type}\","
        f"\"session\":\"{session_label}\","
        f"\"horizon\":\"{horizon_days}d\""
        "}"
    )


def _label_from_excess_return(*, excess_return: float, threshold: float) -> tuple[str, str]:
    magnitude = abs(excess_return)
    if excess_return >= threshold:
        label = "bullish"
    elif excess_return <= -threshold:
        label = "bearish"
    else:
        label = "neutral"
    if magnitude >= threshold * 2.5:
        strength = "high"
    elif magnitude >= threshold * 1.5:
        strength = "medium"
    else:
        strength = "low"
    return label, strength


def _daily_returns(prices: list[float]) -> list[float]:
    returns = [0.0]
    for index in range(1, len(prices)):
        previous = prices[index - 1]
        returns.append((prices[index] / previous) - 1.0 if previous > 0 else 0.0)
    return returns


def _session_label_for_article(
    *,
    created: datetime,
    effective_day: date,
    market_open: time,
    market_close: time,
) -> str:
    if created.date() < effective_day:
        return "post_close"
    created_time = created.timetz().replace(tzinfo=None)
    if created_time < market_open:
        return "pre_market"
    if created_time >= market_close:
        return "post_close"
    return "intraday"


def _majority_session(pre_market: int, intraday: int, post_close: int) -> str:
    counts = {
        "pre_market": pre_market,
        "intraday": intraday,
        "post_close": post_close,
    }
    winner, winner_count = max(counts.items(), key=lambda item: item[1])
    tied = [name for name, count in counts.items() if count == winner_count and count > 0]
    return winner if len(tied) == 1 else "mixed"


def _infer_event_type(text: str) -> str:
    normalized = text.lower()
    for event_type, keywords in EVENT_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return event_type
    return "other"


def _effective_trading_day(*, created: datetime, trading_days: list[date], market_close_hour: int) -> date | None:
    article_day = created.date()
    for trading_day in trading_days:
        if trading_day < article_day:
            continue
        if trading_day == article_day and created.hour < market_close_hour:
            return trading_day
        if trading_day > article_day:
            return trading_day
    return None
