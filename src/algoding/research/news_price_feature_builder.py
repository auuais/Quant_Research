from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import sqrt

from algoding.research.historical_backtest import BacktestBar


@dataclass(frozen=True)
class FeatureRow:
    symbol: str
    trading_day: str
    split: str
    features: dict[str, float]
    target_long: int
    forward_excess_return_5d: float


def build_feature_rows(
    *,
    symbol: str,
    bars: list[BacktestBar],
    benchmark_bars: list[BacktestBar] | None,
    llm_outputs_by_day: dict[str, dict[str, object]],
    split_by_day: dict[str, str],
    horizon_bars: int = 5,
    target_threshold: float = 0.01,
) -> list[FeatureRow]:
    trading_days = [datetime.fromisoformat(bar.timestamp).date().isoformat() for bar in bars]
    closes = [float(bar.close) for bar in bars]
    volumes = [float(bar.volume or 0.0) for bar in bars]
    benchmark_by_day = (
        {datetime.fromisoformat(bar.timestamp).date().isoformat(): float(bar.close) for bar in benchmark_bars}
        if benchmark_bars
        else {}
    )
    rows: list[FeatureRow] = []
    for index in range(20, len(bars) - horizon_bars):
        trading_day = trading_days[index]
        split = split_by_day.get(trading_day)
        if split is None:
            continue
        current_close = closes[index]
        future_close = closes[index + horizon_bars]
        forward_return = (future_close / current_close) - 1.0 if current_close > 0 else 0.0
        if benchmark_by_day and trading_day in benchmark_by_day and trading_days[index + horizon_bars] in benchmark_by_day:
            bench_start = benchmark_by_day[trading_day]
            bench_end = benchmark_by_day[trading_days[index + horizon_bars]]
            benchmark_return = (bench_end / bench_start) - 1.0 if bench_start > 0 else 0.0
        else:
            benchmark_return = 0.0
        excess_return = forward_return - benchmark_return

        llm_payload = llm_outputs_by_day.get(
            trading_day,
            {
                "label": "neutral",
                "strength": "low",
                "event_type": "other",
                "session": "mixed",
            },
        )
        features = _build_numeric_features(
            symbol=symbol,
            closes=closes,
            volumes=volumes,
            index=index,
            llm_payload=llm_payload,
        )
        rows.append(
            FeatureRow(
                symbol=symbol,
                trading_day=trading_day,
                split=split,
                features=features,
                target_long=1 if excess_return >= target_threshold else 0,
                forward_excess_return_5d=round(excess_return, 6),
            )
        )
    return rows


def _build_numeric_features(
    *,
    symbol: str,
    closes: list[float],
    volumes: list[float],
    index: int,
    llm_payload: dict[str, object],
) -> dict[str, float]:
    recent_20 = closes[index - 19 : index + 1]
    recent_10 = closes[index - 9 : index + 1]
    returns = [0.0]
    for offset in range(index - 19, index + 1):
        if offset <= 0:
            returns.append(0.0)
        else:
            previous = closes[offset - 1]
            returns.append((closes[offset] / previous) - 1.0 if previous > 0 else 0.0)
    vol20 = _stdev(returns[-20:])
    average_20 = sum(recent_20) / len(recent_20)
    average_10 = sum(recent_10) / len(recent_10)
    volume_recent = volumes[index - 19 : index + 1]
    volume_mean = sum(volume_recent) / len(volume_recent) if volume_recent else 0.0
    volume_std = _stdev(volume_recent)
    current_volume = volumes[index]

    label_score = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}.get(str(llm_payload.get("label", "neutral")), 0.0)
    strength_score = {"low": 1.0, "medium": 2.0, "high": 3.0}.get(str(llm_payload.get("strength", "low")), 1.0)
    event_type = str(llm_payload.get("event_type", "other"))
    session = str(llm_payload.get("session", "mixed"))
    features = {
        "ret_1": _return(closes, index, 1),
        "ret_3": _return(closes, index, 3),
        "ret_5": _return(closes, index, 5),
        "ret_10": _return(closes, index, 10),
        "ret_20": _return(closes, index, 20),
        "dist_sma_10": (closes[index] / average_10) - 1.0 if average_10 > 0 else 0.0,
        "dist_sma_20": (closes[index] / average_20) - 1.0 if average_20 > 0 else 0.0,
        "vol_20": vol20,
        "volume_z": ((current_volume - volume_mean) / volume_std) if volume_std > 0 else 0.0,
        "llm_label_score": label_score,
        "llm_strength_score": strength_score,
        "llm_signed_strength": label_score * strength_score,
    }
    for known_symbol in ["AVGO", "NVDA", "TSLA", "GOOGL", "XOM"]:
        features[f"symbol_{known_symbol.lower()}"] = 1.0 if symbol == known_symbol else 0.0
    for known_event in ["earnings", "guidance", "analyst", "mna", "regulatory", "litigation", "product", "macro", "other"]:
        features[f"event_{known_event}"] = 1.0 if event_type == known_event else 0.0
    for known_session in ["pre_market", "intraday", "post_close", "mixed"]:
        features[f"session_{known_session}"] = 1.0 if session == known_session else 0.0
    return features


def _return(prices: list[float], index: int, lookback: int) -> float:
    previous_index = max(0, index - lookback)
    previous = prices[previous_index]
    current = prices[index]
    return (current / previous) - 1.0 if previous > 0 else 0.0


def _stdev(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return sqrt(variance)
