"""Generic event-study harness: CAR profiles, bootstrap CIs, and split-sample confirmation.

Shared by the D2 premia battery and (later) D6 disclosure events. The confirmation rule is deliberately
strict: an effect counts only if it is significant in the early *and* late subperiod independently and
is larger than the cost of trading it. That is what stops a battery of hypotheses from manufacturing
winners by chance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


DEFAULT_BOOTSTRAP = 10_000
DEFAULT_SPLIT = 0.70


@dataclass(frozen=True)
class SubperiodStats:
    label: str
    n: int
    mean: float
    ci_low: float
    ci_high: float
    excludes_zero: bool
    positive_fraction: float

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "n": self.n,
            "mean": round(self.mean, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "excludes_zero": self.excludes_zero,
            "positive_fraction": round(self.positive_fraction, 4),
        }


@dataclass(frozen=True)
class EffectTest:
    name: str
    description: str
    round_trip_cost: float
    full: SubperiodStats
    early: SubperiodStats
    late: SubperiodStats
    annualized_estimate: float | None = None
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def confirmed_both_subperiods(self) -> bool:
        return self.early.excludes_zero and self.late.excludes_zero and (
            np.sign(self.early.mean) == np.sign(self.late.mean)
        )

    @property
    def exceeds_cost_hurdle(self) -> bool:
        return abs(self.full.mean) >= 2.0 * self.round_trip_cost

    @property
    def survives(self) -> bool:
        return self.confirmed_both_subperiods and self.exceeds_cost_hurdle

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "round_trip_cost": self.round_trip_cost,
            "full": self.full.as_dict(),
            "early": self.early.as_dict(),
            "late": self.late.as_dict(),
            "annualized_estimate": round(self.annualized_estimate, 6)
            if self.annualized_estimate is not None
            else None,
            "confirmed_both_subperiods": self.confirmed_both_subperiods,
            "exceeds_cost_hurdle": self.exceeds_cost_hurdle,
            "survives": self.survives,
            "verdict": "SURVIVES" if self.survives else "DROPPED",
            "drop_reason": None
            if self.survives
            else "failed split-sample confirmation"
            if not self.confirmed_both_subperiods
            else "effect smaller than 2x round-trip cost",
            **self.extras,
        }


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float]:
    values = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if len(values) < 3:
        return (float("nan"), float("nan"))
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return (
        float(np.quantile(draws, alpha / 2.0)),
        float(np.quantile(draws, 1.0 - alpha / 2.0)),
    )


def _stats(label: str, values: pd.Series, *, seed: int) -> SubperiodStats:
    clean = values.dropna()
    array = clean.to_numpy(dtype=float)
    low, high = bootstrap_mean_ci(array, seed=seed)
    excludes_zero = bool(np.isfinite(low) and np.isfinite(high) and (low > 0 or high < 0))
    return SubperiodStats(
        label=label,
        n=int(len(array)),
        mean=float(array.mean()) if len(array) else float("nan"),
        ci_low=low,
        ci_high=high,
        excludes_zero=excludes_zero,
        positive_fraction=float((array > 0).mean()) if len(array) else float("nan"),
    )


def test_effect(
    name: str,
    description: str,
    event_returns: pd.Series,
    *,
    round_trip_cost: float,
    split: float = DEFAULT_SPLIT,
    events_per_year: float | None = None,
    seed: int = 7,
    extras: dict[str, object] | None = None,
) -> EffectTest:
    """Test one hypothesis: is the mean event return non-zero in both halves and bigger than costs?"""
    ordered = event_returns.dropna().sort_index()
    if len(ordered) < 10:
        empty = _stats("insufficient", ordered, seed=seed)
        return EffectTest(
            name=name,
            description=description,
            round_trip_cost=round_trip_cost,
            full=empty,
            early=empty,
            late=empty,
            extras={**(extras or {}), "note": "fewer than 10 events; not testable"},
        )
    cut = int(len(ordered) * split)
    annualized = None
    if events_per_year:
        annualized = float((1.0 + ordered.mean()) ** events_per_year - 1.0)
    return EffectTest(
        name=name,
        description=description,
        round_trip_cost=round_trip_cost,
        full=_stats("full", ordered, seed=seed),
        early=_stats(f"early_{int(split * 100)}pct", ordered.iloc[:cut], seed=seed + 1),
        late=_stats(f"late_{int((1 - split) * 100)}pct", ordered.iloc[cut:], seed=seed + 2),
        annualized_estimate=annualized,
        extras=extras or {},
    )


def car_profile(
    daily_returns: pd.Series,
    event_dates: pd.DatetimeIndex,
    *,
    pre: int = 5,
    post: int = 10,
) -> dict[str, object]:
    """Average cumulative return path around events, indexed by trading days from the event."""
    series = daily_returns.dropna()
    calendar = series.index
    position_of = {stamp: index for index, stamp in enumerate(calendar)}
    offsets = list(range(-pre, post + 1))
    buckets: dict[int, list[float]] = {offset: [] for offset in offsets}
    used = 0
    for stamp in event_dates:
        anchor = position_of.get(stamp)
        if anchor is None:
            candidates = calendar[calendar >= stamp]
            if len(candidates) == 0:
                continue
            anchor = position_of[candidates[0]]
        if anchor - pre < 0 or anchor + post >= len(calendar):
            continue
        used += 1
        for offset in offsets:
            buckets[offset].append(float(series.iloc[anchor + offset]))
    means = {offset: float(np.mean(values)) if values else float("nan") for offset, values in buckets.items()}
    cumulative: dict[int, float] = {}
    running = 0.0
    for offset in offsets:
        running += means[offset] if np.isfinite(means[offset]) else 0.0
        cumulative[offset] = running
    return {
        "events_used": used,
        "offsets": offsets,
        "mean_daily_return_by_offset": {str(k): round(v, 6) for k, v in means.items()},
        "cumulative_return_by_offset": {str(k): round(v, 6) for k, v in cumulative.items()},
    }


def deflated_sharpe_hurdle(*, trials: int, observations: int, periods_per_year: float = 252.0) -> dict[str, object]:
    """Expected best-of-N Sharpe under a zero-edge null, for battery-wide multiple-testing context."""
    from math import e, sqrt
    from statistics import NormalDist

    euler_gamma = 0.5772156649
    normal = NormalDist()
    standard_error = sqrt(periods_per_year / max(1, observations))
    expected_max_z = (1 - euler_gamma) * normal.inv_cdf(1 - 1 / max(2, trials)) + euler_gamma * normal.inv_cdf(
        1 - 1 / (max(2, trials) * e)
    )
    return {
        "trials": trials,
        "observations": observations,
        "sharpe_standard_error": round(standard_error, 4),
        "expected_max_sharpe_under_null": round(expected_max_z * standard_error, 4),
    }
