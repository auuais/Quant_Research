"""VOL-CARRY-V1: volatility risk premium / VIX term-structure sleeve (frontier direction D1).

Pre-registered in reports/research/frontier_hypotheses.md before this ran: 12 variants
(3 slope thresholds x VRP gate on/off x long-vol leg on/off), gate is net Sharpe >= 1.0,
>= 70% profitable windows, standalone worst DD <= 25%, and beating always-short-vol on
drawdown in the 2018-02 and 2020-03 slices.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import numpy as np
import pandas as pd

from algoding.data.volatility import VolatilityDataClient
from algoding.research.vol_signals import build_signals


TRADING_DAYS = 252.0
PRIMARY_COST_BPS = 3.0
STRESS_COST_BPS = 10.0
SLOPE_THRESHOLDS = (0.0, 0.02, 0.05)
EULER_GAMMA = 0.5772156649
# VIX cash settles 16:15 ET, the ETP trades the 16:00 close: a lag-1 decision would peek 15 minutes ahead.
HEADLINE_SIGNAL_LAG = 2

CRISIS_SLICES = {
    "aug_2015_vol_spike": ("2015-08-17", "2015-09-01"),
    "brexit_2016": ("2016-06-22", "2016-06-30"),
    "volmageddon_2018_02": ("2018-02-01", "2018-02-12"),
    "covid_2020_03": ("2020-02-20", "2020-03-23"),
    "bear_2022_h1": ("2022-01-01", "2022-06-30"),
    "yen_unwind_2024_08": ("2024-07-31", "2024-08-08"),
    "tariff_2025_04": ("2025-04-01", "2025-04-10"),
}


@dataclass(frozen=True)
class VariantSpec:
    name: str
    slope_threshold: float
    vrp_gate: bool
    long_vol_leg: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "slope_threshold": self.slope_threshold,
            "vrp_gate": self.vrp_gate,
            "long_vol_leg": self.long_vol_leg,
        }


def pre_registered_variants() -> list[VariantSpec]:
    variants: list[VariantSpec] = []
    for threshold, vrp_gate, long_leg in product(SLOPE_THRESHOLDS, (False, True), (False, True)):
        suffix = f"slope_{int(threshold * 100):02d}"
        if vrp_gate:
            suffix += "_vrp"
        if long_leg:
            suffix += "_longvol"
        variants.append(
            VariantSpec(name=suffix, slope_threshold=threshold, vrp_gate=vrp_gate, long_vol_leg=long_leg)
        )
    return variants


class VolCarryResearchLab:
    def __init__(self, settings=None) -> None:
        self._settings = settings
        self._client = VolatilityDataClient()

    def run(
        self,
        *,
        output_root: str = "reports/research/vol_carry_v1",
        window_days: int = 126,
        window_count: int = 5,
        portfolio_cap: float = 0.05,
    ) -> dict[str, object]:
        output_dir = Path(output_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        panel = self._client.build_panel()
        frame = panel.frame
        signals, signal_diagnostics = build_signals(frame)

        merged = frame.join(signals, how="left")
        required = ["slope_3m", "vrp", "short_vol_return", "cash_return", "spy_return"]
        evaluable = merged.dropna(subset=required)
        if len(evaluable) < window_days * 2:
            raise RuntimeError("not enough overlapping history to evaluate the vol sleeve")

        variants = pre_registered_variants()
        strategies: dict[str, pd.Series] = {}
        exposures: dict[str, dict[str, object]] = {}
        lookahead_strategies: dict[str, pd.Series] = {}
        for variant in variants:
            for cost_name, cost_bps in (("primary", PRIMARY_COST_BPS), ("stress", STRESS_COST_BPS)):
                returns, exposure = _run_variant(
                    evaluable, variant, cost_bps=cost_bps, signal_lag=HEADLINE_SIGNAL_LAG
                )
                strategies[f"{variant.name}|{cost_name}"] = returns
                if cost_name == "primary":
                    exposures[variant.name] = exposure
            optimistic, _ = _run_variant(evaluable, variant, cost_bps=PRIMARY_COST_BPS, signal_lag=1)
            lookahead_strategies[variant.name] = optimistic
        for cost_name, cost_bps in (("primary", PRIMARY_COST_BPS), ("stress", STRESS_COST_BPS)):
            for label, returns in _baselines(evaluable, cost_bps=cost_bps).items():
                strategies[f"{label}|{cost_name}"] = returns

        windows = _validation_windows(list(evaluable.index), window_days=window_days, window_count=window_count)
        runs: list[dict[str, object]] = []
        for key, returns in strategies.items():
            name, cost_name = key.split("|")
            for window in windows:
                sliced = returns.loc[window["start"] : window["end"]]
                if len(sliced) < 20:
                    continue
                runs.append(
                    {
                        "strategy": name,
                        "cost_profile": cost_name,
                        "window": window["name"],
                        "window_range": f"{window['start']} -> {window['end']} ({len(sliced)} days)",
                        "metrics": _risk_metrics(sliced),
                    }
                )

        aggregates = _aggregate(runs)
        crisis = {
            key.split("|")[0]: _crisis_table(returns)
            for key, returns in strategies.items()
            if key.endswith("|primary")
        }
        era_split = {
            key.split("|")[0]: _era_split(evaluable, returns)
            for key, returns in strategies.items()
            if key.endswith("|primary")
        }
        full_history = {
            key.split("|")[0]: _risk_metrics(returns)
            for key, returns in strategies.items()
            if key.endswith("|primary")
        }
        gap_scenario = {
            name: _overnight_gap_scenario(evaluable, strategies[f"{name}|primary"], exposures.get(name, {}))
            for name in [variant.name for variant in variants]
        }
        multiple_testing = _multiple_testing_hurdle(
            trials=len(variants), observations=int(len(evaluable))
        )
        lookahead_sensitivity = _lookahead_sensitivity(strategies, lookahead_strategies, variants)
        decision = _gate_decision(
            aggregates=aggregates,
            crisis=crisis,
            full_history=full_history,
            variant_names=[variant.name for variant in variants],
            multiple_testing=multiple_testing,
        )
        robustness = _svix_robustness(merged, decision, cost_bps=PRIMARY_COST_BPS)

        result: dict[str, object] = {
            "version": "VOL-CARRY-V1",
            "direction": "D1",
            "thesis": (
                "the VIX term-structure slope plus a realized-vol forecast times short-volatility ETP "
                "exposure well enough to beat both always-short-vol and cash after costs, including crises"
            ),
            "execution": (
                "signals known at T close; position held for the T+1 close-to-close return of the real ETP; "
                "turnover charged at the VIX-ETP cost profile"
            ),
            "pre_registration": {
                "registry": "reports/research/frontier_hypotheses.md#d1",
                "declared_variants": len(variants),
                "realized_variants": len(variants),
                "gate": (
                    "net Sharpe >= 1.0, >= 70% profitable windows, standalone worst DD <= 25%, "
                    "and beats always-short-vol on drawdown in the 2018-02 and 2020-03 slices"
                ),
                "methodology_amendment_2026_07_28": (
                    "Pre-registration said the tradeable leg would be reconstructed from ETP returns. During "
                    "implementation the reconstruction was tested and rejected: a synthetic -1x series rebuilt "
                    "from VXX loses 48.7% over Volmageddon while the real -1x product lost 90.4%, because VIX "
                    "futures moved between the 16:00 equity close and 16:15 futures settlement. The backtest "
                    "therefore uses the real closes of the product actually holdable (SVXY throughout, whose "
                    "target exposure genuinely fell from -1x to -0.5x on 2018-02-28). This tightens the "
                    "methodology and was decided before any strategy results were examined."
                ),
            },
            "cost_profiles": {"primary_bps_per_side": PRIMARY_COST_BPS, "stress_bps_per_side": STRESS_COST_BPS},
            "config": {
                "window_days": window_days,
                "window_count": window_count,
                "portfolio_cap": portfolio_cap,
                "evaluation_start": evaluable.index.min().date().isoformat(),
                "evaluation_end": evaluable.index.max().date().isoformat(),
                "evaluation_days": int(len(evaluable)),
                "common_start_reason": (
                    "all variants share one start date so the grid is comparable; the VRP gate needs "
                    "504 training rows plus a 21-day horizon before its first HAR-RV forecast"
                ),
                "headline_signal_lag": HEADLINE_SIGNAL_LAG,
                "signal_lag_reason": (
                    "VIX cash indices settle at 16:15 ET but the ETP trades the 16:00 close, so the usual "
                    "lag-1 convention would embed up to 15 minutes of look-ahead exactly where Feb-2018's "
                    "damage occurred; headline numbers therefore use a fully settled prior-day signal"
                ),
                "window_construction": (
                    "non-overlapping windows tiled across the entire sample so every crisis regime is "
                    "inside some window, plus a recent-252-day window and the full history"
                ),
            },
            "data_provenance": panel.provenance,
            "signal_diagnostics": signal_diagnostics,
            "variants": [variant.as_dict() for variant in variants],
            "runs": runs,
            "aggregates": aggregates,
            "full_history": full_history,
            "era_split": era_split,
            "crisis_slices": {"definitions": CRISIS_SLICES, "by_strategy": crisis},
            "overnight_gap_scenario": gap_scenario,
            "exposures": exposures,
            "multiple_testing": multiple_testing,
            "lookahead_sensitivity": lookahead_sensitivity,
            "portfolio_view": _portfolio_view(full_history, portfolio_cap),
            "decision": decision,
            "robustness_svix": robustness,
        }
        (output_dir / "vol_carry_v1_run.json").write_text(
            json.dumps(_jsonable(result), indent=2), encoding="utf-8"
        )
        (output_dir / "vol_carry_v1_summary.md").write_text(_summary_md(result), encoding="utf-8")
        _write_charts(output_dir, evaluable, strategies, decision)
        return result


def _target_weights(row: pd.Series, variant: VariantSpec) -> tuple[float, float]:
    """(short_vol_weight, long_vol_weight) decided from information known at this row's close."""
    slope = float(row["slope_3m"])
    if slope > variant.slope_threshold:
        if variant.vrp_gate and not bool(row.get("vrp", 0.0) > 0):
            return 0.0, 0.0
        return 1.0, 0.0
    if variant.long_vol_leg and slope < 0.0 and float(row.get("vix_trend_5d", 0.0) or 0.0) > 0.0:
        return 0.0, 1.0
    return 0.0, 0.0


def _run_variant(
    frame: pd.DataFrame,
    variant: VariantSpec,
    *,
    cost_bps: float,
    signal_lag: int = 1,
) -> tuple[pd.Series, dict[str, object]]:
    """Backtest one variant.

    `signal_lag` is the number of closes between the signal and the return it earns. Lag 1 is the
    normal convention (decide at T close, earn T+1) but the VIX cash indices settle at 16:15 ET while
    the ETP trades at the 16:00 close, so lag 1 embeds up to 15 minutes of look-ahead -- which lands
    exactly on the Feb-2018 blow-up. Lag 2 uses a fully settled prior-day signal and is look-ahead
    free; both are reported and the conservative one is the headline.
    """
    short_weight = 0.0
    long_weight = 0.0
    net_returns: list[float] = []
    index: list[pd.Timestamp] = []
    turnovers: list[float] = []
    state_counts = {"short_vol": 0, "long_vol": 0, "cash": 0}
    has_long_leg = "long_vol_return" in frame

    rows = list(frame.itertuples(index=True))
    for position, row in enumerate(rows):
        stamp = row.Index
        short_return = float(getattr(row, "short_vol_return", 0.0) or 0.0)
        long_return = float(getattr(row, "long_vol_return", 0.0) or 0.0) if has_long_leg else 0.0
        cash_return = float(getattr(row, "cash_return", 0.0) or 0.0)
        gross = (
            short_weight * short_return
            + long_weight * long_return
            + max(0.0, 1.0 - short_weight - long_weight) * cash_return
        )

        signal_position = position - (signal_lag - 1)
        if position < len(rows) - 1 and signal_position >= 0:
            row_series = pd.Series(rows[signal_position]._asdict())
            target_short, target_long = _target_weights(row_series, variant)
        elif signal_position < 0:
            target_short, target_long = 0.0, 0.0
        else:
            target_short, target_long = short_weight, long_weight
        turnover = abs(target_short - short_weight) + abs(target_long - long_weight)
        cost = turnover * (cost_bps / 10_000.0)
        if turnover > 0:
            turnovers.append(turnover)
        short_weight, long_weight = target_short, target_long
        state_counts[
            "short_vol" if short_weight > 0 else "long_vol" if long_weight > 0 else "cash"
        ] += 1

        net_returns.append(gross - cost)
        index.append(stamp)

    series = pd.Series(net_returns, index=pd.DatetimeIndex(index), name=variant.name)
    total_days = max(1, sum(state_counts.values()))
    exposure = {
        "signal_lag": signal_lag,
        "state_day_counts": state_counts,
        "short_vol_fraction": round(state_counts["short_vol"] / total_days, 4),
        "long_vol_fraction": round(state_counts["long_vol"] / total_days, 4),
        "cash_fraction": round(state_counts["cash"] / total_days, 4),
        "round_trips": len(turnovers),
        "mean_turnover_per_change": round(mean(turnovers), 4) if turnovers else 0.0,
    }
    return series, exposure


def _baselines(frame: pd.DataFrame, *, cost_bps: float) -> dict[str, pd.Series]:
    del cost_bps  # buy-and-hold baselines trade once; the one-off cost is immaterial to their metrics
    return {
        "baseline_always_short_vol": frame["short_vol_return"].astype(float).rename("always_short_vol"),
        "baseline_cash_only": frame["cash_return"].astype(float).rename("cash_only"),
        "baseline_spy": frame["spy_return"].astype(float).rename("spy"),
    }


def _validation_windows(calendar: list[pd.Timestamp], *, window_days: int, window_count: int) -> list[dict[str, object]]:
    """Tile non-overlapping windows across the WHOLE sample, not just the tail.

    The older research harnesses take the last `window_count` windows, which for a 12-year volatility
    sample would exclude every crisis before 2024 and make always-short-vol look survivable. Regime
    coverage is the entire point of this sleeve's evaluation, so every non-overlapping window is used
    and `window_count` only sets a floor.
    """
    windows: list[dict[str, object]] = []
    available = len(calendar) // window_days
    if available < window_count:
        window_days = max(20, len(calendar) // max(1, window_count))
        available = len(calendar) // window_days
    for idx in range(available):
        days = calendar[idx * window_days : (idx + 1) * window_days]
        windows.append(
            {
                "name": f"w{idx + 1:02d}_{days[0].date()}_{days[-1].date()}",
                "start": days[0],
                "end": days[-1],
            }
        )
    if len(calendar) >= 252:
        recent = calendar[-252:]
        windows.append({"name": f"recent_252_{recent[0].date()}_{recent[-1].date()}", "start": recent[0], "end": recent[-1]})
    windows.append({"name": f"full_{calendar[0].date()}_{calendar[-1].date()}", "start": calendar[0], "end": calendar[-1]})
    return windows


def _risk_metrics(returns: Iterable[float]) -> dict[str, float]:
    series = pd.Series(list(returns), dtype=float).dropna()
    if series.empty:
        return {key: 0.0 for key in ("total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "calmar", "days")}
    equity = (1.0 + series).cumprod()
    total_return = float(equity.iloc[-1]) - 1.0
    periods = len(series)
    volatility = float(series.std(ddof=0))
    cagr = (1.0 + total_return) ** (TRADING_DAYS / periods) - 1.0 if total_return > -1.0 else -1.0
    max_dd = float((equity / equity.cummax() - 1.0).min())
    return {
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6),
        "ann_vol": round(volatility * math.sqrt(TRADING_DAYS), 6),
        "sharpe": round(float(series.mean()) / volatility * math.sqrt(TRADING_DAYS), 4) if volatility > 0 else 0.0,
        "max_drawdown": round(max_dd, 6),
        "calmar": round(cagr / abs(max_dd), 4) if max_dd < 0 else 0.0,
        "days": periods,
    }


def _aggregate(runs: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for run in runs:
        if str(run["window"]).startswith("full_"):
            continue
        grouped.setdefault((str(run["strategy"]), str(run["cost_profile"])), []).append(run)
    out: dict[str, object] = {}
    for (strategy, cost_profile), values in grouped.items():
        totals = [float(run["metrics"]["total_return"]) for run in values]
        sharpes = [float(run["metrics"]["sharpe"]) for run in values]
        drawdowns = [float(run["metrics"]["max_drawdown"]) for run in values]
        out[f"{strategy}|{cost_profile}"] = {
            "strategy": strategy,
            "cost_profile": cost_profile,
            "windows": len(values),
            "mean_total_return": round(mean(totals), 6),
            "median_total_return": round(median(totals), 6),
            "mean_sharpe": round(mean(sharpes), 4),
            "worst_max_drawdown": round(min(drawdowns), 6),
            "profitable_fraction": round(sum(1 for value in totals if value > 0) / len(totals), 4),
        }
    return out


def _crisis_table(returns: pd.Series) -> dict[str, object]:
    out: dict[str, object] = {}
    for label, (start, end) in CRISIS_SLICES.items():
        sliced = returns.loc[start:end]
        if sliced.empty:
            continue
        equity = (1.0 + sliced).cumprod()
        out[label] = {
            "days": int(len(sliced)),
            "total_return": round(float(equity.iloc[-1]) - 1.0, 6),
            "worst_day": round(float(sliced.min()), 6),
            "max_drawdown": round(float((equity / equity.cummax() - 1.0).min()), 6),
        }
    return out


def _era_split(frame: pd.DataFrame, returns: pd.Series) -> dict[str, object]:
    """Split on the SVXY leverage change: the -1x era is the honest stress test."""
    boundary = pd.Timestamp("2018-02-28")
    eras = {
        "minus_1x_era_to_2018_02_27": returns.loc[returns.index < boundary],
        "minus_0p5x_era_from_2018_02_28": returns.loc[returns.index >= boundary],
    }
    return {
        label: {**_risk_metrics(series), "note": "real product exposure in this era"}
        for label, series in eras.items()
        if len(series) > 20
    }


def _overnight_gap_scenario(frame: pd.DataFrame, returns: pd.Series, exposure: dict[str, object]) -> dict[str, object]:
    """Stress the worst realistic tail: a 60% overnight loss on the short-vol leg while invested."""
    short_fraction = float(exposure.get("short_vol_fraction", 0.0) or 0.0)
    equity = (1.0 + returns).cumprod()
    baseline_total = float(equity.iloc[-1]) - 1.0
    shocked = returns.copy()
    invested_days = [stamp for stamp in returns.index if stamp in frame.index]
    if not invested_days:
        return {"note": "no invested days"}
    worst_stamp = returns.loc[invested_days].idxmin()
    shocked.loc[worst_stamp] = -0.60
    shocked_equity = (1.0 + shocked).cumprod()
    return {
        "assumption": "replace the sleeve's worst realized day with a -60% short-vol gap (Feb-2018 scale)",
        "shock_date_used": worst_stamp.date().isoformat(),
        "short_vol_fraction_of_days": short_fraction,
        "total_return_before": round(baseline_total, 6),
        "total_return_after": round(float(shocked_equity.iloc[-1]) - 1.0, 6),
        "max_drawdown_after": round(float((shocked_equity / shocked_equity.cummax() - 1.0).min()), 6),
    }


def _multiple_testing_hurdle(*, trials: int, observations: int) -> dict[str, object]:
    """Expected best-of-N annualized Sharpe under a zero-edge null (Bailey / Lopez de Prado)."""
    from statistics import NormalDist

    normal = NormalDist()
    standard_error = math.sqrt(TRADING_DAYS / max(1, observations))
    expected_max_z = (1 - EULER_GAMMA) * normal.inv_cdf(1 - 1 / trials) + EULER_GAMMA * normal.inv_cdf(
        1 - 1 / (trials * math.e)
    )
    return {
        "trials": trials,
        "observations": observations,
        "sharpe_standard_error": round(standard_error, 4),
        "expected_max_sharpe_under_null": round(expected_max_z * standard_error, 4),
        "interpretation": (
            "a variant must clear this Sharpe before its edge is distinguishable from best-of-N selection luck"
        ),
    }


def _portfolio_view(full_history: dict[str, object], cap: float) -> dict[str, object]:
    """Scale standalone sleeve metrics to the deployment cap (volatility and drawdown scale linearly)."""
    out: dict[str, object] = {}
    for strategy, metrics in full_history.items():
        out[strategy] = {
            "capped_weight": cap,
            "portfolio_ann_vol_contribution": round(float(metrics["ann_vol"]) * cap, 6),
            "portfolio_worst_drawdown_contribution": round(float(metrics["max_drawdown"]) * cap, 6),
            "portfolio_cagr_contribution_linear_approx": round(float(metrics["cagr"]) * cap, 6),
        }
    return out


def _lookahead_sensitivity(
    headline: dict[str, pd.Series],
    optimistic: dict[str, pd.Series],
    variants: list[VariantSpec],
) -> dict[str, object]:
    """Compare the look-ahead-free signal lag against the optimistic one, crisis slices included.

    If the crisis protection only exists at lag 1, it came from the 16:00/16:15 settlement mismatch
    rather than from the signal, and the sleeve's entire risk case collapses.
    """
    rows: dict[str, object] = {}
    for variant in variants:
        conservative = headline.get(f"{variant.name}|primary")
        aggressive = optimistic.get(variant.name)
        if conservative is None or aggressive is None:
            continue
        conservative_crisis = _crisis_table(conservative)
        aggressive_crisis = _crisis_table(aggressive)
        rows[variant.name] = {
            f"lag{HEADLINE_SIGNAL_LAG}_metrics": _risk_metrics(conservative),
            "lag1_metrics": _risk_metrics(aggressive),
            "crisis_volmageddon": {
                f"lag{HEADLINE_SIGNAL_LAG}": conservative_crisis.get("volmageddon_2018_02", {}).get("total_return"),
                "lag1": aggressive_crisis.get("volmageddon_2018_02", {}).get("total_return"),
            },
            "crisis_covid": {
                f"lag{HEADLINE_SIGNAL_LAG}": conservative_crisis.get("covid_2020_03", {}).get("total_return"),
                "lag1": aggressive_crisis.get("covid_2020_03", {}).get("total_return"),
            },
        }
    return {
        "explanation": (
            "VIX cash indices settle at 16:15 ET while the ETP trades at the 16:00 close, so a same-close "
            "decision (lag 1) can use up to 15 minutes of future information. Feb-2018's damage happened "
            f"inside that window, so lag {HEADLINE_SIGNAL_LAG} is used for every headline number."
        ),
        "by_variant": rows,
    }


def _gate_decision(
    *,
    aggregates: dict[str, object],
    crisis: dict[str, object],
    full_history: dict[str, object],
    variant_names: list[str],
    multiple_testing: dict[str, object],
) -> dict[str, object]:
    candidates = []
    for key, row in aggregates.items():
        strategy, cost_profile = key.split("|")
        if cost_profile != "primary" or strategy not in variant_names:
            continue
        candidates.append(row)
    if not candidates:
        return {"promoted": False, "reason": "no variant produced results"}

    ranked = sorted(candidates, key=lambda row: (row["mean_sharpe"], -abs(row["worst_max_drawdown"])), reverse=True)
    best = ranked[0]
    baseline = aggregates.get("baseline_always_short_vol|primary", {})
    best_crisis = crisis.get(str(best["strategy"]), {})
    baseline_crisis = crisis.get("baseline_always_short_vol", {})
    # Drawdown is judged on full history: a 126-day window structurally cannot show a multi-year drawdown.
    best_full = full_history.get(str(best["strategy"]), {})
    baseline_full = full_history.get("baseline_always_short_vol", {})

    checks = {
        "mean_sharpe_at_least_1_0": bool(best["mean_sharpe"] >= 1.0),
        "profitable_windows_at_least_70pct": bool(best["profitable_fraction"] >= 0.70),
        "full_history_worst_dd_within_25pct": bool(abs(float(best_full.get("max_drawdown", -1.0))) <= 0.25),
        "beats_always_short_vol_dd_2018_02": bool(
            "volmageddon_2018_02" in best_crisis
            and "volmageddon_2018_02" in baseline_crisis
            and best_crisis["volmageddon_2018_02"]["max_drawdown"]
            > baseline_crisis["volmageddon_2018_02"]["max_drawdown"]
        ),
        "beats_always_short_vol_dd_2020_03": bool(
            "covid_2020_03" in best_crisis
            and "covid_2020_03" in baseline_crisis
            and best_crisis["covid_2020_03"]["max_drawdown"] > baseline_crisis["covid_2020_03"]["max_drawdown"]
        ),
        "clears_multiple_testing_hurdle": bool(
            best["mean_sharpe"] > float(multiple_testing["expected_max_sharpe_under_null"])
        ),
        "beats_always_short_vol_sharpe_full_history": bool(
            best_full and baseline_full and float(best_full["sharpe"]) > float(baseline_full["sharpe"])
        ),
    }
    promoted = all(checks.values())
    # The pre-registered kill rule: no gated variant beats always-short-vol, or none survives the crises.
    beats_baseline = checks["beats_always_short_vol_sharpe_full_history"]
    survives_crises = checks["beats_always_short_vol_dd_2018_02"] and checks["beats_always_short_vol_dd_2020_03"]
    killed = not beats_baseline or not survives_crises
    return {
        "best_variant": best["strategy"],
        "best_variant_window_metrics": best,
        "best_variant_full_history_metrics": best_full,
        "baseline_always_short_vol_window_metrics": baseline,
        "baseline_always_short_vol_full_history_metrics": baseline_full,
        "gate_checks": checks,
        "promoted": promoted,
        "ranked_primary": ranked,
        "decision_note": (
            "PROMOTED needs every check to pass. KILLED is reserved for the pre-registered kill rule "
            "(no gated variant beats always-short-vol, or none survives the crisis slices). Anything "
            "else is WATCHLIST, with the failing checks named."
        ),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "status": "PROMOTED" if promoted else "KILLED" if killed else "WATCHLIST",
    }


def _svix_robustness(merged: pd.DataFrame, decision: dict[str, object], *, cost_bps: float) -> dict[str, object]:
    """Re-run the winning variant on SVIX (real -1x, 2022-03-30+) as an instrument-policy check."""
    best_name = str(decision.get("best_variant", ""))
    variant = next((item for item in pre_registered_variants() if item.name == best_name), None)
    if variant is None or "svix_return" not in merged:
        return {"note": "no winner or SVIX unavailable"}
    frame = merged.copy()
    frame = frame.dropna(subset=["slope_3m", "vrp", "svix_return", "cash_return"])
    if len(frame) < 200:
        return {"note": "insufficient SVIX overlap"}
    frame = frame.assign(short_vol_return=frame["svix_return"])
    returns, exposure = _run_variant(frame, variant, cost_bps=cost_bps, signal_lag=HEADLINE_SIGNAL_LAG)
    svxy_same_period = merged.loc[frame.index]
    svxy_returns, _ = _run_variant(
        svxy_same_period.dropna(subset=["slope_3m", "vrp", "short_vol_return", "cash_return"]),
        variant,
        cost_bps=cost_bps,
        signal_lag=HEADLINE_SIGNAL_LAG,
    )
    return {
        "variant": best_name,
        "period": f"{frame.index.min().date()} -> {frame.index.max().date()}",
        "svix_metrics": _risk_metrics(returns),
        "svxy_same_period_metrics": _risk_metrics(svxy_returns),
        "exposure": exposure,
        "note": (
            "SVIX is a real -1x product, so it should roughly double the -0.5x-era SVXY result; a large "
            "gap would mean the instrument policy, not the signal, drives the sleeve"
        ),
    }


def _summary_md(result: dict[str, object]) -> str:
    config = result["config"]
    decision = result["decision"]
    lines = [
        "# VOL-CARRY-V1 (Direction D1)",
        "",
        f"- thesis: `{result['thesis']}`",
        f"- execution: `{result['execution']}`",
        f"- evaluation: `{config['evaluation_start']}` -> `{config['evaluation_end']}` "
        f"(`{config['evaluation_days']}` days)",
        f"- costs: primary `{PRIMARY_COST_BPS}` bps/side, stress `{STRESS_COST_BPS}` bps/side",
        f"- pre-registered variants: `{result['pre_registration']['declared_variants']}`",
        f"- decision: **`{decision['status']}`** (best variant `{decision.get('best_variant')}`)",
        "",
        "## Methodology amendment",
        "",
        result["pre_registration"]["methodology_amendment_2026_07_28"],
        "",
        "## Full-History Net Results (primary costs)",
        "",
        "| Strategy | Total | CAGR | Ann vol | Sharpe | Worst DD | Calmar |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    full = result["full_history"]
    ordered = sorted(full.items(), key=lambda item: float(item[1]["sharpe"]), reverse=True)
    for name, metrics in ordered:
        lines.append(
            f"| `{name}` | `{metrics['total_return']:.1%}` | `{metrics['cagr']:.2%}` | "
            f"`{metrics['ann_vol']:.1%}` | `{metrics['sharpe']:.2f}` | `{metrics['max_drawdown']:.1%}` | "
            f"`{metrics['calmar']:.2f}` |"
        )

    lines += [
        "",
        "## Window Aggregates (primary costs, excludes full-history window)",
        "",
        "| Strategy | Windows | Mean total | Mean Sharpe | Worst DD | Profitable |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    primary = [row for key, row in result["aggregates"].items() if key.endswith("|primary")]
    for row in sorted(primary, key=lambda item: float(item["mean_sharpe"]), reverse=True):
        lines.append(
            f"| `{row['strategy']}` | `{row['windows']}` | `{row['mean_total_return']:.2%}` | "
            f"`{row['mean_sharpe']:.2f}` | `{row['worst_max_drawdown']:.2%}` | "
            f"`{row['profitable_fraction']:.0%}` |"
        )

    lines += [
        "",
        "## Crisis Slices (primary costs, net total return)",
        "",
        "| Strategy | " + " | ".join(f"`{label}`" for label in CRISIS_SLICES) + " |",
        "|---" * (len(CRISIS_SLICES) + 1) + "|",
    ]
    crisis_by_strategy = result["crisis_slices"]["by_strategy"]
    interesting = [decision.get("best_variant"), "baseline_always_short_vol", "baseline_cash_only", "baseline_spy"]
    for name in [item for item in interesting if item and item in crisis_by_strategy]:
        cells = []
        for label in CRISIS_SLICES:
            entry = crisis_by_strategy[name].get(label)
            cells.append(f"`{entry['total_return']:.1%}`" if entry else "`n/a`")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Leverage-Era Split (the -1x era is the honest stress test)",
        "",
        "| Strategy | Era | Total | CAGR | Sharpe | Worst DD |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name in [item for item in interesting if item and item in result["era_split"]]:
        for era, metrics in result["era_split"][name].items():
            lines.append(
                f"| `{name}` | `{era}` | `{metrics['total_return']:.1%}` | `{metrics['cagr']:.2%}` | "
                f"`{metrics['sharpe']:.2f}` | `{metrics['max_drawdown']:.1%}` |"
            )

    gate = decision.get("gate_checks", {})
    lines += [
        "",
        "## Gate Checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for check, passed in gate.items():
        lines.append(f"| `{check}` | {'PASS' if passed else 'FAIL'} |")

    multiple_testing = result["multiple_testing"]
    lines += [
        "",
        f"Multiple-testing hurdle: with `{multiple_testing['trials']}` pre-registered variants over "
        f"`{multiple_testing['observations']}` days, best-of-N selection alone is expected to produce an "
        f"annualized Sharpe of about `{multiple_testing['expected_max_sharpe_under_null']}` under a zero-edge null.",
        "",
        "## Overnight Gap Scenario",
        "",
        "| Variant | Shock date | Total before | Total after | Worst DD after |",
        "|---|---|---:|---:|---:|",
    ]
    for name, scenario in result["overnight_gap_scenario"].items():
        if "total_return_before" not in scenario:
            continue
        lines.append(
            f"| `{name}` | `{scenario['shock_date_used']}` | `{scenario['total_return_before']:.1%}` | "
            f"`{scenario['total_return_after']:.1%}` | `{scenario['max_drawdown_after']:.1%}` |"
        )

    lookahead = result["lookahead_sensitivity"]
    lines += [
        "",
        "## Look-Ahead Sensitivity (the decisive robustness test)",
        "",
        lookahead["explanation"],
        "",
        "| Variant | Sharpe (lag 2, headline) | Sharpe (lag 1, optimistic) | Volmageddon lag 2 | Volmageddon lag 1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in lookahead["by_variant"].items():
        conservative = row[f"lag{HEADLINE_SIGNAL_LAG}_metrics"]["sharpe"]
        aggressive = row["lag1_metrics"]["sharpe"]
        crisis_conservative = row["crisis_volmageddon"].get(f"lag{HEADLINE_SIGNAL_LAG}")
        crisis_aggressive = row["crisis_volmageddon"].get("lag1")
        lines.append(
            f"| `{name}` | `{conservative:.2f}` | `{aggressive:.2f}` | "
            f"`{crisis_conservative:.1%}` | `{crisis_aggressive:.1%}` |"
            if crisis_conservative is not None and crisis_aggressive is not None
            else f"| `{name}` | `{conservative:.2f}` | `{aggressive:.2f}` | `n/a` | `n/a` |"
        )

    signal = result["signal_diagnostics"]
    har = signal["har_rv"]
    lines += [
        "",
        "## Signal Diagnostics",
        "",
        f"- contango fraction (slope > 0): `{signal['slope_contango_fraction']:.1%}`",
        f"- VRP positive fraction: `{signal['vrp_positive_fraction']:.1%}`",
        f"- HAR-RV out-of-sample correlation with realized variance: `{har.get('oos_correlation_forecast_vs_actual')}`",
        f"- HAR-RV beats an expanding-mean baseline on RMSE: `{har.get('beats_expanding_mean_rmse')}`",
        f"- HAR-RV refits: `{har.get('refits')}`, first forecast `{har.get('first_forecast')}`",
        "",
        "## Data Provenance",
        "",
        "```json",
        json.dumps(_jsonable(result["data_provenance"]), indent=2)[:2600],
        "```",
        "",
    ]
    robustness = result.get("robustness_svix", {})
    if "svix_metrics" in robustness:
        lines += [
            "## SVIX Instrument-Policy Robustness",
            "",
            f"Period `{robustness['period']}`, variant `{robustness['variant']}`.",
            "",
            "| Instrument | Total | CAGR | Sharpe | Worst DD |",
            "|---|---:|---:|---:|---:|",
            f"| `SVIX (-1x real)` | `{robustness['svix_metrics']['total_return']:.1%}` | "
            f"`{robustness['svix_metrics']['cagr']:.2%}` | `{robustness['svix_metrics']['sharpe']:.2f}` | "
            f"`{robustness['svix_metrics']['max_drawdown']:.1%}` |",
            f"| `SVXY (-0.5x real)` | `{robustness['svxy_same_period_metrics']['total_return']:.1%}` | "
            f"`{robustness['svxy_same_period_metrics']['cagr']:.2%}` | "
            f"`{robustness['svxy_same_period_metrics']['sharpe']:.2f}` | "
            f"`{robustness['svxy_same_period_metrics']['max_drawdown']:.1%}` |",
            "",
        ]
    return "\n".join(lines)


def _write_charts(
    output_dir: Path,
    frame: pd.DataFrame,
    strategies: dict[str, pd.Series],
    decision: dict[str, object],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    best = str(decision.get("best_variant", ""))
    to_plot = [
        (f"{best}|primary", f"{best} (gated)", "#4C72B0"),
        ("baseline_always_short_vol|primary", "always short vol", "#C44E52"),
        ("baseline_cash_only|primary", "cash (BIL)", "#8172B2"),
        ("baseline_spy|primary", "SPY", "#55A868"),
    ]
    figure, axis = plt.subplots(figsize=(12, 6))
    for key, label, color in to_plot:
        if key not in strategies:
            continue
        equity = (1.0 + strategies[key]).cumprod()
        axis.plot(equity.index, equity.to_numpy(), label=label, color=color, linewidth=1.3)
    axis.set_yscale("log")
    axis.set_title("VOL-CARRY-V1: net equity curves (log scale, primary costs)")
    axis.set_ylabel("Equity (start = 1.0)")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "vol_carry_equity_curves.png", dpi=130)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 4.5))
    slope = frame["slope_3m"]
    axis.plot(slope.index, slope.to_numpy(), color="#4C72B0", linewidth=0.8)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.fill_between(slope.index, 0, slope.to_numpy(), where=(slope.to_numpy() < 0), color="#C44E52", alpha=0.5)
    axis.set_title("VIX3M/VIX slope: contango above zero, backwardation shaded")
    figure.tight_layout()
    figure.savefig(output_dir / "vol_carry_term_structure.png", dpi=130)
    plt.close(figure)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.date().isoformat()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if hasattr(value, "item"):
        return value.item()
    return value
