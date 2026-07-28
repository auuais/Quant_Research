"""ALPHA-FACTORY-V1: automated formulaic-alpha search (frontier direction D3).

Pre-registered protocol (reports/research/frontier_hypotheses.md#d3):
  train  -> generation and in-sample IC
  val    -> acceptance decisions (RankIC >= 0.02, ICIR >= 0.25), dedupe at |corr| > 0.7
  test   -> run exactly ONCE, on the final ensemble only
Kill rule: if validation->test RankIC decays more than 50% across the accepted pool, freeze the factory
until a point-in-time panel exists. Do not iterate against the test period.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import numpy as np
import pandas as pd

from algoding.data.events import EventCalendarClient
from algoding.execution.long_short_research import UNIVERSE_100
from algoding.research.alpha_dsl import Panel, evaluate, parse_text, to_text, validate
from algoding.research.alpha_search import (
    FactorScore,
    crossover,
    factor_correlation,
    mutate,
    propose_with_llm,
    random_expression,
    score_factor,
    seed_expressions,
    wrap_cross_sectional,
)


TRADING_DAYS = 252.0
FORWARD_HORIZON = 5
EMBARGO_DAYS = 5
TRAIN_END = "2024-06-30"
VALIDATION_START = "2024-07-01"
VALIDATION_END = "2025-06-30"
TEST_START = "2025-07-01"
SEARCH_BUDGET = 4000
ACCEPT_RANK_IC = 0.02
ACCEPT_ICIR = 0.25
MAX_TURNOVER = 0.45
DEDUPE_CORRELATION = 0.70
ENSEMBLE_SIZE = 8
BOOK_SIZE = 20
REBALANCE_DAYS = 5
PRIMARY_COST_BPS = 1.0
STRESS_COST_BPS = 3.0


@dataclass
class AcceptedFactor:
    expression: str
    origin: str
    direction: float
    train: FactorScore
    validation: FactorScore
    test: FactorScore | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "expression": self.expression,
            "origin": self.origin,
            "direction": self.direction,
            "train": self.train.as_dict(),
            "validation": self.validation.as_dict(),
            "test": self.test.as_dict() if self.test else None,
        }


class AlphaFactoryResearchLab:
    def __init__(self, settings=None) -> None:
        self._settings = settings
        self._events = EventCalendarClient()

    def run(
        self,
        *,
        output_root: str = "reports/research/alpha_factory_v1",
        start: str = "2005-01-01",
        search_budget: int = SEARCH_BUDGET,
        seed: int = 11,
        llm_model_path: str | None = None,
        llm_proposals: int = 0,
    ) -> dict[str, object]:
        output_dir = Path(output_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)

        ohlc = self._events.load_ohlc(list(UNIVERSE_100), start=start)
        panel = Panel.from_ohlc(ohlc)
        forward = panel.forward_return(FORWARD_HORIZON)

        train_end = pd.Timestamp(TRAIN_END)
        validation_start = pd.Timestamp(VALIDATION_START)
        validation_end = pd.Timestamp(VALIDATION_END)
        test_start = pd.Timestamp(TEST_START)

        # Embargo: the forward horizon plus the declared buffer must not straddle a split boundary.
        train_forward_cut = train_end - pd.Timedelta(days=FORWARD_HORIZON + EMBARGO_DAYS)
        train_panel = panel.slice(None, train_end)
        train_forward = forward.loc[forward.index <= train_forward_cut]
        validation_panel = panel.slice(None, validation_end)
        test_panel = panel

        search = self._search(
            rng=rng,
            train_panel=train_panel,
            train_forward=train_forward,
            budget=search_budget,
            llm_model_path=llm_model_path,
            llm_proposals=llm_proposals,
        )
        accepted = self._accept_on_validation(
            candidates=search["survivors"],
            validation_panel=validation_panel,
            forward=forward,
            validation_start=validation_start,
            validation_end=validation_end,
        )

        baselines = _baseline_factors(panel)
        baseline_scores = {
            name: {
                "validation": _score_frame(frame, forward, validation_start, validation_end),
                "test": _score_frame(frame, forward, test_start, panel.calendar.max()),
            }
            for name, frame in baselines.items()
        }

        # ---- Single, final test-period evaluation -------------------------------------------------
        test_end = panel.calendar.max()
        for factor in accepted:
            expr = parse_text(factor.expression)
            factor.test = score_factor(
                expr,
                test_panel,
                forward,
                origin=factor.origin,
                eval_start=test_start,
                direction=factor.direction,
            )
        ensemble = self._ensemble_books(
            accepted=accepted,
            panel=panel,
            baselines=baselines,
            test_start=test_start,
            test_end=test_end,
            validation_start=validation_start,
            validation_end=validation_end,
        )
        decay = _decay_analysis(accepted)
        regime = _validation_regime_note(baseline_scores)
        decision = _decide(decay=decay, ensemble=ensemble, accepted=accepted, regime=regime)

        result: dict[str, object] = {
            "version": "ALPHA-FACTORY-V1",
            "direction": "D3",
            "thesis": (
                "automated search over a formulaic price/volume DSL finds factor ensembles with higher "
                "validation RankIC than the hand-built winners, and they survive a once-run test period"
            ),
            "pre_registration": {
                "registry": "reports/research/frontier_hypotheses.md#d3",
                "declared_budget": SEARCH_BUDGET,
                "realized_evaluations": search["evaluations"],
                "acceptance": {
                    "rank_ic": ACCEPT_RANK_IC,
                    "icir": ACCEPT_ICIR,
                    "max_turnover": MAX_TURNOVER,
                    "dedupe_correlation": DEDUPE_CORRELATION,
                },
                "kill_rule": "freeze the factory if validation->test RankIC decays more than 50%",
                "test_runs": 1,
                "panel_note": (
                    "the panel is UNIVERSE_100 loaded from Yahoo over ~21 years, which is far longer than "
                    "the ~3.5-year Alpaca panel the registry's null assumed; it is also a currently-listed "
                    "mega-cap set, so survivorship bias is SEVERE over this span and any accepted factor "
                    "partly encodes 'these particular names went up'"
                ),
            },
            "config": {
                "start": start,
                "panel_range": f"{panel.calendar.min().date()} -> {panel.calendar.max().date()}",
                "panel_days": int(len(panel.calendar)),
                "panel_symbols": int(len(panel.symbols)),
                "forward_horizon": FORWARD_HORIZON,
                "splits": {
                    "train": f"{panel.calendar.min().date()} -> {train_end.date()}",
                    "train_forward_cut": str(train_forward_cut.date()),
                    "validation": f"{validation_start.date()} -> {validation_end.date()}",
                    "test": f"{test_start.date()} -> {test_end.date()}",
                    "embargo_days": EMBARGO_DAYS,
                },
                "book": {"size": BOOK_SIZE, "rebalance_days": REBALANCE_DAYS},
                "costs": {"primary_bps": PRIMARY_COST_BPS, "stress_bps": STRESS_COST_BPS},
                "seed": seed,
            },
            "search": {key: value for key, value in search.items() if key != "survivors"},
            "accepted_factors": [factor.as_dict() for factor in accepted],
            "baseline_scores": baseline_scores,
            "validation_regime": regime,
            "ensemble": ensemble,
            "decay_analysis": decay,
            "decision": decision,
        }
        (output_dir / "alpha_factory_v1_run.json").write_text(
            json.dumps(_jsonable(result), indent=2), encoding="utf-8"
        )
        (output_dir / "alpha_factory_v1_summary.md").write_text(_summary_md(result), encoding="utf-8")
        _write_charts(output_dir, result)
        return result

    def _search(
        self,
        *,
        rng: random.Random,
        train_panel: Panel,
        train_forward: pd.DataFrame,
        budget: int,
        llm_model_path: str | None,
        llm_proposals: int,
    ) -> dict[str, object]:
        """Evolutionary search scored purely on train data."""
        population: list[tuple[FactorScore, object]] = []
        seen: set[str] = set()
        evaluations = 0
        origin_counts: dict[str, int] = {}
        accepted_on_train: list[FactorScore] = []

        def consider(expr, origin: str) -> FactorScore | None:
            nonlocal evaluations
            try:
                expr = wrap_cross_sectional(expr)
                validate(expr)
            except Exception:
                return None
            text = to_text(expr)
            if text in seen:
                return None
            seen.add(text)
            evaluations += 1
            origin_counts[origin] = origin_counts.get(origin, 0) + 1
            score = score_factor(expr, train_panel, train_forward, origin=origin)
            if score is None:
                return None
            population.append((score, expr))
            return score

        for expr in seed_expressions():
            consider(expr, "seed")

        llm_diagnostics: dict[str, object] = {"status": "not requested"}
        if llm_proposals > 0:
            ranked_seeds = sorted(population, key=lambda item: abs(item[0].rank_ic), reverse=True)
            proposals, llm_diagnostics = propose_with_llm(
                accepted=[score for score, _ in ranked_seeds[:8]],
                count=llm_proposals,
                model_path=llm_model_path,
                hypothesis_hint=(
                    "Focus on hypotheses about liquidity provision, short-horizon overreaction, and "
                    "volume-price divergence."
                ),
            )
            for expr in proposals:
                consider(expr, "llm")

        while evaluations < budget:
            remaining = budget - evaluations
            if remaining <= 0:
                break
            roll = rng.random()
            if not population or roll < 0.35:
                consider(random_expression(rng, max_depth=rng.choice([2, 3, 3, 4])), "random")
                continue
            ranked = sorted(population, key=lambda item: abs(item[0].rank_ic), reverse=True)
            elite = ranked[: max(5, len(ranked) // 5)]
            if roll < 0.80:
                _, parent = rng.choice(elite)
                consider(mutate(parent, rng), "mutation")
            else:
                _, left = rng.choice(elite)
                _, right = rng.choice(elite)
                consider(crossover(left, right, rng), "crossover")

        for score, expr in population:
            if (
                abs(score.rank_ic) >= ACCEPT_RANK_IC
                and abs(score.icir) >= ACCEPT_ICIR
                and score.turnover <= MAX_TURNOVER
            ):
                accepted_on_train.append(score)
        accepted_on_train.sort(key=lambda item: abs(item.icir), reverse=True)
        return {
            "evaluations": evaluations,
            "scored": len(population),
            "origin_counts": origin_counts,
            "train_passing": len(accepted_on_train),
            "llm": llm_diagnostics,
            "survivors": accepted_on_train[:60],
            "best_train_rank_ic": round(max((abs(item.rank_ic) for item in accepted_on_train), default=0.0), 6),
        }

    def _accept_on_validation(
        self,
        *,
        candidates: list[FactorScore],
        validation_panel: Panel,
        forward: pd.DataFrame,
        validation_start: pd.Timestamp,
        validation_end: pd.Timestamp,
    ) -> list[AcceptedFactor]:
        """Re-score train survivors on validation, keeping only decorrelated passers."""
        validation_forward = forward.loc[forward.index <= validation_end]
        accepted: list[AcceptedFactor] = []
        accepted_frames: list[pd.DataFrame] = []
        for candidate in candidates:
            expr = parse_text(candidate.text)
            direction = 1.0 if candidate.rank_ic >= 0 else -1.0
            score = score_factor(
                expr,
                validation_panel,
                validation_forward,
                origin=candidate.origin,
                eval_start=validation_start,
                direction=direction,
            )
            if score is None or score.rank_ic < ACCEPT_RANK_IC or score.icir < ACCEPT_ICIR:
                continue
            try:
                frame = evaluate(expr, validation_panel).loc[validation_start:validation_end]
            except Exception:
                continue
            if any(
                abs(factor_correlation(frame, existing)) > DEDUPE_CORRELATION for existing in accepted_frames
            ):
                continue
            accepted_frames.append(frame)
            accepted.append(
                AcceptedFactor(
                    expression=candidate.text,
                    origin=candidate.origin,
                    direction=direction,
                    train=candidate,
                    validation=score,
                )
            )
        accepted.sort(key=lambda item: item.validation.icir, reverse=True)
        return accepted

    def _ensemble_books(
        self,
        *,
        accepted: list[AcceptedFactor],
        panel: Panel,
        baselines: dict[str, pd.DataFrame],
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
        validation_start: pd.Timestamp,
        validation_end: pd.Timestamp,
    ) -> dict[str, object]:
        if not accepted:
            return {"note": "no accepted factors; no ensemble built"}
        chosen = accepted[:ENSEMBLE_SIZE]
        frames = []
        for factor in chosen:
            expr = parse_text(factor.expression)
            try:
                values = evaluate(expr, panel)
            except Exception:
                continue
            frames.append(values.rank(axis=1, pct=True) * factor.direction)
        if not frames:
            return {"note": "ensemble frames could not be evaluated"}
        combined = sum(frames) / len(frames)

        books: dict[str, object] = {}
        for label, factor_frame in [("ensemble_rank_average", combined)] + [
            (name, frame) for name, frame in baselines.items()
        ]:
            for period_label, period_start, period_end in (
                ("validation", validation_start, validation_end),
                ("test", test_start, test_end),
            ):
                for cost_label, cost_bps in (("primary", PRIMARY_COST_BPS), ("stress", STRESS_COST_BPS)):
                    books[f"{label}|{period_label}|{cost_label}"] = _long_only_book(
                        factor_frame=factor_frame,
                        panel=panel,
                        start=period_start,
                        end=period_end,
                        cost_bps=cost_bps,
                    )
        return {
            "ensemble_members": [factor.expression for factor in chosen],
            "ensemble_size": len(frames),
            "books": books,
            "construction": (
                f"per-factor cross-sectional percentile rank, sign-corrected by the train-period IC sign, "
                f"averaged; long-only top {BOOK_SIZE}, rebalanced every {REBALANCE_DAYS} sessions"
            ),
        }


def _baseline_factors(panel: Panel) -> dict[str, pd.DataFrame]:
    """The two incumbent winners, rebuilt on this panel so the comparison is apples-to-apples."""
    close = panel.field("close")
    ret_20d = close / close.shift(20) - 1.0
    daily = close.pct_change()
    market = daily.mean(axis=1)
    window = 60
    market_var = market.rolling(window).var()
    covariance = daily.rolling(window).cov(market)
    beta = covariance.div(market_var.where(market_var > 0), axis=0)
    residual = daily.sub(beta.mul(market, axis=0))
    resid_mom_60d = residual.rolling(window).sum()
    return {"baseline_ret_20d": ret_20d, "baseline_resid_mom_60d": resid_mom_60d}


def _score_frame(
    frame: pd.DataFrame,
    forward: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, object]:
    sliced = frame.loc[start:end]
    aligned = forward.reindex_like(sliced)
    valid = sliced.notna() & aligned.notna()
    usable = valid.sum(axis=1) >= 20
    if usable.sum() < 20:
        return {"rank_ic": None, "icir": None, "days": int(usable.sum())}
    factor_ranks = sliced.where(valid).rank(axis=1)
    forward_ranks = aligned.where(valid).rank(axis=1)
    left = factor_ranks.loc[usable]
    right = forward_ranks.loc[usable]
    left_centered = left.sub(left.mean(axis=1), axis=0)
    right_centered = right.sub(right.mean(axis=1), axis=0)
    numerator = (left_centered * right_centered).sum(axis=1)
    denominator = np.sqrt((left_centered**2).sum(axis=1) * (right_centered**2).sum(axis=1))
    ic = (numerator / denominator.where(denominator > 0)).dropna()
    if len(ic) < 20:
        return {"rank_ic": None, "icir": None, "days": int(len(ic))}
    std = float(ic.std(ddof=0))
    return {
        "rank_ic": round(float(ic.mean()), 6),
        # Annualized, matching alpha_search.score_factor so baselines and candidates are comparable.
        "icir": round(float(ic.mean()) / std * math.sqrt(TRADING_DAYS), 4) if std > 0 else 0.0,
        "days": int(len(ic)),
    }


def _long_only_book(
    *,
    factor_frame: pd.DataFrame,
    panel: Panel,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_bps: float,
) -> dict[str, object]:
    """Top-N long-only book: decide at T close from the factor, earn T+1 onward, rebalance every N days."""
    close = panel.field("close")
    daily = close.pct_change()
    calendar = [stamp for stamp in panel.calendar if start <= stamp <= end]
    if len(calendar) < REBALANCE_DAYS * 3:
        return {"note": "period too short"}

    weights = pd.Series(dtype=float)
    net_returns: list[float] = []
    gross_returns: list[float] = []
    turnovers: list[float] = []
    for index, stamp in enumerate(calendar):
        if len(weights):
            realized = daily.loc[stamp, weights.index].fillna(0.0)
            gross = float((weights * realized).sum())
        else:
            gross = 0.0
        cost = 0.0
        if index % REBALANCE_DAYS == 0 and index < len(calendar) - 1:
            scores = factor_frame.loc[stamp].dropna()
            if len(scores) >= BOOK_SIZE:
                chosen = scores.nlargest(BOOK_SIZE).index
                target = pd.Series(1.0 / BOOK_SIZE, index=chosen)
            else:
                target = pd.Series(dtype=float)
            union = weights.index.union(target.index)
            turnover = float(
                (target.reindex(union).fillna(0.0) - weights.reindex(union).fillna(0.0)).abs().sum()
            )
            cost = turnover * (cost_bps / 10_000.0)
            turnovers.append(turnover)
            weights = target
        gross_returns.append(gross)
        net_returns.append(gross - cost)

    net = pd.Series(net_returns, index=pd.DatetimeIndex(calendar))
    gross_series = pd.Series(gross_returns, index=pd.DatetimeIndex(calendar))
    equal_weight = daily.loc[net.index].mean(axis=1)
    return {
        "net_metrics": _metrics(net, equal_weight),
        "gross_metrics": _metrics(gross_series, equal_weight),
        "mean_turnover": round(mean(turnovers), 4) if turnovers else 0.0,
        "rebalances": len(turnovers),
        "benchmark_equal_weight_metrics": _metrics(equal_weight),
    }


def _metrics(returns: pd.Series, benchmark: pd.Series | None = None) -> dict[str, float]:
    series = returns.dropna()
    if series.empty:
        return {key: 0.0 for key in ("total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "days")}
    equity = (1.0 + series).cumprod()
    total = float(equity.iloc[-1]) - 1.0
    volatility = float(series.std(ddof=0))
    out = {
        "total_return": round(total, 6),
        "cagr": round((1.0 + total) ** (TRADING_DAYS / len(series)) - 1.0 if total > -1 else -1.0, 6),
        "ann_vol": round(volatility * math.sqrt(TRADING_DAYS), 6),
        "sharpe": round(float(series.mean()) / volatility * math.sqrt(TRADING_DAYS), 4) if volatility > 0 else 0.0,
        "max_drawdown": round(float((equity / equity.cummax() - 1.0).min()), 6),
        "days": int(len(series)),
    }
    if benchmark is not None:
        joined = pd.concat({"strategy": series, "benchmark": benchmark}, axis=1).dropna()
        if len(joined) > 30 and joined["benchmark"].var() > 0:
            beta = float(joined["strategy"].cov(joined["benchmark"]) / joined["benchmark"].var())
            out["beta_vs_equal_weight"] = round(beta, 4)
    return out


def _validation_regime_note(baseline_scores: dict[str, object]) -> dict[str, object]:
    """Did the incumbent factors themselves work during validation?

    If the hand-built winners have negative IC over the validation window, then 'no mined factor passed
    validation' cannot be attributed to the mining alone -- the window was hostile to the whole factor
    family, and a single contiguous validation year cannot separate the two explanations.
    """
    rows = {}
    for name, scores in baseline_scores.items():
        validation = scores.get("validation", {}) if isinstance(scores, dict) else {}
        test = scores.get("test", {}) if isinstance(scores, dict) else {}
        rows[name] = {
            "validation_rank_ic": validation.get("rank_ic"),
            "validation_icir": validation.get("icir"),
            "test_rank_ic": test.get("rank_ic"),
            "test_icir": test.get("icir"),
        }
    validation_ics = [
        row["validation_rank_ic"] for row in rows.values() if row["validation_rank_ic"] is not None
    ]
    all_negative = bool(validation_ics) and all(value <= 0 for value in validation_ics)
    would_pass = [
        name
        for name, row in rows.items()
        if row["validation_rank_ic"] is not None and row["validation_rank_ic"] >= ACCEPT_RANK_IC
    ]
    return {
        "baselines": rows,
        "all_baselines_negative_in_validation": all_negative,
        "baselines_clearing_acceptance_in_validation": would_pass,
        "interpretation": (
            "the incumbent factors also fail the acceptance bar over this validation window, so the window "
            "is hostile to the entire cross-sectional price/volume family; a single contiguous validation "
            "year cannot distinguish 'mining produces nothing' from 'nothing worked that year'"
            if all_negative
            else "at least one incumbent factor works in validation, so the window discriminates between factors"
        ),
        "required_fix_for_next_registration": (
            "replace the single contiguous validation year with purged multi-fold walk-forward validation so "
            "acceptance is not decided by one regime"
        ),
    }


def _decay_analysis(accepted: list[AcceptedFactor]) -> dict[str, object]:
    rows = [
        {
            "expression": factor.expression,
            "train_rank_ic": factor.train.rank_ic,
            "validation_rank_ic": factor.validation.rank_ic,
            "test_rank_ic": factor.test.rank_ic if factor.test else None,
        }
        for factor in accepted
    ]
    with_test = [row for row in rows if row["test_rank_ic"] is not None]
    if not with_test:
        return {"rows": rows, "note": "no test scores available"}
    validation_mean = mean(row["validation_rank_ic"] for row in with_test)
    test_mean = mean(row["test_rank_ic"] for row in with_test)
    decay = (validation_mean - test_mean) / validation_mean if validation_mean else float("nan")
    return {
        "rows": rows,
        "accepted_count": len(accepted),
        "mean_validation_rank_ic": round(validation_mean, 6),
        "mean_test_rank_ic": round(test_mean, 6),
        "median_test_rank_ic": round(median(row["test_rank_ic"] for row in with_test), 6),
        "validation_to_test_decay": round(decay, 4) if np.isfinite(decay) else None,
        "kill_threshold": 0.50,
        "kill_rule_fired": bool(np.isfinite(decay) and decay > 0.50),
        "test_positive_fraction": round(
            sum(1 for row in with_test if row["test_rank_ic"] > 0) / len(with_test), 4
        ),
    }


def _decide(
    *,
    decay: dict[str, object],
    ensemble: dict[str, object],
    accepted: list[AcceptedFactor],
    regime: dict[str, object] | None = None,
) -> dict[str, object]:
    regime = regime or {}
    if not accepted:
        hostile = bool(regime.get("all_baselines_negative_in_validation"))
        return {
            "status": "KILLED",
            "reason": "no candidate passed validation acceptance",
            "promoted": False,
            "kill_rule_fired": True,
            "accepted_factor_count": 0,
            "decay_test_reached": False,
            "validation_window_hostile_to_all_baselines": hostile,
            "note": (
                "Nothing survived validation acceptance, so the validation->test decay test never ran; the "
                "factory is frozen either way. But the incumbent factors are ALSO negative over this "
                "validation window, so this run cannot separate 'the search finds nothing real' from 'this "
                "one validation year was hostile to every cross-sectional price/volume factor'. The fix is a "
                "new registration with purged multi-fold validation -- not looser thresholds on this one."
                if hostile
                else "Nothing survived validation acceptance while at least one incumbent factor did, so the "
                "search genuinely produced no out-of-sample factor: the null in the D3 registration holds."
            ),
        }
    books = ensemble.get("books", {}) if isinstance(ensemble, dict) else {}
    ensemble_test = books.get("ensemble_rank_average|test|primary", {})
    incumbent_test = books.get("baseline_resid_mom_60d|test|primary", {})
    momentum_test = books.get("baseline_ret_20d|test|primary", {})

    def sharpe(entry: object) -> float | None:
        if isinstance(entry, dict) and isinstance(entry.get("net_metrics"), dict):
            return float(entry["net_metrics"]["sharpe"])
        return None

    ensemble_sharpe = sharpe(ensemble_test)
    incumbent_sharpe = sharpe(incumbent_test)
    beats_incumbent = (
        ensemble_sharpe is not None
        and incumbent_sharpe is not None
        and ensemble_sharpe >= incumbent_sharpe + 0.05
    )
    kill_fired = bool(decay.get("kill_rule_fired"))
    return {
        "status": "KILLED" if kill_fired else "PROMOTED" if beats_incumbent else "WATCHLIST",
        "promoted": bool(beats_incumbent and not kill_fired),
        "kill_rule_fired": kill_fired,
        "validation_to_test_decay": decay.get("validation_to_test_decay"),
        "ensemble_test_sharpe": ensemble_sharpe,
        "incumbent_resid_mom_60d_test_sharpe": incumbent_sharpe,
        "ret_20d_test_sharpe": sharpe(momentum_test),
        "beats_incumbent_by_0_05": beats_incumbent,
        "accepted_factor_count": len(accepted),
        "note": (
            "the kill rule is evaluated first and overrides a favourable book comparison: an ensemble whose "
            "component ICs decayed more than 50% is not a finding, it is an overfit on too small a panel"
        ),
    }


def _summary_md(result: dict[str, object]) -> str:
    config = result["config"]
    search = result["search"]
    decay = result["decay_analysis"]
    decision = result["decision"]
    lines = [
        "# ALPHA-FACTORY-V1 (Direction D3)",
        "",
        f"- thesis: `{result['thesis']}`",
        f"- panel: `{config['panel_range']}` (`{config['panel_days']}` days x `{config['panel_symbols']}` symbols)",
        f"- splits: train `{config['splits']['train']}`, validation `{config['splits']['validation']}`, "
        f"test `{config['splits']['test']}` (embargo `{config['splits']['embargo_days']}` days)",
        f"- declared search budget: `{result['pre_registration']['declared_budget']}`, "
        f"realized evaluations: `{search['evaluations']}`",
        f"- decision: **`{decision['status']}`**",
        "",
        "## Severe caveat on this panel",
        "",
        result["pre_registration"]["panel_note"],
        "",
        "## Search",
        "",
        f"- expressions scored: `{search['scored']}`",
        f"- passing train thresholds: `{search['train_passing']}`",
        f"- best absolute train RankIC: `{search['best_train_rank_ic']}`",
        f"- generator mix: `{search['origin_counts']}`",
        f"- LLM proposer: `{search['llm'].get('status')}`",
        "",
        "## Validation-Period Context (read this before the factor table)",
        "",
        "Whether the acceptance stage can discriminate at all depends on whether the *incumbent* factors work "
        "over the validation window. Rebuilt on this same panel:",
        "",
        "| Baseline | Validation IC | Validation ICIR | Test IC | Test ICIR |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in result.get("validation_regime", {}).get("baselines", {}).items():
        lines.append(
            f"| `{name}` | `{row['validation_rank_ic']}` | `{row['validation_icir']}` | "
            f"`{row['test_rank_ic']}` | `{row['test_icir']}` |"
        )
    lines += [
        "",
        result.get("validation_regime", {}).get("interpretation", ""),
        "",
        f"Required fix for the next registration: {result.get('validation_regime', {}).get('required_fix_for_next_registration', '')}",
        "",
        "## Accepted Factors (validation-accepted, then scored once on test)",
        "",
        "| Expression | Origin | Dir | Train IC | Val IC | Val ICIR | Test IC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for factor in result["accepted_factors"]:
        test_ic = factor["test"]["rank_ic"] if factor["test"] else None
        lines.append(
            f"| `{factor['expression'][:70]}` | `{factor['origin']}` | `{factor['direction']:+.0f}` | "
            f"`{factor['train']['rank_ic']:+.4f}` | `{factor['validation']['rank_ic']:+.4f}` | "
            f"`{factor['validation']['icir']:+.3f}` | "
            + (f"`{test_ic:+.4f}` |" if test_ic is not None else "`n/a` |")
        )

    lines += [
        "",
        "## Validation -> Test Decay (the pre-registered kill test)",
        "",
        f"- accepted factors: `{decay.get('accepted_count')}`",
        f"- mean validation RankIC: `{decay.get('mean_validation_rank_ic')}`",
        f"- mean test RankIC: `{decay.get('mean_test_rank_ic')}`",
        f"- decay: `{decay.get('validation_to_test_decay')}` against a kill threshold of "
        f"`{decay.get('kill_threshold')}`",
        f"- kill rule fired: **`{decay.get('kill_rule_fired')}`**",
        f"- fraction of accepted factors with positive test IC: `{decay.get('test_positive_fraction')}`",
    ]

    books = result.get("ensemble", {}).get("books", {})
    if books:
        lines += [
            "",
            "## Book Comparison (top-20 long only, net of primary costs)",
            "",
            "| Book | Period | Net CAGR | Net Sharpe | Worst DD | Beta vs EW | Turnover |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for key in sorted(books):
            if not key.endswith("|primary"):
                continue
            entry = books[key]
            if not isinstance(entry, dict) or "net_metrics" not in entry:
                continue
            label, period, _ = key.split("|")
            net = entry["net_metrics"]
            lines.append(
                f"| `{label}` | `{period}` | `{net['cagr']:.2%}` | `{net['sharpe']:.2f}` | "
                f"`{net['max_drawdown']:.1%}` | `{net.get('beta_vs_equal_weight', 'n/a')}` | "
                f"`{entry['mean_turnover']:.2f}` |"
            )

    lines += [
        "",
        "## Decision",
        "",
        f"- status: **`{decision['status']}`**",
        f"- ensemble test Sharpe: `{decision.get('ensemble_test_sharpe')}`",
        f"- incumbent `resid_mom_60d` test Sharpe: `{decision.get('incumbent_resid_mom_60d_test_sharpe')}`",
        f"- `ret_20d` test Sharpe: `{decision.get('ret_20d_test_sharpe')}`",
        "",
        decision["note"],
        "",
    ]
    return "\n".join(lines)


def _write_charts(output_dir: Path, result: dict[str, object]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = [row for row in result["decay_analysis"].get("rows", []) if row.get("test_rank_ic") is not None]
    if not rows:
        return
    labels = [f"f{index + 1}" for index in range(len(rows))]
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar(x - 0.27, [row["train_rank_ic"] for row in rows], 0.27, label="train", color="#8172B2")
    axis.bar(x, [row["validation_rank_ic"] for row in rows], 0.27, label="validation", color="#4C72B0")
    axis.bar(x + 0.27, [row["test_rank_ic"] for row in rows], 0.27, label="test (run once)", color="#C44E52")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axhline(ACCEPT_RANK_IC, color="grey", linestyle="--", linewidth=0.8, label="acceptance threshold")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Rank IC")
    axis.set_title("ALPHA-FACTORY-V1: per-factor RankIC by split")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "alpha_factory_ic_decay.png", dpi=130)
    plt.close(figure)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, FactorScore):
        return value.as_dict()
    if hasattr(value, "item"):
        return value.item()
    return value
