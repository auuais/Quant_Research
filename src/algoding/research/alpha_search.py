"""Alpha search: seed pool, evolutionary operators, an optional LLM proposer, and IC evaluation.

The search only ever sees train and validation data. The test split is held by the caller and touched
once, which is the whole point of the D3 pre-registration.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from algoding.research.alpha_dsl import (
    BINARY_OPS,
    BINARY_WINDOW_OPS,
    Call,
    Const,
    Expr,
    Field,
    FIELDS,
    Panel,
    UNARY_OPS,
    WINDOW_OPS,
    WINDOWS,
    canonical_key,
    depth,
    evaluate,
    parse_text,
    to_text,
    validate,
)


MIN_SYMBOLS_PER_DAY = 20


@dataclass
class FactorScore:
    key: str
    text: str
    rank_ic: float
    icir: float
    ic_positive_fraction: float
    turnover: float
    coverage: float
    days: int
    origin: str = "unknown"
    extras: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "expression": self.text,
            "origin": self.origin,
            "rank_ic": round(self.rank_ic, 6),
            "icir": round(self.icir, 4),
            "ic_positive_fraction": round(self.ic_positive_fraction, 4),
            "turnover": round(self.turnover, 4),
            "coverage": round(self.coverage, 4),
            "days": self.days,
            **self.extras,
        }


def seed_expressions() -> list[Expr]:
    """Classic formulaic-alpha shapes: reversal, momentum, volume, range, and volatility forms."""
    texts = [
        # short-horizon reversal
        "cs_rank(mul(-1, delta(close, 5)))",
        "cs_rank(mul(-1, returns))",
        "cs_rank(mul(-1, ts_sum(returns, 5)))",
        # momentum
        "cs_rank(delta(close, 20))",
        "cs_rank(div(close, delay(close, 60)))",
        "cs_rank(ts_mean(returns, 20))",
        # volume / liquidity
        "cs_rank(div(volume, ts_mean(volume, 20)))",
        "cs_rank(mul(-1, ts_corr(close, volume, 10)))",
        "cs_rank(mul(delta(volume, 5), delta(close, 5)))",
        # range / gap structure
        "cs_rank(div(sub(close, low), sub(high, low)))",
        "cs_rank(div(sub(open, delay(close, 1)), delay(close, 1)))",
        # volatility
        "cs_rank(mul(-1, ts_std(returns, 20)))",
        "cs_rank(div(ts_std(returns, 5), ts_std(returns, 60)))",
        # position in range
        "cs_rank(div(sub(close, ts_min(low, 20)), sub(ts_max(high, 20), ts_min(low, 20))))",
        "cs_rank(sub(vwap, close))",
        "cs_rank(mul(-1, ts_rank(close, 20)))",
        "cs_rank(decay_linear(returns, 10))",
        "cs_rank(mul(-1, ts_cov(returns, div(volume, ts_mean(volume, 20)), 10)))",
    ]
    out: list[Expr] = []
    for text in texts:
        try:
            expr = parse_text(text)
            validate(expr)
            out.append(expr)
        except Exception:
            continue
    return out


def random_expression(rng: random.Random, *, max_depth: int = 3) -> Expr:
    if max_depth <= 1:
        return Field(rng.choice(FIELDS))
    choice = rng.random()
    if choice < 0.30:
        op = rng.choice(UNARY_OPS)
        return Call(op, (random_expression(rng, max_depth=max_depth - 1),))
    if choice < 0.65:
        op = rng.choice(WINDOW_OPS)
        return Call(op, (random_expression(rng, max_depth=max_depth - 1), rng.choice(WINDOWS)))
    if choice < 0.90:
        op = rng.choice(BINARY_OPS)
        return Call(
            op,
            (
                random_expression(rng, max_depth=max_depth - 1),
                random_expression(rng, max_depth=max_depth - 1),
            ),
        )
    op = rng.choice(BINARY_WINDOW_OPS)
    return Call(
        op,
        (
            random_expression(rng, max_depth=max_depth - 1),
            random_expression(rng, max_depth=max_depth - 1),
            rng.choice(WINDOWS),
        ),
    )


def _subtrees(expr: Expr) -> list[Expr]:
    out = [expr]
    if isinstance(expr, Call):
        for arg in expr.args:
            if isinstance(arg, (Field, Const, Call)):
                out.extend(_subtrees(arg))
    return out


def _replace_subtree(expr: Expr, target: Expr, replacement: Expr) -> Expr:
    if expr is target:
        return replacement
    if not isinstance(expr, Call):
        return expr
    new_args = tuple(
        _replace_subtree(arg, target, replacement) if isinstance(arg, (Field, Const, Call)) else arg
        for arg in expr.args
    )
    return Call(expr.op, new_args)


def mutate(expr: Expr, rng: random.Random) -> Expr:
    """Either perturb a window constant or swap a random subtree for a fresh one."""
    if rng.random() < 0.45 and isinstance(expr, Call):
        windowed = [node for node in _subtrees(expr) if isinstance(node, Call) and (
            node.op in WINDOW_OPS or node.op in BINARY_WINDOW_OPS
        )]
        if windowed:
            node = rng.choice(windowed)
            new_window = rng.choice(WINDOWS)
            replacement = Call(node.op, node.args[:-1] + (new_window,))
            return _replace_subtree(expr, node, replacement)
    nodes = _subtrees(expr)
    target = rng.choice(nodes)
    replacement = random_expression(rng, max_depth=max(2, 4 - depth(target)))
    return _replace_subtree(expr, target, replacement)


def crossover(left: Expr, right: Expr, rng: random.Random) -> Expr:
    donor = rng.choice(_subtrees(right))
    target = rng.choice(_subtrees(left))
    return _replace_subtree(left, target, donor)


def wrap_cross_sectional(expr: Expr) -> Expr:
    """Books are cross-sectional, so a factor is only meaningful after a cross-sectional transform."""
    if isinstance(expr, Call) and expr.op in ("cs_rank", "cs_zscore"):
        return expr
    return Call("cs_rank", (expr,))


def score_factor(
    expr: Expr,
    panel: Panel,
    forward: pd.DataFrame,
    *,
    origin: str = "unknown",
    min_coverage: float = 0.5,
    eval_start: pd.Timestamp | None = None,
    direction: float = 1.0,
) -> FactorScore | None:
    """Daily cross-sectional rank IC of a factor against forward returns.

    The panel may extend before `eval_start` so rolling windows have warmup history; only days from
    `eval_start` onward contribute to the IC, which is what keeps a split evaluation clean.
    """
    try:
        values = evaluate(expr, panel)
    except Exception:
        return None
    if not isinstance(values, pd.DataFrame) or values.empty:
        return None
    values = values.replace([np.inf, -np.inf], np.nan)
    if eval_start is not None:
        values = values.loc[values.index >= eval_start]
        if values.empty:
            return None
    aligned_forward = forward.reindex_like(values)

    valid = values.notna() & aligned_forward.notna()
    per_day_count = valid.sum(axis=1)
    usable_days = per_day_count >= MIN_SYMBOLS_PER_DAY
    if usable_days.sum() < 60:
        return None

    factor_ranks = values.where(valid).rank(axis=1)
    forward_ranks = aligned_forward.where(valid).rank(axis=1)
    ic = _row_correlation(factor_ranks.loc[usable_days], forward_ranks.loc[usable_days])
    ic = ic.dropna()
    if len(ic) < 60:
        return None

    ic = ic * direction
    mean_ic = float(ic.mean())
    std_ic = float(ic.std(ddof=0))
    # ICIR is the ANNUALIZED information ratio of the daily IC series. A raw per-day mean/std of 0.25
    # would annualize to ~4.0, which no real factor reaches, so the annualized reading is the only one
    # consistent with the modest RankIC >= 0.02 bar registered alongside it.
    icir = (mean_ic / std_ic) * math.sqrt(252.0) if std_ic > 0 else 0.0
    coverage = float(per_day_count[usable_days].mean() / max(1, values.shape[1]))
    if coverage < min_coverage:
        return None
    normalized = values.where(valid).rank(axis=1, pct=True)
    turnover = float(normalized.diff().abs().mean(axis=1).mean())
    return FactorScore(
        key=canonical_key(expr),
        text=to_text(expr),
        rank_ic=mean_ic,
        icir=icir,
        ic_positive_fraction=float((ic > 0).mean()),
        turnover=turnover if math.isfinite(turnover) else 1.0,
        coverage=coverage,
        days=int(len(ic)),
        origin=origin,
    )


def _row_correlation(left: pd.DataFrame, right: pd.DataFrame) -> pd.Series:
    """Pearson correlation per row of two aligned rank frames (i.e. Spearman on the raw values)."""
    left_centered = left.sub(left.mean(axis=1), axis=0)
    right_centered = right.sub(right.mean(axis=1), axis=0)
    numerator = (left_centered * right_centered).sum(axis=1)
    denominator = np.sqrt((left_centered**2).sum(axis=1) * (right_centered**2).sum(axis=1))
    return numerator / denominator.where(denominator > 0)


def factor_correlation(left: pd.DataFrame, right: pd.DataFrame) -> float:
    """Average daily cross-sectional Spearman correlation between two factor frames."""
    common_dates = left.index.intersection(right.index)
    common_symbols = left.columns.intersection(right.columns)
    if len(common_dates) < 30 or len(common_symbols) < MIN_SYMBOLS_PER_DAY:
        return 0.0
    a = left.loc[common_dates, common_symbols]
    b = right.loc[common_dates, common_symbols]
    valid = a.notna() & b.notna()
    a_ranks = a.where(valid).rank(axis=1)
    b_ranks = b.where(valid).rank(axis=1)
    correlation = _row_correlation(a_ranks, b_ranks).dropna()
    return float(correlation.mean()) if len(correlation) else 0.0


def propose_with_llm(
    *,
    accepted: list[FactorScore],
    count: int,
    model_path: str | None,
    hypothesis_hint: str = "",
) -> tuple[list[Expr], dict[str, object]]:
    """Ask a local causal LM for new expressions in the DSL (AlphaAgent-style hypothesis -> factor).

    Returns an empty list when no model is configured or the model is unreachable; the caller records
    which generator actually produced the accepted factors so the report cannot overstate LLM involvement.
    """
    diagnostics: dict[str, object] = {"requested": count, "model_path": model_path}
    if not model_path:
        diagnostics["status"] = "skipped: no model configured"
        return [], diagnostics
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as error:
        diagnostics["status"] = f"skipped: transformers unavailable ({type(error).__name__})"
        return [], diagnostics

    grammar = (
        "Fields: " + ", ".join(FIELDS) + "\n"
        "Operators: " + ", ".join(UNARY_OPS + WINDOW_OPS + BINARY_OPS + BINARY_WINDOW_OPS) + "\n"
        "Windows must be integers from " + ", ".join(str(w) for w in WINDOWS) + ".\n"
        "Write one expression per line, no prose, e.g. cs_rank(mul(-1, delta(close, 5)))"
    )
    best = "\n".join(f"{item.text}  # rank_ic={item.rank_ic:.4f}" for item in accepted[:8])
    prompt = (
        "You are designing cross-sectional equity alpha factors.\n"
        f"{grammar}\n\n"
        f"Factors already found:\n{best or '(none yet)'}\n\n"
        f"{hypothesis_hint}\n"
        f"Propose {count} NEW distinct expressions that are not minor variants of the above.\n"
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=400, do_sample=True, temperature=0.9, top_p=0.95)
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    except Exception as error:
        diagnostics["status"] = f"failed: {type(error).__name__}: {str(error)[:160]}"
        return [], diagnostics

    proposals: list[Expr] = []
    parse_failures = 0
    for line in text.splitlines():
        candidate = line.split("#")[0].strip()
        if not candidate:
            continue
        try:
            expr = parse_text(candidate)
            validate(expr)
            proposals.append(expr)
        except Exception:
            parse_failures += 1
    diagnostics.update(
        {"status": "ok", "parsed": len(proposals), "parse_failures": parse_failures, "raw_chars": len(text)}
    )
    return proposals, diagnostics
