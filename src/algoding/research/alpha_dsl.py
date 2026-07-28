"""A small formulaic-alpha DSL: expression trees, a text form, a parser, and a vectorized evaluator.

Expressions are trees, never Python source, so nothing here evaluates untrusted strings. Panels are held
as wide frames (index = date, columns = symbol); time-series operators work down the index and
cross-sectional operators across the columns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd


FIELDS = ("open", "high", "low", "close", "volume", "vwap", "returns")
WINDOWS = (2, 3, 5, 10, 20, 30, 60)
MAX_DEPTH = 6

UNARY_OPS = ("log", "abs", "sign", "cs_rank", "cs_zscore", "winsorize")
WINDOW_OPS = ("ts_mean", "ts_std", "ts_min", "ts_max", "ts_rank", "ts_sum", "delta", "delay", "decay_linear")
BINARY_OPS = ("add", "sub", "mul", "div")
BINARY_WINDOW_OPS = ("ts_corr", "ts_cov")
ALL_OPS = UNARY_OPS + WINDOW_OPS + BINARY_OPS + BINARY_WINDOW_OPS


@dataclass(frozen=True)
class Field:
    name: str

    def __post_init__(self) -> None:
        if self.name not in FIELDS:
            raise ValueError(f"unknown field {self.name}")


@dataclass(frozen=True)
class Const:
    value: float


@dataclass(frozen=True)
class Call:
    op: str
    args: tuple

    def __post_init__(self) -> None:
        if self.op not in ALL_OPS:
            raise ValueError(f"unknown operator {self.op}")


Expr = Union[Field, Const, Call]


class Panel:
    """Wide price/volume frames plus forward returns, aligned on one calendar."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        missing = [field for field in ("open", "high", "low", "close", "volume") if field not in frames]
        if missing:
            raise ValueError(f"panel missing fields: {missing}")
        self._frames = dict(frames)
        if "vwap" not in self._frames:
            self._frames["vwap"] = (frames["high"] + frames["low"] + frames["close"]) / 3.0
        if "returns" not in self._frames:
            self._frames["returns"] = frames["close"].pct_change()

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return self._frames["close"].index

    @property
    def symbols(self) -> pd.Index:
        return self._frames["close"].columns

    def field(self, name: str) -> pd.DataFrame:
        return self._frames[name]

    def slice(self, start: pd.Timestamp | None, end: pd.Timestamp | None) -> "Panel":
        return Panel({name: frame.loc[start:end] for name, frame in self._frames.items()})

    def forward_return(self, horizon: int) -> pd.DataFrame:
        close = self._frames["close"]
        return close.shift(-horizon) / close - 1.0

    @classmethod
    def from_ohlc(cls, ohlc: dict[str, pd.DataFrame], *, min_history: int = 300) -> "Panel":
        usable = {symbol: frame for symbol, frame in ohlc.items() if len(frame) >= min_history}
        if not usable:
            raise ValueError("no symbols with enough history")
        frames: dict[str, pd.DataFrame] = {}
        for field in ("open", "high", "low", "close", "volume"):
            frames[field] = pd.DataFrame(
                {symbol: frame[field] for symbol, frame in usable.items()}
            ).sort_index()
        return cls(frames)


def evaluate(expr: Expr, panel: Panel) -> pd.DataFrame:
    """Evaluate an expression to a wide frame of factor values."""
    if isinstance(expr, Field):
        return panel.field(expr.name)
    if isinstance(expr, Const):
        close = panel.field("close")
        return pd.DataFrame(expr.value, index=close.index, columns=close.columns)
    if not isinstance(expr, Call):
        raise TypeError(f"cannot evaluate {expr!r}")

    if expr.op in UNARY_OPS:
        value = evaluate(expr.args[0], panel)
        return _apply_unary(expr.op, value)
    if expr.op in WINDOW_OPS:
        value = evaluate(expr.args[0], panel)
        window = int(expr.args[1])
        return _apply_window(expr.op, value, window)
    if expr.op in BINARY_OPS:
        left = _operand(expr.args[0], panel)
        right = _operand(expr.args[1], panel)
        return _apply_binary(expr.op, left, right)
    if expr.op in BINARY_WINDOW_OPS:
        left = evaluate(expr.args[0], panel)
        right = evaluate(expr.args[1], panel)
        window = int(expr.args[2])
        if expr.op == "ts_corr":
            return left.rolling(window).corr(right)
        return left.rolling(window).cov(right)
    raise ValueError(f"unhandled operator {expr.op}")


def _sanitize(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan)


def _apply_unary(op: str, value: pd.DataFrame) -> pd.DataFrame:
    if op == "log":
        return _sanitize(np.log(value.where(value > 0)))
    if op == "abs":
        return value.abs()
    if op == "sign":
        return np.sign(value)
    if op == "cs_rank":
        return value.rank(axis=1, pct=True)
    if op == "cs_zscore":
        mean = value.mean(axis=1)
        std = value.std(axis=1)
        return _sanitize(value.sub(mean, axis=0).div(std.where(std > 0), axis=0))
    if op == "winsorize":
        low = value.quantile(0.01, axis=1)
        high = value.quantile(0.99, axis=1)
        return value.clip(lower=low, upper=high, axis=0)
    raise ValueError(op)


def _apply_window(op: str, value: pd.DataFrame, window: int) -> pd.DataFrame:
    window = max(2, int(window))
    if op == "ts_mean":
        return value.rolling(window).mean()
    if op == "ts_std":
        return value.rolling(window).std()
    if op == "ts_min":
        return value.rolling(window).min()
    if op == "ts_max":
        return value.rolling(window).max()
    if op == "ts_sum":
        return value.rolling(window).sum()
    if op == "ts_rank":
        return value.rolling(window).rank(pct=True)
    if op == "delta":
        return value - value.shift(window)
    if op == "delay":
        return value.shift(window)
    if op == "decay_linear":
        # Vectorized weighted sum of lags: rolling().apply() with a Python callback is orders of
        # magnitude slower and the search evaluates thousands of candidates.
        weights = np.arange(window, 0, -1, dtype=float)
        weights /= weights.sum()
        total = value * weights[0]
        for lag in range(1, window):
            total = total + value.shift(lag) * weights[lag]
        return total
    raise ValueError(op)


def _operand(arg, panel: Panel):
    """Binary operands may be bare numbers (e.g. the -1 in mul(-1, x)) as well as sub-expressions."""
    if isinstance(arg, (int, float)) and not isinstance(arg, bool):
        return float(arg)
    return evaluate(arg, panel)


def _apply_binary(op: str, left, right):
    if op == "add":
        return left + right
    if op == "sub":
        return left - right
    if op == "mul":
        return left * right
    if op == "div":
        if isinstance(right, float):
            if abs(right) < 1e-12:
                raise ValueError("division by zero constant")
            return _sanitize(left / right)
        return _sanitize(left / right.where(right.abs() > 1e-12))
    raise ValueError(op)


def to_text(expr: Expr) -> str:
    if isinstance(expr, Field):
        return expr.name
    if isinstance(expr, Const):
        return f"{expr.value:g}"
    return f"{expr.op}({', '.join(to_text(arg) if not isinstance(arg, (int, float)) else f'{arg:g}' for arg in expr.args)})"


_TOKEN = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*|-?\d+\.?\d*|\(|\)|,)")


def parse_text(text: str) -> Expr:
    """Parse the text form back into a tree. Raises ValueError on anything unrecognized."""
    tokens: list[str] = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if not match:
            if text[position:].strip() == "":
                break
            raise ValueError(f"unparseable at {position}: {text[position:position + 20]!r}")
        tokens.append(match.group(1))
        position = match.end()
    expr, index = _parse_expr(tokens, 0)
    if index != len(tokens):
        raise ValueError("trailing tokens in expression")
    return expr


def _parse_expr(tokens: list[str], index: int) -> tuple[Expr, int]:
    if index >= len(tokens):
        raise ValueError("unexpected end of expression")
    token = tokens[index]
    if re.fullmatch(r"-?\d+\.?\d*", token):
        return Const(float(token)), index + 1
    if token in FIELDS:
        return Field(token), index + 1
    if token in ALL_OPS:
        if index + 1 >= len(tokens) or tokens[index + 1] != "(":
            raise ValueError(f"operator {token} must be followed by (")
        args: list = []
        index += 2
        while True:
            if index < len(tokens) and tokens[index] == ")":
                index += 1
                break
            if re.fullmatch(r"-?\d+\.?\d*", tokens[index]) and (
                index + 1 < len(tokens) and tokens[index + 1] in (")", ",")
            ):
                args.append(float(tokens[index]) if "." in tokens[index] else int(tokens[index]))
                index += 1
            else:
                arg, index = _parse_expr(tokens, index)
                args.append(arg)
            if index < len(tokens) and tokens[index] == ",":
                index += 1
        return Call(token, tuple(args)), index
    raise ValueError(f"unknown token {token!r}")


def depth(expr: Expr) -> int:
    if isinstance(expr, (Field, Const)):
        return 1
    return 1 + max(
        (depth(arg) for arg in expr.args if isinstance(arg, (Field, Const, Call))),
        default=0,
    )


def validate(expr: Expr) -> None:
    """Structural checks: arity, window bounds, and depth."""
    if isinstance(expr, (Field, Const)):
        return
    if not isinstance(expr, Call):
        raise ValueError("not an expression")
    expected = {
        **{op: 1 for op in UNARY_OPS},
        **{op: 2 for op in WINDOW_OPS},
        **{op: 2 for op in BINARY_OPS},
        **{op: 3 for op in BINARY_WINDOW_OPS},
    }[expr.op]
    if len(expr.args) != expected:
        raise ValueError(f"{expr.op} expects {expected} args, got {len(expr.args)}")
    if expr.op in WINDOW_OPS or expr.op in BINARY_WINDOW_OPS:
        window = expr.args[-1]
        if not isinstance(window, (int, float)) or not (2 <= int(window) <= 250):
            raise ValueError(f"{expr.op} window out of range: {window}")
    for arg in expr.args:
        if isinstance(arg, (Field, Const, Call)):
            validate(arg)
    if depth(expr) > MAX_DEPTH:
        raise ValueError(f"expression deeper than {MAX_DEPTH}")


def canonical_key(expr: Expr) -> str:
    """Stable identity for dedupe: commutative operators get sorted arguments."""
    if isinstance(expr, Field):
        return expr.name
    if isinstance(expr, Const):
        return f"c{expr.value:g}"
    parts = [canonical_key(arg) if isinstance(arg, (Field, Const, Call)) else f"{arg:g}" for arg in expr.args]
    if expr.op in ("add", "mul"):
        parts = sorted(parts)
    return f"{expr.op}({','.join(parts)})"
