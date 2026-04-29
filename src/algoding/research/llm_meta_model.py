from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from algoding.portfolio.risk import RiskLimits
from algoding.research.historical_backtest import BacktestBar, _infer_periods_per_year, _max_drawdown


@dataclass(frozen=True)
class MetaModelPrediction:
    trading_day: str
    probability_long: float


@dataclass(frozen=True)
class MetaTradeDefinition:
    name: str
    lookback: int = 20
    threshold: float = 0.02
    entry_probability: float = 0.6
    exit_probability: float = 0.4


class LlmPriceMetaModel:
    def __init__(self) -> None:
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=400,
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        )
        self._feature_names: list[str] = []

    def fit(self, feature_rows: list[dict[str, float]], targets: list[int]) -> None:
        self._feature_names = sorted(feature_rows[0].keys()) if feature_rows else []
        matrix = [[row.get(name, 0.0) for name in self._feature_names] for row in feature_rows]
        self._pipeline.fit(matrix, targets)

    def predict_probabilities(self, feature_rows: list[dict[str, float]]) -> list[float]:
        matrix = [[row.get(name, 0.0) for name in self._feature_names] for row in feature_rows]
        probabilities = self._pipeline.predict_proba(matrix)
        return [float(row[1]) for row in probabilities]


def select_best_thresholds(
    *,
    bars: list[BacktestBar],
    probabilities: list[MetaModelPrediction],
    risk_limits: RiskLimits,
    definition_name: str,
) -> tuple[MetaTradeDefinition, dict[str, object]]:
    best_definition: MetaTradeDefinition | None = None
    best_trace: dict[str, object] | None = None
    best_score = float("-inf")
    for entry_probability in [0.55, 0.6, 0.65]:
        for exit_probability in [0.35, 0.4, 0.45]:
            definition = MetaTradeDefinition(
                name=definition_name,
                entry_probability=entry_probability,
                exit_probability=exit_probability,
            )
            trace = build_meta_overlay_trace(
                bars=bars,
                predictions=probabilities,
                definition=definition,
                risk_limits=risk_limits,
            )
            score = float(trace["summary"]["score"])
            if score > best_score:
                best_score = score
                best_definition = definition
                best_trace = trace
    if best_definition is None or best_trace is None:
        raise RuntimeError("Unable to select thresholds for meta model.")
    return best_definition, best_trace


def build_meta_overlay_trace(
    *,
    bars: list[BacktestBar],
    predictions: list[MetaModelPrediction],
    definition: MetaTradeDefinition,
    risk_limits: RiskLimits,
) -> dict[str, object]:
    if len(predictions) != len(bars):
        raise ValueError("Meta-model predictions must align one-to-one with bars.")
    prices = [bar.close for bar in bars]
    equity = 1.0
    equity_curve = [equity]
    in_position = False
    cash_equity = 1.0
    position_units = 0.0
    entry_price: float | None = None
    high_water_mark: float | None = None
    trade_start_equity: float | None = None
    trade_returns: list[float] = []
    exit_reasons: dict[str, int] = {}
    events: list[dict[str, object]] = []
    trace_points: list[dict[str, object]] = []

    for index, bar in enumerate(bars):
        momentum_signal = _momentum_signal(prices[: index + 1], definition.lookback, definition.threshold)
        probability_long = predictions[index].probability_long
        indicator_value = _momentum_indicator(prices[: index + 1], definition.lookback)
        fixed_stop = None
        trailing_stop = None
        effective_stop = None

        if in_position and entry_price is not None:
            high_water_mark = max(high_water_mark or entry_price, bar.high, bar.close)
            fixed_stop = entry_price * (1 - risk_limits.stop_loss_pct)
            trailing_stop = high_water_mark * (1 - risk_limits.trailing_stop_pct)
            effective_stop = max(fixed_stop, trailing_stop)
            if bar.low <= effective_stop:
                cash_equity += position_units * effective_stop
                position_units = 0.0
                if trade_start_equity is not None and trade_start_equity > 0:
                    trade_returns.append((cash_equity / trade_start_equity) - 1)
                exit_reasons["risk_exit"] = exit_reasons.get("risk_exit", 0) + 1
                events.append(
                    {
                        "timestamp": bar.timestamp,
                        "event": "sell",
                        "price": round(effective_stop, 6),
                        "reason": "risk_exit",
                        "fraction": 1.0,
                    }
                )
                in_position = False
                entry_price = None
                high_water_mark = None
                trade_start_equity = None
            elif momentum_signal == 0 or probability_long <= definition.exit_probability:
                cash_equity += position_units * bar.close
                position_units = 0.0
                if trade_start_equity is not None and trade_start_equity > 0:
                    trade_returns.append((cash_equity / trade_start_equity) - 1)
                reason = "signal_exit" if momentum_signal == 0 else "probability_exit"
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
                events.append(
                    {
                        "timestamp": bar.timestamp,
                        "event": "sell",
                        "price": round(bar.close, 6),
                        "reason": reason,
                        "fraction": 1.0,
                    }
                )
                in_position = False
                entry_price = None
                high_water_mark = None
                trade_start_equity = None

        if not in_position and momentum_signal == 1 and probability_long >= definition.entry_probability:
            in_position = True
            trade_start_equity = cash_equity
            entry_price = bar.close
            high_water_mark = max(bar.high, bar.close)
            position_units = cash_equity / bar.close if bar.close > 0 else 0.0
            cash_equity = 0.0
            fixed_stop = entry_price * (1 - risk_limits.stop_loss_pct)
            trailing_stop = high_water_mark * (1 - risk_limits.trailing_stop_pct)
            effective_stop = max(fixed_stop, trailing_stop)
            events.append(
                {
                    "timestamp": bar.timestamp,
                    "event": "buy",
                    "price": round(bar.close, 6),
                    "reason": "meta_entry",
                    "fraction": 1.0,
                }
            )

        equity = cash_equity + (position_units * bar.close)
        equity_curve.append(equity)
        trace_points.append(
            {
                "timestamp": bar.timestamp,
                "close": round(bar.close, 6),
                "signal": "long" if momentum_signal == 1 else "flat",
                "indicator_value": round(indicator_value, 6) if indicator_value is not None else None,
                "probability_long": round(probability_long, 6),
                "entry_probability": definition.entry_probability,
                "exit_probability": definition.exit_probability,
                "in_position": in_position,
                "equity": round(equity, 6),
                "fixed_stop_price": round(fixed_stop, 6) if fixed_stop is not None else None,
                "trailing_stop_price": round(trailing_stop, 6) if trailing_stop is not None else None,
                "effective_stop_price": round(effective_stop, 6) if effective_stop is not None else None,
            }
        )

    total_return = equity_curve[-1] - 1.0
    max_drawdown = _max_drawdown(equity_curve)
    positive = [value for value in trade_returns if value > 0]
    negative = [value for value in trade_returns if value < 0]
    win_rate = len(positive) / len(trade_returns) if trade_returns else 0.0
    periods_per_year = _infer_periods_per_year(bars)
    annualized_return = (
        ((1 + total_return) ** (periods_per_year / max(1, len(bars) - 1))) - 1 if total_return > -1 else -1.0
    )
    periodic_returns = [(equity_curve[i] / equity_curve[i - 1]) - 1 for i in range(1, len(equity_curve))]
    variance = sum(value * value for value in periodic_returns) / max(1, len(periodic_returns))
    annualized_volatility = sqrt(variance) * sqrt(periods_per_year) if periodic_returns else 0.0
    profit_factor = (
        sum(positive) / abs(sum(negative))
        if negative and abs(sum(negative)) > 0
        else float(len(positive)) if positive else 0.0
    )
    score = total_return + (max_drawdown * 0.5) + (win_rate * 0.1) + (min(profit_factor, 3.0) * 0.02)

    summary = {
        "strategy_name": definition.name,
        "strategy_family": "deepseek_news_price_meta",
        "run_type": "historical_deepseek_meta",
        "latest_signal": trace_points[-1]["signal"] if trace_points else "flat",
        "total_return": round(total_return, 6),
        "annualized_return": round(annualized_return, 6),
        "annualized_volatility": round(annualized_volatility, 6),
        "max_drawdown": round(max_drawdown, 6),
        "trades": len(trade_returns),
        "win_rate": round(win_rate, 6),
        "profit_factor": round(profit_factor, 6),
        "score": round(score, 6),
        "bars_tested": len(bars),
        "periods_per_year": round(periods_per_year, 2),
        "risk": {
            "stop_loss_pct": risk_limits.stop_loss_pct,
            "trailing_stop_pct": risk_limits.trailing_stop_pct,
            "entry_probability": definition.entry_probability,
            "exit_probability": definition.exit_probability,
        },
        "exit_reasons": exit_reasons,
    }
    return {"summary": summary, "events": events, "trace_points": trace_points}


def _momentum_signal(prices: list[float], lookback: int, threshold: float) -> int:
    if len(prices) <= lookback:
        return 0
    recent = prices[-lookback:]
    current = recent[-1]
    average = sum(recent) / len(recent)
    return 1 if current >= average * (1 + threshold) else 0


def _momentum_indicator(prices: list[float], lookback: int) -> float | None:
    if len(prices) <= lookback:
        return None
    recent = prices[-lookback:]
    current = recent[-1]
    average = sum(recent) / len(recent)
    return (current / average) - 1 if average > 0 else None
