from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from algoding.portfolio.risk import RiskLimits
from algoding.research.historical_backtest import BacktestBar, _infer_periods_per_year, _max_drawdown, _trading_day_key

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


@dataclass(frozen=True)
class MlModelDefinition:
    name: str
    family: str
    probability_threshold: float = 0.55
    min_train_size_override: int | None = None
    retrain_step_override: int | None = None

    def build_estimator(self) -> object:
        if self.name == "logistic_regression":
            return Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1_000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            )
        if self.name == "hist_gradient_boosting":
            return HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=3,
                max_iter=150,
                min_samples_leaf=20,
                random_state=42,
            )
        if self.name == "mlp_classifier":
            return Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    (
                        "model",
                        MLPClassifier(
                            hidden_layer_sizes=(32, 16),
                            activation="relu",
                            alpha=0.001,
                            batch_size=64,
                            learning_rate_init=0.001,
                            early_stopping=True,
                            max_iter=400,
                            random_state=42,
                        ),
                    ),
                ]
            )
        if self.name == "torch_mlp":
            return TorchBinaryMlpClassifier(
                hidden_layer_sizes=(64, 32),
                learning_rate=0.001,
                batch_size=256,
                max_epochs=60,
                patience=8,
                random_state=42,
            )
        raise ValueError(f"Unsupported ML model: {self.name}")


@dataclass(frozen=True)
class MlDataset:
    bars: list[BacktestBar]
    features: np.ndarray
    labels: np.ndarray
    feature_names: list[str]


def default_ml_models() -> list[MlModelDefinition]:
    models = [
        MlModelDefinition(name="logistic_regression", family="ml_logistic"),
        MlModelDefinition(name="hist_gradient_boosting", family="ml_tree"),
        MlModelDefinition(name="mlp_classifier", family="dl_mlp"),
    ]
    if torch is not None and torch.cuda.is_available():
        models.append(
            MlModelDefinition(
                name="torch_mlp",
                family="dl_torch",
                probability_threshold=0.55,
                min_train_size_override=160,
                retrain_step_override=80,
            )
        )
    return models


def build_ml_dataset(bars: list[BacktestBar], max_lookback: int = 20) -> MlDataset:
    if len(bars) < max_lookback + 30:
        raise ValueError("Not enough bars to build an ML dataset.")

    closes = np.array([bar.close for bar in bars], dtype=float)
    highs = np.array([bar.high for bar in bars], dtype=float)
    lows = np.array([bar.low for bar in bars], dtype=float)
    volumes = np.array([float(bar.volume or 0.0) for bar in bars], dtype=float)

    rows: list[list[float]] = []
    labels: list[int] = []
    aligned_bars: list[BacktestBar] = []
    feature_names = [
        "ret_1",
        "ret_3",
        "ret_5",
        "ret_10",
        "ret_20",
        "sma_gap_5",
        "sma_gap_10",
        "sma_gap_20",
        "vol_5",
        "vol_10",
        "range_1",
        "range_5_avg",
        "close_vs_high_20",
        "close_vs_low_20",
        "volume_ratio_5",
        "volume_ratio_20",
    ]

    for index in range(max_lookback, len(bars) - 1):
        current_close = closes[index]
        returns_1 = _safe_return(current_close, closes[index - 1])
        returns_3 = _safe_return(current_close, closes[index - 3])
        returns_5 = _safe_return(current_close, closes[index - 5])
        returns_10 = _safe_return(current_close, closes[index - 10])
        returns_20 = _safe_return(current_close, closes[index - 20])

        sma_5 = float(np.mean(closes[index - 4 : index + 1]))
        sma_10 = float(np.mean(closes[index - 9 : index + 1]))
        sma_20 = float(np.mean(closes[index - 19 : index + 1]))

        one_bar_returns = np.diff(closes[index - 10 : index + 1]) / closes[index - 10 : index]
        vol_5 = float(np.std(one_bar_returns[-5:])) if len(one_bar_returns) >= 5 else 0.0
        vol_10 = float(np.std(one_bar_returns)) if len(one_bar_returns) > 0 else 0.0

        range_1 = _safe_return(highs[index], lows[index])
        recent_ranges = [
            _safe_return(highs[offset], lows[offset])
            for offset in range(index - 4, index + 1)
        ]
        range_5_avg = float(np.mean(recent_ranges))

        high_20 = float(np.max(highs[index - 19 : index + 1]))
        low_20 = float(np.min(lows[index - 19 : index + 1]))

        volume_ratio_5 = _safe_ratio(volumes[index], float(np.mean(volumes[index - 4 : index + 1])))
        volume_ratio_20 = _safe_ratio(volumes[index], float(np.mean(volumes[index - 19 : index + 1])))

        rows.append(
            [
                returns_1,
                returns_3,
                returns_5,
                returns_10,
                returns_20,
                _safe_return(current_close, sma_5),
                _safe_return(current_close, sma_10),
                _safe_return(current_close, sma_20),
                vol_5,
                vol_10,
                range_1,
                range_5_avg,
                _safe_return(current_close, high_20),
                _safe_return(current_close, low_20),
                volume_ratio_5,
                volume_ratio_20,
            ]
        )
        labels.append(1 if closes[index + 1] > current_close else 0)
        aligned_bars.append(bars[index])

    return MlDataset(
        bars=aligned_bars,
        features=np.asarray(rows, dtype=float),
        labels=np.asarray(labels, dtype=int),
        feature_names=feature_names,
    )


def run_walk_forward_ml_trace(
    dataset: MlDataset,
    model_definition: MlModelDefinition,
    symbol: str,
    market: str,
    window_name: str,
    risk_limits: RiskLimits,
    *,
    no_same_day_reentry: bool = False,
    min_train_size: int | None = None,
    retrain_step: int | None = None,
) -> dict[str, object]:
    sample_count = len(dataset.labels)
    if sample_count < 120:
        raise ValueError("Not enough labeled samples to run ML walk-forward research.")

    train_size = model_definition.min_train_size_override or min_train_size or max(100, sample_count // 3)
    step = model_definition.retrain_step_override or retrain_step or max(10, min(40, sample_count // 10))
    probabilities = np.full(sample_count, 0.5, dtype=float)
    training_device = "cpu"

    for train_end in range(train_size, sample_count, step):
        test_end = min(sample_count, train_end + step)
        y_train = dataset.labels[:train_end]
        if len(np.unique(y_train)) < 2:
            continue

        estimator = model_definition.build_estimator()
        estimator.fit(dataset.features[:train_end], y_train)
        training_device = getattr(estimator, "device_label", training_device)
        probabilities[train_end:test_end] = _predict_probabilities(
            estimator,
            dataset.features[train_end:test_end],
        )

    signals = (probabilities >= model_definition.probability_threshold).astype(int)
    return _build_ml_signal_trace(
        bars=dataset.bars,
        probabilities=probabilities,
        signals=signals,
        symbol=symbol,
        market=market,
        window_name=window_name,
        model_definition=model_definition,
        risk_limits=risk_limits,
        no_same_day_reentry=no_same_day_reentry,
        min_train_size=train_size,
        retrain_step=step,
        feature_count=dataset.features.shape[1],
        training_device=training_device,
    )


def _build_ml_signal_trace(
    *,
    bars: list[BacktestBar],
    probabilities: np.ndarray,
    signals: np.ndarray,
    symbol: str,
    market: str,
    window_name: str,
    model_definition: MlModelDefinition,
    risk_limits: RiskLimits,
    no_same_day_reentry: bool,
    min_train_size: int,
    retrain_step: int,
    feature_count: int,
    training_device: str,
) -> dict[str, object]:
    equity = 1.0
    equity_curve = [equity]
    in_position = False
    entry_price: float | None = None
    high_water_mark: float | None = None
    trade_returns: list[float] = []
    exit_reasons: dict[str, int] = {}
    position_days = 0
    events: list[dict[str, object]] = []
    trace_points: list[dict[str, object]] = []
    blocked_reentry_day: str | None = None

    for index, bar in enumerate(bars):
        signal = int(signals[index])
        probability = float(probabilities[index])
        fixed_stop = None
        trailing_stop = None
        effective_stop = None
        exited_this_bar = False
        trading_day = _trading_day_key(bar.timestamp)

        if in_position and entry_price is not None and index > 0:
            prev_close = bars[index - 1].close
            prior_high_water = max(high_water_mark or entry_price, entry_price)
            fixed_stop = entry_price * (1 - risk_limits.stop_loss_pct)
            trailing_stop = prior_high_water * (1 - risk_limits.trailing_stop_pct)
            effective_stop = max(value for value in [fixed_stop, trailing_stop] if value is not None)

            if bar.low <= effective_stop:
                bar_return = (effective_stop / prev_close) - 1 if prev_close > 0 else 0.0
                equity *= 1 + bar_return
                trade_returns.append((effective_stop / entry_price) - 1 if entry_price > 0 else 0.0)
                position_days += 1
                equity_curve.append(equity)
                exit_reason = "stop_loss" if effective_stop == fixed_stop else "trailing_stop"
                exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1
                events.append(
                    {
                        "timestamp": bar.timestamp,
                        "event": "sell",
                        "price": round(effective_stop, 6),
                        "reason": exit_reason,
                    }
                )
                in_position = False
                entry_price = None
                high_water_mark = None
                exited_this_bar = True
                if no_same_day_reentry:
                    blocked_reentry_day = trading_day
            else:
                high_water_mark = max(prior_high_water, bar.high, bar.close)
                trailing_stop = high_water_mark * (1 - risk_limits.trailing_stop_pct)
                effective_stop = max(value for value in [fixed_stop, trailing_stop] if value is not None)
                bar_return = (bar.close / prev_close) - 1 if prev_close > 0 else 0.0
                equity *= 1 + bar_return
                position_days += 1
                if signal == 0:
                    trade_returns.append((bar.close / entry_price) - 1 if entry_price > 0 else 0.0)
                    exit_reasons["signal_exit"] = exit_reasons.get("signal_exit", 0) + 1
                    events.append(
                        {
                            "timestamp": bar.timestamp,
                            "event": "sell",
                            "price": round(bar.close, 6),
                            "reason": "signal_exit",
                        }
                    )
                    in_position = False
                    entry_price = None
                    high_water_mark = None
                    exited_this_bar = True
                    if no_same_day_reentry:
                        blocked_reentry_day = trading_day
                equity_curve.append(equity)

        can_reenter_today = not (no_same_day_reentry and blocked_reentry_day == trading_day)
        if not in_position and signal == 1 and not exited_this_bar and can_reenter_today:
            in_position = True
            entry_price = bar.close
            high_water_mark = max(bar.high, bar.close)
            fixed_stop = entry_price * (1 - risk_limits.stop_loss_pct)
            trailing_stop = high_water_mark * (1 - risk_limits.trailing_stop_pct)
            effective_stop = max(value for value in [fixed_stop, trailing_stop] if value is not None)
            events.append(
                {
                    "timestamp": bar.timestamp,
                    "event": "buy",
                    "price": round(bar.close, 6),
                    "reason": "signal_entry",
                }
            )

        trace_points.append(
            {
                "timestamp": bar.timestamp,
                "close": round(bar.close, 6),
                "signal": "long" if signal == 1 else "flat",
                "indicator_value": round(probability, 6),
                "threshold_value": model_definition.probability_threshold,
                "in_position": in_position,
                "equity": round(equity, 6),
                "high_water_mark": round(high_water_mark, 6) if high_water_mark is not None else None,
                "fixed_stop_price": round(fixed_stop, 6) if fixed_stop is not None else None,
                "trailing_stop_price": round(trailing_stop, 6) if trailing_stop is not None else None,
                "effective_stop_price": round(effective_stop, 6) if effective_stop is not None else None,
            }
        )

    positive = [value for value in trade_returns if value > 0]
    negative = [value for value in trade_returns if value < 0]
    win_rate = len(positive) / len(trade_returns) if trade_returns else 0.0
    total_return = equity_curve[-1] - 1
    max_drawdown = _max_drawdown(equity_curve)
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
    exposure_ratio = position_days / max(1, len(bars))
    score = total_return + (max_drawdown * 0.5) + (win_rate * 0.1) + (min(profit_factor, 3.0) * 0.02)

    summary = {
        "strategy_name": f"{symbol.lower()}_{model_definition.name}_ml",
        "strategy_family": model_definition.family,
        "model_name": model_definition.name,
        "symbol": symbol,
        "market": market,
        "window_name": window_name,
        "run_type": "historical_ml",
        "latest_signal": "long" if int(signals[-1]) == 1 else "flat",
        "total_return": round(total_return, 6),
        "annualized_return": round(annualized_return, 6),
        "annualized_volatility": round(annualized_volatility, 6),
        "max_drawdown": round(max_drawdown, 6),
        "trades": len(trade_returns),
        "win_rate": round(win_rate, 6),
        "profit_factor": round(profit_factor, 6),
        "exposure_ratio": round(exposure_ratio, 6),
        "score": round(score, 6),
        "bars_tested": len(bars),
        "periods_per_year": round(periods_per_year, 2),
        "feature_count": feature_count,
        "min_train_size": min_train_size,
        "retrain_step": retrain_step,
        "training_device": training_device,
        "probability_threshold": model_definition.probability_threshold,
        "risk": {
            "stop_loss_pct": risk_limits.stop_loss_pct,
            "trailing_stop_pct": risk_limits.trailing_stop_pct,
            "max_daily_loss_bps": risk_limits.max_daily_loss_bps,
            "max_gross_exposure": risk_limits.max_gross_exposure,
            "no_same_day_reentry": no_same_day_reentry,
        },
        "exit_reasons": exit_reasons,
    }
    return {
        "summary": summary,
        "events": events,
        "trace_points": trace_points,
    }


def _predict_probabilities(estimator: object, features: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.array([], dtype=float)
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(features)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.reshape(-1)
    if hasattr(estimator, "decision_function"):
        decision = estimator.decision_function(features)
        return 1.0 / (1.0 + np.exp(-decision))
    predictions = estimator.predict(features)
    return np.asarray(predictions, dtype=float)


def _safe_return(current: float, reference: float) -> float:
    if reference <= 0:
        return 0.0
    return (current / reference) - 1.0


def _safe_ratio(current: float, reference: float) -> float:
    if current <= 0 or reference <= 0:
        return 0.0
    return (current / reference) - 1.0


class TorchBinaryMlpClassifier:
    def __init__(
        self,
        *,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        learning_rate: float = 0.001,
        batch_size: int = 256,
        max_epochs: int = 60,
        patience: int = 8,
        random_state: int = 42,
    ) -> None:
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is not installed.")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: nn.Module | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_label = str(self.device)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "TorchBinaryMlpClassifier":
        if len(features) == 0:
            raise ValueError("No features provided for training.")

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        scaled = self.scaler.fit_transform(features)
        input_dim = scaled.shape[1]
        self.model = self._build_network(input_dim).to(self.device)

        sample_count = len(scaled)
        validation_size = max(32, int(sample_count * 0.15))
        if sample_count - validation_size < 32:
            validation_size = max(1, sample_count // 5)
        train_end = max(1, sample_count - validation_size)

        x_train = torch.tensor(scaled[:train_end], dtype=torch.float32, device=self.device)
        y_train = torch.tensor(labels[:train_end].reshape(-1, 1), dtype=torch.float32, device=self.device)
        x_val = torch.tensor(scaled[train_end:], dtype=torch.float32, device=self.device)
        y_val = torch.tensor(labels[train_end:].reshape(-1, 1), dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        best_state: dict[str, object] | None = None
        best_val_loss = float("inf")
        stale_epochs = 0

        for _ in range(self.max_epochs):
            self.model.train()
            permutation = torch.randperm(x_train.size(0), device=self.device)
            for start in range(0, x_train.size(0), self.batch_size):
                batch_idx = permutation[start : start + self.batch_size]
                batch_features = x_train[batch_idx]
                batch_labels = y_train[batch_idx]
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch_features)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                if x_val.size(0) > 0:
                    val_logits = self.model(x_val)
                    val_loss = float(criterion(val_logits, y_val).item())
                else:
                    train_logits = self.model(x_train)
                    val_loss = float(criterion(train_logits, y_train).item())

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        scaled = self.scaler.transform(features)
        tensor_features = torch.tensor(scaled, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor_features)
            probabilities = torch.sigmoid(logits).squeeze(-1).detach().cpu().numpy()
        probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - probabilities, probabilities])

    def _build_network(self, input_dim: int) -> nn.Module:
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in self.hidden_layer_sizes:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        return nn.Sequential(*layers)
