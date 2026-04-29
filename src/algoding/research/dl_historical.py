from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from sklearn.preprocessing import StandardScaler

from algoding.portfolio.risk import RiskLimits
from algoding.research.historical_backtest import BacktestBar, _infer_periods_per_year, _max_drawdown, _trading_day_key
from algoding.research.ml_historical import TorchBinaryMlpClassifier, build_ml_dataset

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


@dataclass(frozen=True)
class DlModelDefinition:
    name: str
    family: str
    input_type: str
    probability_threshold: float = 0.55
    sequence_length: int = 24
    min_train_size_override: int | None = None
    retrain_step_override: int | None = None

    def build_estimator(self) -> object:
        if self.name == "torch_mlp":
            return TorchBinaryMlpClassifier(
                hidden_layer_sizes=(64, 32),
                learning_rate=0.001,
                batch_size=256,
                max_epochs=60,
                patience=8,
                random_state=42,
            )
        if self.name == "torch_lstm":
            return TorchSequenceLstmClassifier(
                hidden_size=64,
                num_layers=2,
                learning_rate=0.001,
                batch_size=128,
                max_epochs=70,
                patience=10,
                random_state=42,
            )
        if self.name == "torch_cnn":
            return TorchSequenceCnnClassifier(
                hidden_channels=(64, 32),
                learning_rate=0.001,
                batch_size=128,
                max_epochs=70,
                patience=10,
                random_state=42,
            )
        if self.name == "torch_gru":
            return TorchSequenceGruClassifier(
                hidden_size=64,
                num_layers=2,
                learning_rate=0.001,
                batch_size=128,
                max_epochs=70,
                patience=10,
                random_state=42,
            )
        if self.name == "torch_transformer":
            return TorchSequenceTransformerClassifier(
                model_dim=64,
                num_heads=4,
                num_layers=2,
                learning_rate=0.001,
                batch_size=128,
                max_epochs=80,
                patience=10,
                random_state=42,
            )
        if self.name == "torch_transformer_gru":
            return TorchSequenceTransformerGruClassifier(
                model_dim=64,
                num_heads=4,
                num_transformer_layers=2,
                gru_hidden_size=64,
                learning_rate=0.001,
                batch_size=128,
                max_epochs=85,
                patience=12,
                random_state=42,
            )
        raise ValueError(f"Unsupported DL model: {self.name}")


@dataclass(frozen=True)
class DlDataset:
    bars: list[BacktestBar]
    tabular_features: np.ndarray
    sequence_features: np.ndarray
    labels: np.ndarray
    feature_names: list[str]
    sequence_length: int


def available_dl_models() -> list[DlModelDefinition]:
    if torch is None:
        return []
    return [
        DlModelDefinition(
            name="torch_mlp",
            family="dl_tabular",
            input_type="tabular",
            min_train_size_override=180,
            retrain_step_override=80,
        ),
        DlModelDefinition(
            name="torch_lstm",
            family="dl_sequence_lstm",
            input_type="sequence",
            sequence_length=24,
            min_train_size_override=220,
            retrain_step_override=120,
        ),
        DlModelDefinition(
            name="torch_cnn",
            family="dl_sequence_cnn",
            input_type="sequence",
            sequence_length=24,
            min_train_size_override=220,
            retrain_step_override=120,
        ),
        DlModelDefinition(
            name="torch_gru",
            family="dl_sequence_gru",
            input_type="sequence",
            sequence_length=24,
            min_train_size_override=140,
            retrain_step_override=80,
        ),
        DlModelDefinition(
            name="torch_transformer",
            family="dl_sequence_transformer",
            input_type="sequence",
            sequence_length=24,
            min_train_size_override=140,
            retrain_step_override=80,
        ),
        DlModelDefinition(
            name="torch_transformer_gru",
            family="dl_sequence_transformer_gru",
            input_type="sequence",
            sequence_length=24,
            min_train_size_override=160,
            retrain_step_override=90,
        ),
    ]


def default_dl_models() -> list[DlModelDefinition]:
    available = {model.name: model for model in available_dl_models()}
    return [
        available["torch_mlp"],
        available["torch_lstm"],
        available["torch_cnn"],
    ]


def build_dl_dataset(
    bars: list[BacktestBar],
    *,
    max_lookback: int = 20,
    sequence_length: int = 24,
) -> DlDataset:
    base_dataset = build_ml_dataset(bars, max_lookback=max_lookback)
    if len(base_dataset.labels) < sequence_length + 60:
        raise ValueError("Not enough labeled samples to build a DL sequence dataset.")

    start_index = max(0, sequence_length - 1)
    tabular_features = base_dataset.features[start_index:]
    sequence_features = np.asarray(
        [
            base_dataset.features[index - sequence_length + 1 : index + 1]
            for index in range(start_index, len(base_dataset.features))
        ],
        dtype=float,
    )
    return DlDataset(
        bars=base_dataset.bars[start_index:],
        tabular_features=tabular_features,
        sequence_features=sequence_features,
        labels=base_dataset.labels[start_index:],
        feature_names=base_dataset.feature_names,
        sequence_length=sequence_length,
    )


def run_walk_forward_dl_trace(
    dataset: DlDataset,
    model_definition: DlModelDefinition,
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
    if sample_count < 80:
        raise ValueError("Not enough labeled samples to run DL walk-forward research.")

    minimum_eval_window = max(16, min(40, sample_count // 5))
    requested_train_size = model_definition.min_train_size_override or min_train_size or max(80, sample_count // 3)
    train_size = min(requested_train_size, sample_count - minimum_eval_window)
    if train_size < 48:
        raise ValueError("Not enough labeled samples to run DL walk-forward research.")
    requested_step = model_definition.retrain_step_override or retrain_step or max(12, min(80, sample_count // 10))
    step = max(8, min(requested_step, max(8, sample_count - train_size)))
    probabilities = np.full(sample_count, 0.5, dtype=float)
    training_device = "cpu"
    model_inputs = dataset.tabular_features if model_definition.input_type == "tabular" else dataset.sequence_features

    for train_end in range(train_size, sample_count, step):
        test_end = min(sample_count, train_end + step)
        y_train = dataset.labels[:train_end]
        if len(np.unique(y_train)) < 2:
            continue

        estimator = model_definition.build_estimator()
        estimator.fit(model_inputs[:train_end], y_train)
        training_device = getattr(estimator, "device_label", training_device)
        probabilities[train_end:test_end] = _predict_probabilities(estimator, model_inputs[train_end:test_end])

    signals = (probabilities >= model_definition.probability_threshold).astype(int)
    return _build_dl_signal_trace(
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
        feature_count=dataset.tabular_features.shape[1],
        training_device=training_device,
    )


def run_fixed_split_dl_trace(
    dataset: DlDataset,
    model_definition: DlModelDefinition,
    symbol: str,
    market: str,
    window_name: str,
    risk_limits: RiskLimits,
    *,
    train_size: int,
    test_size: int,
    no_same_day_reentry: bool = False,
) -> dict[str, object]:
    sample_count = len(dataset.labels)
    if train_size <= 0 or test_size <= 0:
        raise ValueError("Train and test sizes must be positive.")
    if sample_count < train_size + test_size:
        raise ValueError("Not enough labeled samples to run the requested fixed DL split.")

    model_inputs = dataset.tabular_features if model_definition.input_type == "tabular" else dataset.sequence_features
    start_index = sample_count - (train_size + test_size)
    end_train = start_index + train_size
    end_test = end_train + test_size

    x_train = model_inputs[start_index:end_train]
    y_train = dataset.labels[start_index:end_train]
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training labels are not diverse enough for the requested fixed DL split.")

    estimator = model_definition.build_estimator()
    estimator.fit(x_train, y_train)
    training_device = getattr(estimator, "device_label", "cpu")
    probabilities = _predict_probabilities(estimator, model_inputs[end_train:end_test])
    signals = (probabilities >= model_definition.probability_threshold).astype(int)

    return _build_dl_signal_trace(
        bars=dataset.bars[end_train:end_test],
        probabilities=probabilities,
        signals=signals,
        symbol=symbol,
        market=market,
        window_name=window_name,
        model_definition=model_definition,
        risk_limits=risk_limits,
        no_same_day_reentry=no_same_day_reentry,
        min_train_size=train_size,
        retrain_step=0,
        feature_count=dataset.tabular_features.shape[1],
        training_device=training_device,
    )


def _build_dl_signal_trace(
    *,
    bars: list[BacktestBar],
    probabilities: np.ndarray,
    signals: np.ndarray,
    symbol: str,
    market: str,
    window_name: str,
    model_definition: DlModelDefinition,
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
        "strategy_name": f"{symbol.lower()}_{model_definition.name}_dl",
        "strategy_family": model_definition.family,
        "model_name": model_definition.name,
        "model_input_type": model_definition.input_type,
        "symbol": symbol,
        "market": market,
        "window_name": window_name,
        "run_type": "historical_dl",
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
        "sequence_length": model_definition.sequence_length,
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
    probabilities = estimator.predict_proba(features)
    if probabilities.ndim == 2 and probabilities.shape[1] > 1:
        return probabilities[:, 1]
    return probabilities.reshape(-1)


class TorchSequenceClassifierBase:
    def __init__(
        self,
        *,
        learning_rate: float,
        batch_size: int,
        max_epochs: int,
        patience: int,
        random_state: int,
    ) -> None:
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is not installed.")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: nn.Module | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_label = str(self.device)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "TorchSequenceClassifierBase":
        if len(features) == 0:
            raise ValueError("No sequence features provided for training.")

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        scaled = self._scale_sequences(features, fit=True)
        sequence_length = scaled.shape[1]
        feature_count = scaled.shape[2]
        self.model = self._build_network(sequence_length, feature_count).to(self.device)

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
        scaled = self._scale_sequences(features, fit=False)
        tensor_features = torch.tensor(scaled, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor_features)
            probabilities = torch.sigmoid(logits).squeeze(-1).detach().cpu().numpy()
        probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - probabilities, probabilities])

    def _scale_sequences(self, features: np.ndarray, *, fit: bool) -> np.ndarray:
        sequence_shape = features.shape
        flattened = features.reshape(-1, sequence_shape[-1])
        if fit:
            scaled = self.scaler.fit_transform(flattened)
        else:
            scaled = self.scaler.transform(flattened)
        return scaled.reshape(sequence_shape)

    def _build_network(self, sequence_length: int, feature_count: int) -> nn.Module:
        raise NotImplementedError


class TorchSequenceLstmClassifier(TorchSequenceClassifierBase):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        learning_rate: float,
        batch_size: int,
        max_epochs: int,
        patience: int,
        random_state: int,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            random_state=random_state,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def _build_network(self, sequence_length: int, feature_count: int) -> nn.Module:
        del sequence_length
        return _LstmBinaryNetwork(
            input_size=feature_count,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )


class TorchSequenceCnnClassifier(TorchSequenceClassifierBase):
    def __init__(
        self,
        *,
        hidden_channels: tuple[int, int],
        learning_rate: float,
        batch_size: int,
        max_epochs: int,
        patience: int,
        random_state: int,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            random_state=random_state,
        )
        self.hidden_channels = hidden_channels

    def _build_network(self, sequence_length: int, feature_count: int) -> nn.Module:
        del sequence_length
        return _TemporalConvBinaryNetwork(
            input_size=feature_count,
            hidden_channels=self.hidden_channels,
        )


class TorchSequenceGruClassifier(TorchSequenceClassifierBase):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        learning_rate: float,
        batch_size: int,
        max_epochs: int,
        patience: int,
        random_state: int,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            random_state=random_state,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def _build_network(self, sequence_length: int, feature_count: int) -> nn.Module:
        del sequence_length
        return _GruBinaryNetwork(
            input_size=feature_count,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )


class TorchSequenceTransformerClassifier(TorchSequenceClassifierBase):
    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        learning_rate: float,
        batch_size: int,
        max_epochs: int,
        patience: int,
        random_state: int,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            random_state=random_state,
        )
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

    def _build_network(self, sequence_length: int, feature_count: int) -> nn.Module:
        return _TransformerBinaryNetwork(
            input_size=feature_count,
            sequence_length=sequence_length,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
        )


class TorchSequenceTransformerGruClassifier(TorchSequenceClassifierBase):
    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        num_transformer_layers: int,
        gru_hidden_size: int,
        learning_rate: float,
        batch_size: int,
        max_epochs: int,
        patience: int,
        random_state: int,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            random_state=random_state,
        )
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.gru_hidden_size = gru_hidden_size

    def _build_network(self, sequence_length: int, feature_count: int) -> nn.Module:
        return _TransformerGruBinaryNetwork(
            input_size=feature_count,
            sequence_length=sequence_length,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_transformer_layers=self.num_transformer_layers,
            gru_hidden_size=self.gru_hidden_size,
        )


class _LstmBinaryNetwork(nn.Module):
    def __init__(self, *, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=0.1)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _, (hidden_state, _) = self.lstm(features)
        final_hidden = hidden_state[-1]
        return self.head(self.dropout(final_hidden))


class _GruBinaryNetwork(nn.Module):
    def __init__(self, *, input_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=0.1)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _, hidden_state = self.gru(features)
        final_hidden = hidden_state[-1]
        return self.head(self.dropout(final_hidden))


class _TemporalConvBinaryNetwork(nn.Module):
    def __init__(self, *, input_size: int, hidden_channels: tuple[int, int]) -> None:
        super().__init__()
        first_channel, second_channel = hidden_channels
        self.network = nn.Sequential(
            nn.Conv1d(input_size, first_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(first_channel, second_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(second_channel, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        channels_first = features.transpose(1, 2)
        encoded = self.network(channels_first).squeeze(-1)
        return self.head(encoded)


class _TransformerBinaryNetwork(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        sequence_length: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.head = nn.Linear(model_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(features) + self.position_embedding[:, : features.size(1), :]
        encoded = self.encoder(encoded)
        pooled = self.norm(encoded.mean(dim=1))
        return self.head(self.dropout(pooled))


class _TransformerGruBinaryNetwork(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        sequence_length: int,
        model_dim: int,
        num_heads: int,
        num_transformer_layers: int,
        gru_hidden_size: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.gru = nn.GRU(
            input_size=model_dim,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=0.1)
        self.head = nn.Linear(gru_hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(features) + self.position_embedding[:, : features.size(1), :]
        encoded = self.encoder(encoded)
        _, hidden_state = self.gru(encoded)
        final_hidden = hidden_state[-1]
        return self.head(self.dropout(final_hidden))
