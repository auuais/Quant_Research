from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
from pathlib import Path
from typing import Any, Callable
from datetime import datetime

import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from algoding.portfolio.risk import RiskLimits
from algoding.research.historical_backtest import BacktestBar, _infer_periods_per_year, _max_drawdown
from algoding.research.news_directional_labeler import DirectionalNewsExample, build_directional_prompt
from algoding.research.news_labeler import BundleMetadata


STRENGTH_ORDER = {"low": 1, "medium": 2, "high": 3}
STRENGTH_CONFIDENCE = {"low": 0.55, "medium": 0.7, "high": 0.85}
DIRECTION_LABELS = ["bearish", "neutral", "bullish"]
RELATIVE_LABELS = ["underperform", "inline", "outperform"]


@dataclass(frozen=True)
class DirectionalPrediction:
    symbol: str
    trading_day: str
    direction: str
    strength: str
    relative_to_spy: str
    event_type: str
    session: str
    raw_completion: str

    @property
    def confidence(self) -> float:
        return STRENGTH_CONFIDENCE.get(self.strength, 0.55)


@dataclass(frozen=True)
class DirectionalTradeDefinition:
    name: str
    lookback: int = 20
    threshold: float = 0.02
    minimum_strength: str = "low"
    require_relative_confirmation: bool = False
    require_momentum_confirmation: bool = False


class DeepseekDirectionalScorer:
    def __init__(self, *, base_model_path: str, cache_path: str | Path, adapter_path: str | None = None) -> None:
        self._base_model_path = base_model_path
        self._adapter_path = adapter_path
        self._cache_path = Path(cache_path)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load_cache()
        model_path = adapter_path or base_model_path
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        self._model = PeftModel.from_pretrained(base_model, adapter_path) if adapter_path else base_model
        self._model.eval()

    def score_bundles(
        self,
        bundles: list[BundleMetadata],
        batch_size: int = 2,
        prompt_builder: Callable[[Any], str] | None = None,
    ) -> dict[str, DirectionalPrediction]:
        results: dict[str, DirectionalPrediction] = {}
        pending: list[BundleMetadata] = []
        for bundle in bundles:
            cache_key = self._cache_key(bundle)
            cached = self._cache.get(cache_key)
            if cached is not None:
                results[self._bundle_time_key(bundle)] = cached
                continue
            pending.append(bundle)
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            builder = prompt_builder or build_directional_prompt
            prompts = [builder(bundle) for bundle in batch]
            parsed = self._generate_and_parse(prompts)
            for bundle, payload in zip(batch, parsed):
                record = DirectionalPrediction(
                    symbol=str(getattr(bundle, "symbol")),
                    trading_day=self._bundle_time_key(bundle),
                    direction=payload["direction"],
                    strength=payload["strength"],
                    relative_to_spy=payload["relative_to_spy"],
                    event_type=payload["event_type"],
                    session=payload["session"],
                    raw_completion=payload["raw_completion"],
                )
                self._cache[self._cache_key(bundle)] = record
                self._append_cache(record, bundle)
                results[self._bundle_time_key(bundle)] = record
        return results

    def _generate_and_parse(self, prompts: list[str]) -> list[dict[str, str]]:
        encoded = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=480,
        ).to("cuda")
        with torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=64,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        resolved: list[dict[str, str]] = []
        for row, prompt in zip(generated, prompts):
            full_text = self._tokenizer.decode(row, skip_special_tokens=True)
            completion = full_text[len(prompt) :].strip() if full_text.startswith(prompt) else full_text.strip()
            resolved.append(self._parse_completion(completion))
        return resolved

    def _parse_completion(self, text: str) -> dict[str, str]:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        raw = match.group(0) if match else text
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {}
        direction = str(payload.get("direction", "neutral")).strip().lower()
        if direction not in set(DIRECTION_LABELS):
            direction = "neutral"
        strength = str(payload.get("strength", "low")).strip().lower()
        if strength not in STRENGTH_ORDER:
            strength = "low"
        relative_to_spy = str(payload.get("relative_to_spy", "inline")).strip().lower()
        if relative_to_spy not in set(RELATIVE_LABELS):
            relative_to_spy = "inline"
        event_type = str(payload.get("event_type", "other")).strip().lower()
        session = str(payload.get("session", "mixed")).strip().lower()
        return {
            "direction": direction,
            "strength": strength,
            "relative_to_spy": relative_to_spy,
            "event_type": event_type,
            "session": session,
            "raw_completion": text,
        }

    def _load_cache(self) -> dict[str, DirectionalPrediction]:
        if not self._cache_path.exists():
            return {}
        cache: dict[str, DirectionalPrediction] = {}
        for line in self._cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            record = DirectionalPrediction(
                symbol=str(payload["symbol"]),
                trading_day=str(payload["trading_day"]),
                direction=str(payload["direction"]),
                strength=str(payload["strength"]),
                relative_to_spy=str(payload["relative_to_spy"]),
                event_type=str(payload.get("event_type", "other")),
                session=str(payload.get("session", "mixed")),
                raw_completion=str(payload.get("raw_completion", "")),
            )
            cache[self._cache_key_from_payload(payload)] = record
        return cache

    def _append_cache(self, record: DirectionalPrediction, bundle: BundleMetadata) -> None:
        payload = {
            "symbol": record.symbol,
            "trading_day": record.trading_day,
            "headline_count": int(getattr(bundle, "headline_count")),
            "text_hash": self._text_hash(str(getattr(bundle, "text"))),
            "direction": record.direction,
            "strength": record.strength,
            "relative_to_spy": record.relative_to_spy,
            "event_type": record.event_type,
            "session": record.session,
            "raw_completion": record.raw_completion,
        }
        with self._cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    @staticmethod
    def _cache_key(bundle: BundleMetadata) -> str:
        return (
            f"{getattr(bundle, 'symbol')}|{DeepseekDirectionalScorer._bundle_time_key(bundle)}|{getattr(bundle, 'headline_count')}|"
            f"{DeepseekDirectionalScorer._text_hash(str(getattr(bundle, 'text')))}"
        )

    @staticmethod
    def _cache_key_from_payload(payload: dict[str, Any]) -> str:
        return (
            f"{payload.get('symbol')}|{payload.get('trading_day')}|"
            f"{payload.get('headline_count', 0)}|{payload.get('text_hash', '')}"
        )

    @staticmethod
    def _text_hash(text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _bundle_time_key(bundle: Any) -> str:
        return str(getattr(bundle, "trading_day", getattr(bundle, "trading_key", "")))


def evaluate_directional_predictions(
    *,
    examples: list[DirectionalNewsExample],
    predictions_by_day: dict[str, DirectionalPrediction],
) -> dict[str, object]:
    actual_direction = [example.direction for example in examples]
    predicted_direction = [predictions_by_day[example.trading_day].direction for example in examples]
    actual_relative = [example.relative_to_spy for example in examples]
    predicted_relative = [predictions_by_day[example.trading_day].relative_to_spy for example in examples]
    direction_precision, direction_recall, direction_f1, _ = precision_recall_fscore_support(
        actual_direction,
        predicted_direction,
        labels=DIRECTION_LABELS,
        average="macro",
        zero_division=0,
    )
    relative_precision, relative_recall, relative_f1, _ = precision_recall_fscore_support(
        actual_relative,
        predicted_relative,
        labels=RELATIVE_LABELS,
        average="macro",
        zero_division=0,
    )
    joint_correct = sum(
        1
        for example in examples
        if predictions_by_day[example.trading_day].direction == example.direction
        and predictions_by_day[example.trading_day].relative_to_spy == example.relative_to_spy
    )
    return {
        "samples": len(examples),
        "direction_accuracy": round(accuracy_score(actual_direction, predicted_direction), 6),
        "direction_balanced_accuracy": round(balanced_accuracy_score(actual_direction, predicted_direction), 6),
        "direction_macro_precision": round(direction_precision, 6),
        "direction_macro_recall": round(direction_recall, 6),
        "direction_macro_f1": round(direction_f1, 6),
        "direction_confusion_matrix": confusion_matrix(actual_direction, predicted_direction, labels=DIRECTION_LABELS).tolist(),
        "relative_accuracy": round(accuracy_score(actual_relative, predicted_relative), 6),
        "relative_balanced_accuracy": round(balanced_accuracy_score(actual_relative, predicted_relative), 6),
        "relative_macro_precision": round(relative_precision, 6),
        "relative_macro_recall": round(relative_recall, 6),
        "relative_macro_f1": round(relative_f1, 6),
        "relative_confusion_matrix": confusion_matrix(actual_relative, predicted_relative, labels=RELATIVE_LABELS).tolist(),
        "joint_accuracy": round(joint_correct / len(examples), 6) if examples else 0.0,
        "predicted_direction_distribution": dict(Counter(predicted_direction)),
        "actual_direction_distribution": dict(Counter(actual_direction)),
    }


def select_best_directional_trade_definition(
    *,
    bars: list[BacktestBar],
    examples: list[DirectionalNewsExample],
    predictions_by_day: dict[str, DirectionalPrediction],
    risk_limits: RiskLimits,
    definition_prefix: str,
) -> tuple[DirectionalTradeDefinition, dict[str, object], dict[str, float]]:
    best_definition: DirectionalTradeDefinition | None = None
    best_trace: dict[str, object] | None = None
    best_metrics: dict[str, float] | None = None
    best_score = float("-inf")
    for minimum_strength in ["low", "medium", "high"]:
        for require_relative_confirmation in [False, True]:
            for require_momentum_confirmation in [False, True]:
                definition = DirectionalTradeDefinition(
                    name=f"{definition_prefix}_{minimum_strength}_{'rel' if require_relative_confirmation else 'norel'}_{'mom' if require_momentum_confirmation else 'nomom'}",
                    minimum_strength=minimum_strength,
                    require_relative_confirmation=require_relative_confirmation,
                    require_momentum_confirmation=require_momentum_confirmation,
                )
                trade_metrics = evaluate_trade_signal_correctness(
                    examples=examples,
                    predictions_by_day=predictions_by_day,
                    definition=definition,
                )
                score = trade_metrics["precision"] + (trade_metrics["coverage"] * 0.25) + (trade_metrics["recall"] * 0.25)
                if score > best_score:
                    trace = build_directional_trade_trace(
                        bars=bars,
                        predictions_by_day=predictions_by_day,
                        definition=definition,
                        risk_limits=risk_limits,
                    )
                    best_score = score
                    best_definition = definition
                    best_trace = trace
                    best_metrics = trade_metrics
    if best_definition is None or best_trace is None or best_metrics is None:
        raise RuntimeError("Unable to select a directional trade definition.")
    return best_definition, best_trace, best_metrics


def evaluate_trade_signal_correctness(
    *,
    examples: list[DirectionalNewsExample],
    predictions_by_day: dict[str, DirectionalPrediction],
    definition: DirectionalTradeDefinition,
) -> dict[str, float]:
    taken = 0
    correct = 0
    bullish_total = 0
    for example in examples:
        if example.direction == "bullish":
            bullish_total += 1
        prediction = predictions_by_day[example.trading_day]
        if _should_be_long(prediction=prediction, definition=definition, momentum_ok=True):
            taken += 1
            if example.direction == "bullish":
                correct += 1
    precision = correct / taken if taken else 0.0
    recall = correct / bullish_total if bullish_total else 0.0
    coverage = taken / len(examples) if examples else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "coverage": round(coverage, 6),
    }


def build_directional_trade_trace(
    *,
    bars: list[BacktestBar],
    predictions_by_day: dict[str, DirectionalPrediction],
    definition: DirectionalTradeDefinition,
    risk_limits: RiskLimits,
    no_same_day_reentry: bool = False,
    prediction_key_mode: str = "day",
) -> dict[str, object]:
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
    exit_reasons: Counter[str] = Counter()
    events: list[dict[str, object]] = []
    trace_points: list[dict[str, object]] = []
    blocked_reentry_day: str | None = None

    for index, bar in enumerate(bars):
        trading_day = datetime.fromisoformat(bar.timestamp).date().isoformat()
        prediction_key = bar.timestamp if prediction_key_mode == "timestamp" else trading_day
        prediction = predictions_by_day.get(
            prediction_key,
            DirectionalPrediction(
                symbol="",
                trading_day=prediction_key,
                direction="neutral",
                strength="low",
                relative_to_spy="inline",
                event_type="other",
                session="mixed",
                raw_completion="",
            ),
        )
        momentum_ok = _momentum_signal(prices[: index + 1], definition.lookback, definition.threshold) == 1
        fixed_stop = None
        trailing_stop = None
        effective_stop = None
        exited_this_bar = False
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
                exit_reasons["risk_exit"] += 1
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
                exited_this_bar = True
                if no_same_day_reentry:
                    blocked_reentry_day = trading_day
            elif not _should_be_long(prediction=prediction, definition=definition, momentum_ok=momentum_ok):
                cash_equity += position_units * bar.close
                position_units = 0.0
                if trade_start_equity is not None and trade_start_equity > 0:
                    trade_returns.append((cash_equity / trade_start_equity) - 1)
                exit_reasons["prediction_exit"] += 1
                events.append(
                    {
                        "timestamp": bar.timestamp,
                        "event": "sell",
                        "price": round(bar.close, 6),
                        "reason": "prediction_exit",
                        "fraction": 1.0,
                    }
                )
                in_position = False
                entry_price = None
                high_water_mark = None
                trade_start_equity = None
                exited_this_bar = True
                if no_same_day_reentry:
                    blocked_reentry_day = trading_day

        can_reenter_today = not (no_same_day_reentry and blocked_reentry_day == trading_day)
        if (
            not in_position
            and not exited_this_bar
            and can_reenter_today
            and _should_be_long(prediction=prediction, definition=definition, momentum_ok=momentum_ok)
        ):
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
                    "reason": "directional_entry",
                    "fraction": 1.0,
                }
            )

        equity = cash_equity + (position_units * bar.close)
        equity_curve.append(equity)
        trace_points.append(
            {
                "timestamp": bar.timestamp,
                "close": round(bar.close, 6),
                "direction": prediction.direction,
                "strength": prediction.strength,
                "relative_to_spy": prediction.relative_to_spy,
                "confidence": round(prediction.confidence, 6),
                "momentum_ok": momentum_ok,
                "in_position": in_position,
                "equity": round(equity, 6),
                "fixed_stop_price": round(fixed_stop, 6) if fixed_stop is not None else None,
                "trailing_stop_price": round(trailing_stop, 6) if trailing_stop is not None else None,
                "effective_stop_price": round(effective_stop, 6) if effective_stop is not None else None,
            }
        )

    positive = [value for value in trade_returns if value > 0]
    negative = [value for value in trade_returns if value < 0]
    total_return = equity_curve[-1] - 1.0
    max_drawdown = _max_drawdown(equity_curve)
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
        "strategy_family": "deepseek_directional",
        "run_type": "historical_deepseek_directional",
        "latest_signal": "long" if in_position else "flat",
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
            "no_same_day_reentry": no_same_day_reentry,
            "minimum_strength": definition.minimum_strength,
            "require_relative_confirmation": definition.require_relative_confirmation,
            "require_momentum_confirmation": definition.require_momentum_confirmation,
        },
        "exit_reasons": dict(exit_reasons),
    }
    return {"summary": summary, "events": events, "trace_points": trace_points}


def build_directional_trade_trace_with_momentum_shift_entry(
    *,
    bars: list[BacktestBar],
    predictions_by_day: dict[str, DirectionalPrediction],
    definition: DirectionalTradeDefinition,
    risk_limits: RiskLimits,
    prediction_key_mode: str = "day",
) -> dict[str, object]:
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
    exit_reasons: Counter[str] = Counter()
    events: list[dict[str, object]] = []
    trace_points: list[dict[str, object]] = []

    for index, bar in enumerate(bars):
        trading_day = datetime.fromisoformat(bar.timestamp).date().isoformat()
        prediction_key = bar.timestamp if prediction_key_mode == "timestamp" else trading_day
        prediction = predictions_by_day.get(
            prediction_key,
            DirectionalPrediction(
                symbol="",
                trading_day=prediction_key,
                direction="neutral",
                strength="low",
                relative_to_spy="inline",
                event_type="other",
                session="mixed",
                raw_completion="",
            ),
        )
        momentum_ok = _momentum_signal(prices[: index + 1], definition.lookback, definition.threshold) == 1
        momentum_shift = _momentum_shift_signal(prices[: index + 1], definition.lookback, definition.threshold)
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
                exit_reasons["risk_exit"] += 1
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
            elif not _should_be_long(prediction=prediction, definition=definition, momentum_ok=True):
                cash_equity += position_units * bar.close
                position_units = 0.0
                if trade_start_equity is not None and trade_start_equity > 0:
                    trade_returns.append((cash_equity / trade_start_equity) - 1)
                exit_reasons["prediction_exit"] += 1
                events.append(
                    {
                        "timestamp": bar.timestamp,
                        "event": "sell",
                        "price": round(bar.close, 6),
                        "reason": "prediction_exit",
                        "fraction": 1.0,
                    }
                )
                in_position = False
                entry_price = None
                high_water_mark = None
                trade_start_equity = None

        if not in_position and _should_be_long(prediction=prediction, definition=definition, momentum_ok=True) and momentum_shift:
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
                    "reason": "directional_entry_momentum_shift",
                    "fraction": 1.0,
                }
            )

        equity = cash_equity + (position_units * bar.close)
        equity_curve.append(equity)
        trace_points.append(
            {
                "timestamp": bar.timestamp,
                "close": round(bar.close, 6),
                "direction": prediction.direction,
                "strength": prediction.strength,
                "relative_to_spy": prediction.relative_to_spy,
                "confidence": round(prediction.confidence, 6),
                "momentum_ok": momentum_ok,
                "momentum_shift": momentum_shift,
                "in_position": in_position,
                "equity": round(equity, 6),
                "fixed_stop_price": round(fixed_stop, 6) if fixed_stop is not None else None,
                "trailing_stop_price": round(trailing_stop, 6) if trailing_stop is not None else None,
                "effective_stop_price": round(effective_stop, 6) if effective_stop is not None else None,
            }
        )

    positive = [value for value in trade_returns if value > 0]
    negative = [value for value in trade_returns if value < 0]
    total_return = equity_curve[-1] - 1.0
    max_drawdown = _max_drawdown(equity_curve)
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
        "strategy_name": f"{definition.name}_momentum_shift_entry",
        "strategy_family": "deepseek_directional",
        "run_type": "historical_deepseek_directional_execution_aware",
        "latest_signal": "long" if in_position else "flat",
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
            "minimum_strength": definition.minimum_strength,
            "require_relative_confirmation": definition.require_relative_confirmation,
            "require_momentum_confirmation": definition.require_momentum_confirmation,
            "entry_mode": "momentum_shift",
        },
        "exit_reasons": dict(exit_reasons),
    }
    return {"summary": summary, "events": events, "trace_points": trace_points}


def _should_be_long(
    *,
    prediction: DirectionalPrediction,
    definition: DirectionalTradeDefinition,
    momentum_ok: bool,
) -> bool:
    if prediction.direction != "bullish":
        return False
    if STRENGTH_ORDER.get(prediction.strength, 0) < STRENGTH_ORDER.get(definition.minimum_strength, 0):
        return False
    if definition.require_relative_confirmation and prediction.relative_to_spy == "underperform":
        return False
    if definition.require_momentum_confirmation and not momentum_ok:
        return False
    return True


def _momentum_signal(prices: list[float], lookback: int, threshold: float) -> int:
    if len(prices) <= lookback:
        return 0
    recent = prices[-lookback:]
    current = recent[-1]
    average = sum(recent) / len(recent)
    return 1 if current >= average * (1 + threshold) else 0


def _momentum_shift_signal(prices: list[float], lookback: int, threshold: float) -> bool:
    if len(prices) <= lookback + 1:
        return False
    previous_signal = _momentum_signal(prices[:-1], lookback, threshold)
    current_signal = _momentum_signal(prices, lookback, threshold)
    return previous_signal == 0 and current_signal == 1
