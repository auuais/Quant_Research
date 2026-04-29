from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import quantiles

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from algoding.data.historical import AlpacaHistoricalClient
from algoding.data.news import AlpacaNewsHistoricalClient
from algoding.execution.llm_research import LlmNewsResearchLab
from algoding.portfolio.risk import RiskLimits
from algoding.research.cost_aware_backtest import run_execution_aware_replay
from algoding.research.deepseek_directional_model import (
    DirectionalPrediction,
    DirectionalTradeDefinition,
    build_directional_trade_trace_with_momentum_shift_entry,
)
from algoding.research.historical_backtest import BacktestBar
from algoding.research.llm_sentiment import LocalLlmNewsSentimentEngine
from algoding.research.news_labeler import build_bundle_metadata
from algoding.settings import Settings


HORIZONS = (1, 3, 5)
HORIZON_WEIGHTS = {1: 0.2, 3: 0.3, 5: 0.5}
CLASS_ORDER = ["bearish", "neutral", "bullish"]
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_ORDER)}

SECTOR_UNIVERSES = {
    "semis": ["AVGO", "NVDA", "AMD", "QCOM", "INTC", "AMAT", "LRCX", "KLAC"],
    "growth": ["GOOGL", "TSLA", "META", "AMZN", "NFLX", "AAPL", "MSFT", "CRM", "ORCL", "ADBE"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC"],
}

TOP5_SYMBOLS = ["AVGO", "NVDA", "TSLA", "GOOGL", "XOM"]
SYMBOL_TO_SECTOR = {
    symbol: sector
    for sector, symbols in SECTOR_UNIVERSES.items()
    for symbol in symbols
}


@dataclass(frozen=True)
class V4Row:
    symbol: str
    sector: str
    trading_day: str
    split: str
    features_fused: dict[str, float]
    features_price_only: dict[str, float]
    targets: dict[int, str]
    forward_excess_returns: dict[int, float]


class DeepseekDirectionalV4ResearchLab:
    def __init__(self, settings: Settings) -> None:
        if not settings.has_alpaca_credentials:
            raise RuntimeError("Alpaca credentials are required in .env for DeepSeek V4 research.")
        self._settings = settings
        self._bars_client = AlpacaHistoricalClient(settings)
        self._news_client = AlpacaNewsHistoricalClient(settings)
        self._risk_limits = RiskLimits.from_file().with_overrides(
            stop_loss_pct=0.05,
            trailing_stop_pct=0.01,
        )

    def run(
        self,
        *,
        output_root: str = "reports/research/deepseek_directional_v4",
        base_model_path: str = r"E:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B",
        max_bars: int = 1400,
        neutral_band: float = 0.3,
    ) -> dict[str, object]:
        output_dir = Path(output_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        sentiment_settings = Settings()
        sentiment_settings.alpaca_api_key = self._settings.alpaca_api_key
        sentiment_settings.alpaca_secret_key = self._settings.alpaca_secret_key
        sentiment_settings.llm_news_model_name = base_model_path
        sentiment_settings.llm_news_cache_dir = Path("cache/llm_news_deepseek_v4")
        sentiment_engine = LocalLlmNewsSentimentEngine(sentiment_settings)

        symbols = sorted({symbol for values in SECTOR_UNIVERSES.values() for symbol in values} | {"SPY"})
        bars_by_symbol = self._load_bars(symbols=symbols, max_bars=max_bars)
        benchmark_bars = bars_by_symbol["SPY"]
        split_by_day = self._build_date_splits(benchmark_bars)

        sentiment_by_symbol = {}
        article_count_by_symbol = {}
        for symbol in symbols:
            if symbol == "SPY":
                continue
            bars = bars_by_symbol[symbol]
            articles = self._load_news(symbol=symbol, bars=bars)
            trading_days = [datetime.fromisoformat(bar.timestamp).date() for bar in bars]
            bundles = sentiment_engine.build_trading_day_bundles(
                symbol=symbol,
                articles=articles,
                trading_days=trading_days,
            )
            scores = sentiment_engine.score_bundles(bundles, batch_size=8)
            filled = sentiment_engine.fill_missing_days(trading_days, scores, symbol)
            sentiment_by_symbol[symbol] = {item.trading_day: item for item in filled}
            article_count_by_symbol[symbol] = {
                bundle.trading_day: bundle.headline_count for bundle in bundles
            }

        rows = self._build_rows(
            bars_by_symbol=bars_by_symbol,
            benchmark_bars=benchmark_bars,
            split_by_day=split_by_day,
            sentiment_by_symbol=sentiment_by_symbol,
            article_count_by_symbol=article_count_by_symbol,
        )
        thresholds = self._compute_thresholds(rows=rows, neutral_band=neutral_band)
        rows = self._apply_targets(rows=rows, thresholds=thresholds)
        assumptions = LlmNewsResearchLab._execution_assumptions(
            market="stocks",
            timeframe="day",
            regular_hours_only=False,
        )

        per_symbol_results = []
        sector_thresholds_payload = {
            sector: {str(horizon): values for horizon, values in by_horizon.items()}
            for sector, by_horizon in thresholds.items()
        }
        for symbol in TOP5_SYMBOLS:
            sector = SYMBOL_TO_SECTOR[symbol]
            train_rows = [row for row in rows if row.sector == sector and row.split == "train"]
            validation_rows = [row for row in rows if row.symbol == symbol and row.split == "validation"]
            test_rows = [row for row in rows if row.symbol == symbol and row.split == "test"]

            fused_models = self._fit_horizon_models(train_rows, fused=True)
            price_models = self._fit_horizon_models(train_rows, fused=False)

            fused_validation = self._predict_aggregate(validation_rows, fused_models, fused=True)
            price_validation = self._predict_aggregate(validation_rows, price_models, fused=False)
            fused_test = self._predict_aggregate(test_rows, fused_models, fused=True)
            price_test = self._predict_aggregate(test_rows, price_models, fused=False)

            fused_metrics = self._classification_metrics(test_rows, fused_test)
            price_metrics = self._classification_metrics(test_rows, price_test)
            fused_trade = self._trade_summary(
                symbol=symbol,
                bars=bars_by_symbol[symbol],
                split_by_day=split_by_day,
                predictions=fused_test,
                assumptions=assumptions,
                strategy_name=f"{symbol.lower()}_v4_fused",
            )
            price_trade = self._trade_summary(
                symbol=symbol,
                bars=bars_by_symbol[symbol],
                split_by_day=split_by_day,
                predictions=price_test,
                assumptions=assumptions,
                strategy_name=f"{symbol.lower()}_v4_price_only",
            )

            per_symbol_results.append(
                {
                    "symbol": symbol,
                    "sector": sector,
                    "validation_samples": len(validation_rows),
                    "test_samples": len(test_rows),
                    "fused_validation_accuracy": self._classification_metrics(validation_rows, fused_validation)["direction_accuracy"],
                    "price_validation_accuracy": self._classification_metrics(validation_rows, price_validation)["direction_accuracy"],
                    "fused_test_metrics": fused_metrics,
                    "price_test_metrics": price_metrics,
                    "fused_trade_summary": fused_trade,
                    "price_trade_summary": price_trade,
                }
            )

        result = {
            "version": "V4",
            "goal_directional_accuracy": 0.5,
            "base_model_path": base_model_path,
            "symbols": TOP5_SYMBOLS,
            "sectors": SECTOR_UNIVERSES,
            "date_splits": self._split_summary(split_by_day),
            "rows_total": len(rows),
            "neutral_band": neutral_band,
            "horizons": list(HORIZONS),
            "sector_thresholds": sector_thresholds_payload,
            "per_symbol_results": per_symbol_results,
            "aggregate": self._aggregate(per_symbol_results),
        }
        output_path = output_dir / "deepseek_directional_v4_run.json"
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _load_bars(self, *, symbols: list[str], max_bars: int) -> dict[str, list[BacktestBar]]:
        stock_bars = self._bars_client.get_recent_bars(symbols, max_bars, timeframe="day")
        return {
            symbol: [
                BacktestBar(
                    timestamp=bar.timestamp.isoformat(),
                    open=float(bar.open),
                    close=float(bar.close),
                    high=float(bar.high),
                    low=float(bar.low),
                    volume=float(getattr(bar, "volume", 0.0) or 0.0),
                )
                for bar in values
            ]
            for symbol, values in stock_bars.items()
        }

    def _load_news(self, *, symbol: str, bars: list[BacktestBar]) -> list[object]:
        start = datetime.fromisoformat(bars[0].timestamp).astimezone(timezone.utc) - timedelta(days=3)
        end = datetime.fromisoformat(bars[-1].timestamp).astimezone(timezone.utc) + timedelta(days=1)
        return self._news_client.get_news_articles(symbol=symbol, start=start, end=end, include_content=False)

    @staticmethod
    def _build_date_splits(bars: list[BacktestBar]) -> dict[str, str]:
        trading_days = [datetime.fromisoformat(bar.timestamp).date().isoformat() for bar in bars]
        unique_days = sorted(dict.fromkeys(trading_days))
        train_end = int(len(unique_days) * 0.6)
        validation_end = int(len(unique_days) * 0.8)
        split_by_day: dict[str, str] = {}
        for index, trading_day in enumerate(unique_days):
            if index < train_end:
                split_by_day[trading_day] = "train"
            elif index < validation_end:
                split_by_day[trading_day] = "validation"
            else:
                split_by_day[trading_day] = "test"
        return split_by_day

    def _build_rows(
        self,
        *,
        bars_by_symbol: dict[str, list[BacktestBar]],
        benchmark_bars: list[BacktestBar],
        split_by_day: dict[str, str],
        sentiment_by_symbol: dict[str, dict[str, object]],
        article_count_by_symbol: dict[str, dict[str, int]],
    ) -> list[V4Row]:
        benchmark_by_day = {
            datetime.fromisoformat(bar.timestamp).date().isoformat(): float(bar.close)
            for bar in benchmark_bars
        }
        rows: list[V4Row] = []
        for sector, symbols in SECTOR_UNIVERSES.items():
            for symbol in symbols:
                bars = bars_by_symbol[symbol]
                days = [datetime.fromisoformat(bar.timestamp).date().isoformat() for bar in bars]
                closes = [float(bar.close) for bar in bars]
                volumes = [float(bar.volume or 0.0) for bar in bars]
                sentiment_map = sentiment_by_symbol[symbol]
                article_map = article_count_by_symbol[symbol]
                for index in range(20, len(bars) - max(HORIZONS)):
                    trading_day = days[index]
                    split = split_by_day.get(trading_day)
                    if split is None:
                        continue
                    prompt_score = float(sentiment_map.get(trading_day).score if trading_day in sentiment_map else 0.0)
                    prompt_label = str(sentiment_map.get(trading_day).label if trading_day in sentiment_map else "neutral")
                    features_price = self._price_features(
                        closes=closes,
                        volumes=volumes,
                        index=index,
                    )
                    features_fused = {
                        **features_price,
                        "prompt_score": prompt_score,
                        "prompt_abs_score": abs(prompt_score),
                        "prompt_label_bullish": 1.0 if prompt_label == "bullish" else 0.0,
                        "prompt_label_bearish": 1.0 if prompt_label == "bearish" else 0.0,
                        "prompt_label_neutral": 1.0 if prompt_label == "neutral" else 0.0,
                        "news_present": 1.0 if trading_day in article_map else 0.0,
                        "headline_count": float(article_map.get(trading_day, 0)),
                        "prompt_3d_mean": self._rolling_prompt_mean(days, sentiment_map, index=index, lookback=3),
                        "prompt_5d_mean": self._rolling_prompt_mean(days, sentiment_map, index=index, lookback=5),
                    }
                    forward_excess_returns = {}
                    for horizon in HORIZONS:
                        current_close = closes[index]
                        future_close = closes[index + horizon]
                        forward_return = (future_close / current_close) - 1.0 if current_close > 0 else 0.0
                        bench_start = benchmark_by_day.get(trading_day, current_close)
                        bench_end = benchmark_by_day.get(days[index + horizon], bench_start)
                        benchmark_return = (bench_end / bench_start) - 1.0 if bench_start > 0 else 0.0
                        forward_excess_returns[horizon] = forward_return - benchmark_return
                    rows.append(
                        V4Row(
                            symbol=symbol,
                            sector=sector,
                            trading_day=trading_day,
                            split=split,
                            features_fused=features_fused,
                            features_price_only=features_price,
                            targets={},
                            forward_excess_returns=forward_excess_returns,
                        )
                    )
        return rows

    @staticmethod
    def _price_features(*, closes: list[float], volumes: list[float], index: int) -> dict[str, float]:
        recent_20 = closes[index - 19 : index + 1]
        recent_10 = closes[index - 9 : index + 1]
        average_20 = sum(recent_20) / len(recent_20)
        average_10 = sum(recent_10) / len(recent_10)
        volume_recent = volumes[index - 19 : index + 1]
        volume_mean = sum(volume_recent) / len(volume_recent) if volume_recent else 0.0
        volume_std = DeepseekDirectionalV4ResearchLab._stdev(volume_recent)
        current_volume = volumes[index]
        returns_20 = [DeepseekDirectionalV4ResearchLab._return(closes, idx, 1) for idx in range(index - 19, index + 1)]
        return {
            "ret_1": DeepseekDirectionalV4ResearchLab._return(closes, index, 1),
            "ret_3": DeepseekDirectionalV4ResearchLab._return(closes, index, 3),
            "ret_5": DeepseekDirectionalV4ResearchLab._return(closes, index, 5),
            "ret_10": DeepseekDirectionalV4ResearchLab._return(closes, index, 10),
            "ret_20": DeepseekDirectionalV4ResearchLab._return(closes, index, 20),
            "dist_sma_10": (closes[index] / average_10) - 1.0 if average_10 > 0 else 0.0,
            "dist_sma_20": (closes[index] / average_20) - 1.0 if average_20 > 0 else 0.0,
            "vol_20": DeepseekDirectionalV4ResearchLab._stdev(returns_20),
            "volume_z": ((current_volume - volume_mean) / volume_std) if volume_std > 0 else 0.0,
        }

    @staticmethod
    def _rolling_prompt_mean(days: list[str], sentiment_map: dict[str, object], *, index: int, lookback: int) -> float:
        values = []
        for offset in range(max(0, index - lookback + 1), index + 1):
            trading_day = days[offset]
            if trading_day in sentiment_map:
                values.append(float(sentiment_map[trading_day].score))
        return sum(values) / len(values) if values else 0.0

    def _compute_thresholds(self, *, rows: list[V4Row], neutral_band: float) -> dict[str, dict[int, tuple[float, float]]]:
        thresholds: dict[str, dict[int, tuple[float, float]]] = {}
        lower_q = (1.0 - neutral_band) / 2.0
        upper_q = 1.0 - lower_q
        for sector in SECTOR_UNIVERSES:
            thresholds[sector] = {}
            sector_train_rows = [row for row in rows if row.sector == sector and row.split == "train"]
            for horizon in HORIZONS:
                values = sorted(row.forward_excess_returns[horizon] for row in sector_train_rows)
                if len(values) < 10:
                    thresholds[sector][horizon] = (-0.01, 0.01)
                    continue
                lower_index = max(0, min(len(values) - 1, int(lower_q * (len(values) - 1))))
                upper_index = max(0, min(len(values) - 1, int(upper_q * (len(values) - 1))))
                thresholds[sector][horizon] = (values[lower_index], values[upper_index])
        return thresholds

    def _apply_targets(
        self,
        *,
        rows: list[V4Row],
        thresholds: dict[str, dict[int, tuple[float, float]]],
    ) -> list[V4Row]:
        resolved = []
        for row in rows:
            targets = {}
            for horizon, value in row.forward_excess_returns.items():
                lower, upper = thresholds[row.sector][horizon]
                if value <= lower:
                    targets[horizon] = "bearish"
                elif value >= upper:
                    targets[horizon] = "bullish"
                else:
                    targets[horizon] = "neutral"
            resolved.append(
                V4Row(
                    symbol=row.symbol,
                    sector=row.sector,
                    trading_day=row.trading_day,
                    split=row.split,
                    features_fused=row.features_fused,
                    features_price_only=row.features_price_only,
                    targets=targets,
                    forward_excess_returns=row.forward_excess_returns,
                )
            )
        return resolved

    @staticmethod
    def _fit_horizon_models(rows: list[V4Row], *, fused: bool) -> dict[int, Pipeline]:
        models = {}
        for horizon in HORIZONS:
            features = [row.features_fused if fused else row.features_price_only for row in rows]
            labels = [row.targets[horizon] for row in rows]
            feature_names = sorted(features[0].keys()) if features else []
            matrix = [[feature.get(name, 0.0) for name in feature_names] for feature in features]
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            )
            pipeline.fit(matrix, labels)
            pipeline.feature_names_ = feature_names  # type: ignore[attr-defined]
            models[horizon] = pipeline
        return models

    @staticmethod
    def _predict_aggregate(rows: list[V4Row], models: dict[int, Pipeline], *, fused: bool) -> dict[str, dict[str, float | str]]:
        predictions = {}
        for row in rows:
            combined = [0.0, 0.0, 0.0]
            for horizon, model in models.items():
                feature = row.features_fused if fused else row.features_price_only
                feature_names = model.feature_names_  # type: ignore[attr-defined]
                matrix = [[feature.get(name, 0.0) for name in feature_names]]
                probabilities = model.predict_proba(matrix)[0]
                class_index = {label: idx for idx, label in enumerate(model.classes_)}
                weight = HORIZON_WEIGHTS[horizon]
                for label in CLASS_ORDER:
                    combined[CLASS_TO_INDEX[label]] += weight * float(probabilities[class_index[label]])
            best_index = max(range(len(combined)), key=lambda idx: combined[idx])
            predictions[row.trading_day] = {
                "direction": CLASS_ORDER[best_index],
                "confidence": combined[best_index],
                "bullish_prob": combined[CLASS_TO_INDEX["bullish"]],
                "neutral_prob": combined[CLASS_TO_INDEX["neutral"]],
                "bearish_prob": combined[CLASS_TO_INDEX["bearish"]],
            }
        return predictions

    @staticmethod
    def _classification_metrics(rows: list[V4Row], predictions: dict[str, dict[str, float | str]]) -> dict[str, float]:
        actual = [row.targets[5] for row in rows]
        predicted = [str(predictions[row.trading_day]["direction"]) for row in rows]
        return {
            "direction_accuracy": round(float(accuracy_score(actual, predicted)), 6),
            "direction_macro_f1": round(float(f1_score(actual, predicted, labels=CLASS_ORDER, average="macro", zero_division=0)), 6),
        }

    def _trade_summary(
        self,
        *,
        symbol: str,
        bars: list[BacktestBar],
        split_by_day: dict[str, str],
        predictions: dict[str, dict[str, float | str]],
        assumptions,
        strategy_name: str,
    ) -> dict[str, object]:
        test_bars = [
            bar
            for bar in bars
            if split_by_day.get(datetime.fromisoformat(bar.timestamp).date().isoformat()) == "test"
        ]
        directional_predictions = {}
        for bar in test_bars:
            trading_day = datetime.fromisoformat(bar.timestamp).date().isoformat()
            payload = predictions.get(
                trading_day,
                {
                    "direction": "neutral",
                    "confidence": 0.34,
                    "bullish_prob": 0.33,
                    "neutral_prob": 0.34,
                    "bearish_prob": 0.33,
                },
            )
            confidence = float(payload["confidence"])
            if confidence >= 0.6:
                strength = "high"
            elif confidence >= 0.45:
                strength = "medium"
            else:
                strength = "low"
            directional_predictions[trading_day] = DirectionalPrediction(
                symbol=symbol,
                trading_day=trading_day,
                direction=str(payload["direction"]),
                strength=strength,
                relative_to_spy="outperform" if str(payload["direction"]) == "bullish" else ("underperform" if str(payload["direction"]) == "bearish" else "inline"),
                event_type="other",
                session="mixed",
                raw_completion="",
            )
        definition = DirectionalTradeDefinition(
            name=strategy_name,
            minimum_strength="medium",
            require_relative_confirmation=False,
            require_momentum_confirmation=False,
        )
        trace = build_directional_trade_trace_with_momentum_shift_entry(
            bars=test_bars,
            predictions_by_day=directional_predictions,
            definition=definition,
            risk_limits=self._risk_limits,
        )
        replay = run_execution_aware_replay(test_bars, trace, assumptions)
        return replay["summary"]

    @staticmethod
    def _aggregate(results: list[dict[str, object]]) -> dict[str, object]:
        fused_acc = [float(row["fused_test_metrics"]["direction_accuracy"]) for row in results]
        price_acc = [float(row["price_test_metrics"]["direction_accuracy"]) for row in results]
        fused_f1 = [float(row["fused_test_metrics"]["direction_macro_f1"]) for row in results]
        price_f1 = [float(row["price_test_metrics"]["direction_macro_f1"]) for row in results]
        fused_ret = [float(row["fused_trade_summary"]["total_return"]) for row in results]
        price_ret = [float(row["price_trade_summary"]["total_return"]) for row in results]
        return {
            "fused_mean_direction_accuracy": round(sum(fused_acc) / len(fused_acc), 6),
            "price_only_mean_direction_accuracy": round(sum(price_acc) / len(price_acc), 6),
            "fused_mean_macro_f1": round(sum(fused_f1) / len(fused_f1), 6),
            "price_only_mean_macro_f1": round(sum(price_f1) / len(price_f1), 6),
            "fused_mean_trade_return": round(sum(fused_ret) / len(fused_ret), 6),
            "price_only_mean_trade_return": round(sum(price_ret) / len(price_ret), 6),
            "fused_beats_price_only_accuracy": sum(
                1 for row in results
                if float(row["fused_test_metrics"]["direction_accuracy"]) > float(row["price_test_metrics"]["direction_accuracy"])
            ),
            "fused_hits_goal_symbols": sum(
                1 for row in results
                if float(row["fused_test_metrics"]["direction_accuracy"]) >= 0.5
            ),
        }

    @staticmethod
    def _split_summary(split_by_day: dict[str, str]) -> dict[str, str]:
        by_split: dict[str, list[str]] = defaultdict(list)
        for trading_day, split in split_by_day.items():
            by_split[split].append(trading_day)
        return {
            split: f"{days[0]} -> {days[-1]} ({len(days)} days)"
            for split, days in by_split.items()
        }

    @staticmethod
    def _return(prices: list[float], index: int, lookback: int) -> float:
        previous_index = max(0, index - lookback)
        previous = prices[previous_index]
        current = prices[index]
        return (current / previous) - 1.0 if previous > 0 else 0.0

    @staticmethod
    def _stdev(values: list[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return variance ** 0.5
