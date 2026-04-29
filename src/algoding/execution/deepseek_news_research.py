from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from algoding.data.historical import AlpacaHistoricalClient
from algoding.data.news import AlpacaNewsHistoricalClient
from algoding.execution.llm_research import DEFAULT_BROAD_STOCK_SYMBOLS, LlmNewsResearchLab
from algoding.execution.storage import OrderStore
from algoding.portfolio.risk import RiskLimits
from algoding.research.cost_aware_backtest import run_execution_aware_replay
from algoding.research.deepseek_finetune_dataset import export_finetune_jsonl
from algoding.research.deepseek_lora_trainer import (
    DeepseekLoraTrainer,
    DeepseekLoraTrainingConfig,
    FinetunedDeepseekStructuredScorer,
)
from algoding.research.historical_backtest import BacktestBar
from algoding.research.llm_meta_model import (
    LlmPriceMetaModel,
    MetaModelPrediction,
    build_meta_overlay_trace,
    select_best_thresholds,
)
from algoding.research.llm_overlay_backtest import build_llm_overlay_strategies, build_llm_overlay_trace
from algoding.research.llm_sentiment import LocalLlmNewsSentimentEngine
from algoding.research.news_labeler import build_bundle_metadata, build_labeled_examples
from algoding.research.news_price_feature_builder import build_feature_rows
from algoding.settings import Settings


TOP_US_STOCK_SYMBOLS = ["AVGO", "NVDA", "TSLA", "GOOGL", "XOM"]


class DeepseekNewsResearchLab:
    def __init__(self, settings: Settings) -> None:
        if not settings.has_alpaca_credentials:
            raise RuntimeError("Alpaca credentials are required in .env for DeepSeek research.")
        self._settings = settings
        self._bars_client = AlpacaHistoricalClient(settings)
        self._news_client = AlpacaNewsHistoricalClient(settings)
        self._store = OrderStore(settings)
        self._risk_limits = RiskLimits.from_file().with_overrides(
            stop_loss_pct=0.05,
            trailing_stop_pct=0.015,
        )

    def run(
        self,
        *,
        training_symbols: list[str] | None = None,
        evaluation_symbols: list[str] | None = None,
        base_model_path: str = r"U:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B",
        output_root: str = "reports/research/deepseek_news_price_meta",
        max_bars: int = 800,
    ) -> dict[str, object]:
        train_symbols = training_symbols or DEFAULT_BROAD_STOCK_SYMBOLS
        eval_symbols = evaluation_symbols or TOP_US_STOCK_SYMBOLS
        all_symbols = sorted({*train_symbols, *eval_symbols})
        output_dir = Path(output_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._store.ensure_schema()

        bars_by_symbol = self._load_bars(all_symbols=all_symbols, max_bars=max_bars)
        benchmark_bars = self._load_bars(all_symbols=["SPY"], max_bars=max_bars)["SPY"]
        split_by_day = self._build_date_splits(benchmark_bars)
        bundle_map: dict[str, list[object]] = {}
        labeled_examples: list[object] = []

        for symbol in all_symbols:
            bars = bars_by_symbol[symbol]
            articles = self._load_news(symbol=symbol, bars=bars)
            trading_days = [datetime.fromisoformat(bar.timestamp).date() for bar in bars]
            bundles = build_bundle_metadata(symbol=symbol, articles=articles, trading_days=trading_days)
            bundle_map[symbol] = bundles
            labeled_examples.extend(
                build_labeled_examples(
                    symbol=symbol,
                    bundles=bundles,
                    bars=bars,
                    benchmark_bars=benchmark_bars,
                )
            )

        train_examples = [example for example in labeled_examples if split_by_day.get(example.trading_day) == "train"]
        validation_examples = [example for example in labeled_examples if split_by_day.get(example.trading_day) == "validation"]
        test_examples = [example for example in labeled_examples if split_by_day.get(example.trading_day) == "test"]

        dataset_dir = output_dir / "datasets"
        train_jsonl = export_finetune_jsonl(train_examples, output_path=dataset_dir / "train.jsonl")
        validation_jsonl = export_finetune_jsonl(validation_examples, output_path=dataset_dir / "validation.jsonl")
        test_jsonl = export_finetune_jsonl(test_examples, output_path=dataset_dir / "test.jsonl")

        trainer = DeepseekLoraTrainer(
            DeepseekLoraTrainingConfig(
                base_model_path=base_model_path,
                output_dir=str(output_dir / "deepseek_news_lora"),
                num_train_epochs=1.0,
                max_train_samples=1200,
                max_eval_samples=300,
            )
        )
        training_result = trainer.fit(
            train_records=[asdict(example) for example in train_examples],
            eval_records=[asdict(example) for example in validation_examples],
        )

        scorer = FinetunedDeepseekStructuredScorer(
            base_model_path=base_model_path,
            adapter_path=training_result["output_dir"],
            cache_path=output_dir / "finetuned_scores.jsonl",
        )

        finetuned_scores_by_symbol: dict[str, dict[str, dict[str, object]]] = {}
        baseline_scores_by_symbol: dict[str, list[object]] = {}
        baseline_engine = self._build_baseline_engine()
        assumptions = LlmNewsResearchLab._execution_assumptions(
            market="stocks",
            timeframe="day",
            regular_hours_only=False,
        )

        for symbol in eval_symbols:
            bundles = bundle_map[symbol]
            finetuned_scores_by_symbol[symbol] = scorer.score_bundles(bundles, batch_size=2)
            baseline_bundles = baseline_engine.build_trading_day_bundles(
                symbol=symbol,
                articles=self._load_news(symbol=symbol, bars=bars_by_symbol[symbol]),
                trading_days=[datetime.fromisoformat(bar.timestamp).date() for bar in bars_by_symbol[symbol]],
            )
            bundle_scores = baseline_engine.score_bundles(baseline_bundles, batch_size=4)
            filled = baseline_engine.fill_missing_days(
                [datetime.fromisoformat(bar.timestamp).date() for bar in bars_by_symbol[symbol]],
                bundle_scores,
                symbol,
            )
            baseline_scores_by_symbol[symbol] = filled

        feature_rows = []
        for symbol in eval_symbols:
            feature_rows.extend(
                build_feature_rows(
                    symbol=symbol,
                    bars=bars_by_symbol[symbol],
                    benchmark_bars=benchmark_bars,
                    llm_outputs_by_day=finetuned_scores_by_symbol[symbol],
                    split_by_day=split_by_day,
                )
            )

        train_rows = [row for row in feature_rows if row.split == "train"]
        validation_rows = [row for row in feature_rows if row.split == "validation"]
        test_rows = [row for row in feature_rows if row.split == "test"]

        meta_model = LlmPriceMetaModel()
        meta_model.fit([row.features for row in train_rows], [row.target_long for row in train_rows])
        validation_predictions = meta_model.predict_probabilities([row.features for row in validation_rows])
        test_predictions = meta_model.predict_probabilities([row.features for row in test_rows])

        rows_by_symbol_and_split: dict[tuple[str, str], list[object]] = defaultdict(list)
        for row in feature_rows:
            rows_by_symbol_and_split[(row.symbol, row.split)].append(row)
        validation_probability_map = {
            (row.symbol, row.trading_day): probability
            for row, probability in zip(validation_rows, validation_predictions)
        }
        test_probability_map = {
            (row.symbol, row.trading_day): probability
            for row, probability in zip(test_rows, test_predictions)
        }

        comparison_rows: list[dict[str, object]] = []
        for symbol in eval_symbols:
            validation_symbol_rows = rows_by_symbol_and_split[(symbol, "validation")]
            test_symbol_rows = rows_by_symbol_and_split[(symbol, "test")]
            validation_bars = self._subset_bars(bars_by_symbol[symbol], split_by_day, "validation")
            test_bars = self._subset_bars(bars_by_symbol[symbol], split_by_day, "test")
            validation_meta_predictions = self._align_predictions_to_bars(
                bars=validation_bars,
                probability_map=validation_probability_map,
                symbol=symbol,
            )
            test_meta_predictions = self._align_predictions_to_bars(
                bars=test_bars,
                probability_map=test_probability_map,
                symbol=symbol,
            )
            best_meta_definition, _ = select_best_thresholds(
                bars=validation_bars,
                probabilities=validation_meta_predictions,
                risk_limits=self._risk_limits,
                definition_name=f"{symbol.lower()}_deepseek_meta",
            )
            meta_trace = build_meta_overlay_trace(
                bars=test_bars,
                predictions=test_meta_predictions,
                definition=best_meta_definition,
                risk_limits=self._risk_limits,
            )
            meta_replay = run_execution_aware_replay(test_bars, meta_trace, assumptions)

            baseline_replay = self._best_prompt_overlay_replay(
                symbol=symbol,
                validation_bars=validation_bars,
                test_bars=test_bars,
                baseline_scores=baseline_scores_by_symbol[symbol],
                assumptions=assumptions,
            )
            summary = {
                "symbol": symbol,
                "split": "test",
                "meta_strategy": meta_replay["summary"],
                "prompt_overlay_strategy": baseline_replay["summary"],
            }
            comparison_rows.append(summary)
            self._store.record_strategy_run(
                {
                    **meta_replay["summary"],
                    "symbol": symbol,
                    "market": "stocks",
                    "window_name": "strict_test",
                    "llm_model_name": base_model_path,
                    "run_type": "historical_deepseek_meta_execution_aware",
                }
            )

        aggregate = self._aggregate_results(comparison_rows)
        result = {
            "plan": self._plan_payload(),
            "training_symbols": train_symbols,
            "evaluation_symbols": eval_symbols,
            "date_splits": self._split_summary(split_by_day),
            "datasets": {
                "train_jsonl": train_jsonl,
                "validation_jsonl": validation_jsonl,
                "test_jsonl": test_jsonl,
                "train_examples": len(train_examples),
                "validation_examples": len(validation_examples),
                "test_examples": len(test_examples),
            },
            "training_result": training_result,
            "comparison_rows": comparison_rows,
            "aggregate": aggregate,
        }
        output_path = output_dir / "deepseek_news_price_meta_run.json"
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _build_baseline_engine(self) -> LocalLlmNewsSentimentEngine:
        baseline_settings = Settings()
        baseline_settings.alpaca_api_key = self._settings.alpaca_api_key
        baseline_settings.alpaca_secret_key = self._settings.alpaca_secret_key
        baseline_settings.llm_news_model_name = r"U:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B"
        baseline_settings.llm_news_cache_dir = Path("cache/llm_news_deepseek_baseline")
        return LocalLlmNewsSentimentEngine(baseline_settings)

    def _best_prompt_overlay_replay(
        self,
        *,
        symbol: str,
        validation_bars: list[object],
        test_bars: list[object],
        baseline_scores: list[object],
        assumptions,
    ) -> dict[str, object]:
        val_days = {datetime.fromisoformat(bar.timestamp).date().isoformat() for bar in validation_bars}
        test_days = {datetime.fromisoformat(bar.timestamp).date().isoformat() for bar in test_bars}
        validation_scores = [score for score in baseline_scores if score.trading_day in val_days]
        test_scores = [score for score in baseline_scores if score.trading_day in test_days]
        best_validation_score = float("-inf")
        best_definition = None
        for entry_threshold in [0.05, 0.1, 0.15, 0.2]:
            for exit_threshold in [0.0, 0.02, 0.05, 0.1]:
                for definition in build_llm_overlay_strategies(
                    symbol.lower(),
                    entry_sentiment_min=entry_threshold,
                    exit_sentiment_max=exit_threshold,
                ):
                    trace = build_llm_overlay_trace(
                        bars=validation_bars,
                        sentiment_scores=validation_scores,
                        definition=definition,
                        symbol=symbol,
                        market="stocks",
                        window_name="strict_validation",
                        risk_limits=self._risk_limits,
                    )
                    replay = run_execution_aware_replay(validation_bars, trace, assumptions)
                    if float(replay["summary"]["score"]) > best_validation_score:
                        best_validation_score = float(replay["summary"]["score"])
                        best_definition = definition
        if best_definition is None:
            raise RuntimeError(f"Unable to select prompt overlay for {symbol}.")
        test_trace = build_llm_overlay_trace(
            bars=test_bars,
            sentiment_scores=test_scores,
            definition=best_definition,
            symbol=symbol,
            market="stocks",
            window_name="strict_test",
            risk_limits=self._risk_limits,
        )
        return run_execution_aware_replay(test_bars, test_trace, assumptions)

    def _load_bars(self, *, all_symbols: list[str], max_bars: int) -> dict[str, list[object]]:
        stock_bars = self._bars_client.get_recent_bars(all_symbols, max_bars, timeframe="day")
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

    def _load_news(self, *, symbol: str, bars: list[object]) -> list[object]:
        start = datetime.fromisoformat(bars[0].timestamp).astimezone(timezone.utc) - timedelta(days=3)
        end = datetime.fromisoformat(bars[-1].timestamp).astimezone(timezone.utc) + timedelta(days=1)
        return self._news_client.get_news_articles(symbol=symbol, start=start, end=end, include_content=False)

    @staticmethod
    def _build_date_splits(bars: list[object]) -> dict[str, str]:
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

    @staticmethod
    def _subset_bars(bars: list[object], split_by_day: dict[str, str], split: str) -> list[object]:
        return [
            bar
            for bar in bars
            if split_by_day.get(datetime.fromisoformat(bar.timestamp).date().isoformat()) == split
        ]

    @staticmethod
    def _align_predictions_to_bars(
        *,
        bars: list[BacktestBar],
        probability_map: dict[tuple[str, str], float],
        symbol: str,
        default_probability: float = 0.5,
    ) -> list[MetaModelPrediction]:
        aligned: list[MetaModelPrediction] = []
        for bar in bars:
            trading_day = datetime.fromisoformat(bar.timestamp).date().isoformat()
            aligned.append(
                MetaModelPrediction(
                    trading_day=trading_day,
                    probability_long=float(probability_map.get((symbol, trading_day), default_probability)),
                )
            )
        return aligned

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
    def _aggregate_results(rows: list[dict[str, object]]) -> dict[str, object]:
        meta_returns = [float(row["meta_strategy"]["total_return"]) for row in rows]
        meta_drawdowns = [float(row["meta_strategy"]["max_drawdown"]) for row in rows]
        prompt_returns = [float(row["prompt_overlay_strategy"]["total_return"]) for row in rows]
        prompt_drawdowns = [float(row["prompt_overlay_strategy"]["max_drawdown"]) for row in rows]
        return {
            "meta_mean_total_return": round(sum(meta_returns) / len(meta_returns), 6),
            "meta_mean_max_drawdown": round(sum(meta_drawdowns) / len(meta_drawdowns), 6),
            "prompt_mean_total_return": round(sum(prompt_returns) / len(prompt_returns), 6),
            "prompt_mean_max_drawdown": round(sum(prompt_drawdowns) / len(prompt_drawdowns), 6),
            "symbols_meta_better": sum(
                1
                for row in rows
                if float(row["meta_strategy"]["total_return"]) > float(row["prompt_overlay_strategy"]["total_return"])
            ),
        }

    @staticmethod
    def _plan_payload() -> dict[str, object]:
        return {
            "improved_plan": [
                "Label news bundles with future excess returns instead of manual sentiment only.",
                "Fine-tune DeepSeek with QLoRA for structured finance outputs on chronological train data.",
                "Convert fine-tuned outputs into numeric features and combine them with price/volume state.",
                "Train a separate numeric meta-model on train data and tune thresholds on validation data.",
                "Evaluate strictly on later test dates with the same execution-aware replay used elsewhere in the repo.",
            ]
        }
