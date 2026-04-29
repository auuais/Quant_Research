from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from algoding.data.historical import AlpacaHistoricalClient
from algoding.data.news import AlpacaNewsHistoricalClient
from algoding.execution.deepseek_news_research import TOP_US_STOCK_SYMBOLS
from algoding.execution.llm_research import DEFAULT_BROAD_STOCK_SYMBOLS, LlmNewsResearchLab
from algoding.execution.storage import OrderStore
from algoding.portfolio.risk import RiskLimits
from algoding.research.cost_aware_backtest import run_execution_aware_replay
from algoding.research.deepseek_directional_model import (
    DeepseekDirectionalScorer,
    build_directional_trade_trace,
    evaluate_directional_predictions,
    select_best_directional_trade_definition,
)
from algoding.research.deepseek_lora_trainer import DeepseekLoraTrainer, DeepseekLoraTrainingConfig
from algoding.research.historical_backtest import BacktestBar
from algoding.research.news_directional_labeler import build_directional_examples
from algoding.research.news_labeler import build_bundle_metadata
from algoding.settings import Settings


class DeepseekDirectionalResearchLab:
    def __init__(self, settings: Settings) -> None:
        if not settings.has_alpaca_credentials:
            raise RuntimeError("Alpaca credentials are required in .env for DeepSeek directional research.")
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
        base_model_path: str = r"E:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B",
        output_root: str = "reports/research/deepseek_directional",
        max_bars: int = 800,
        num_train_epochs: float = 1.0,
        max_train_samples: int | None = 1200,
        max_eval_samples: int | None = 300,
        include_base_model: bool = True,
        scoring_batch_size: int = 4,
    ) -> dict[str, object]:
        train_symbols = training_symbols or DEFAULT_BROAD_STOCK_SYMBOLS
        eval_symbols = evaluation_symbols or TOP_US_STOCK_SYMBOLS
        all_symbols = sorted({*train_symbols, *eval_symbols})
        output_dir = Path(output_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._store.ensure_schema()

        bars_by_symbol = self._load_bars(all_symbols=all_symbols + ["SPY"], max_bars=max_bars)
        benchmark_bars = bars_by_symbol["SPY"]
        split_by_day = self._build_date_splits(benchmark_bars)
        bundles_by_symbol: dict[str, list[object]] = {}
        examples_by_symbol: dict[str, list[object]] = {}
        all_examples: list[object] = []

        for symbol in all_symbols:
            bars = bars_by_symbol[symbol]
            articles = self._load_news(symbol=symbol, bars=bars)
            trading_days = [datetime.fromisoformat(bar.timestamp).date() for bar in bars]
            bundles = build_bundle_metadata(symbol=symbol, articles=articles, trading_days=trading_days)
            examples = build_directional_examples(
                symbol=symbol,
                bundles=bundles,
                bars=bars,
                benchmark_bars=benchmark_bars,
            )
            bundles_by_symbol[symbol] = bundles
            examples_by_symbol[symbol] = examples
            all_examples.extend(examples)

        train_examples = [example for example in all_examples if split_by_day.get(example.trading_day) == "train"]
        validation_examples = [example for example in all_examples if split_by_day.get(example.trading_day) == "validation"]
        test_examples = [example for example in all_examples if split_by_day.get(example.trading_day) == "test"]

        trainer = DeepseekLoraTrainer(
            DeepseekLoraTrainingConfig(
                base_model_path=base_model_path,
                output_dir=str(output_dir / "deepseek_directional_lora"),
                num_train_epochs=num_train_epochs,
                max_train_samples=max_train_samples,
                max_eval_samples=max_eval_samples,
            )
        )
        adapter_dir = output_dir / "deepseek_directional_lora"
        if (adapter_dir / "adapter_model.safetensors").exists():
            training_result = {
                "output_dir": str(adapter_dir),
                "train_examples": min(len(train_examples), max_train_samples) if max_train_samples is not None else len(train_examples),
                "eval_examples": min(len(validation_examples), max_eval_samples) if max_eval_samples is not None else len(validation_examples),
                "train_runtime_seconds": 0.0,
                "train_loss": 0.0,
                "reused_existing_adapter": True,
            }
        else:
            training_result = trainer.fit(
                train_records=[asdict(example) for example in train_examples],
                eval_records=[asdict(example) for example in validation_examples],
            )

        base_scorer = (
            DeepseekDirectionalScorer(
                base_model_path=base_model_path,
                adapter_path=None,
                cache_path=output_dir / "base_directional_scores.jsonl",
            )
            if include_base_model
            else None
        )
        finetuned_scorer = DeepseekDirectionalScorer(
            base_model_path=base_model_path,
            adapter_path=training_result["output_dir"],
            cache_path=output_dir / "finetuned_directional_scores.jsonl",
        )
        assumptions = LlmNewsResearchLab._execution_assumptions(
            market="stocks",
            timeframe="day",
            regular_hours_only=False,
        )

        comparison_rows: list[dict[str, object]] = []
        for symbol in eval_symbols:
            bundles = [
                bundle
                for bundle in bundles_by_symbol[symbol]
                if split_by_day.get(bundle.trading_day) in {"validation", "test"}
            ]
            base_predictions = base_scorer.score_bundles(bundles, batch_size=scoring_batch_size) if base_scorer else {}
            finetuned_predictions = finetuned_scorer.score_bundles(bundles, batch_size=scoring_batch_size)
            validation_symbol_examples = [
                example for example in examples_by_symbol[symbol] if split_by_day.get(example.trading_day) == "validation"
            ]
            test_symbol_examples = [
                example for example in examples_by_symbol[symbol] if split_by_day.get(example.trading_day) == "test"
            ]
            validation_bars = self._subset_bars(bars_by_symbol[symbol], split_by_day, "validation")
            test_bars = self._subset_bars(bars_by_symbol[symbol], split_by_day, "test")

            base_validation_metrics = (
                evaluate_directional_predictions(
                    examples=validation_symbol_examples,
                    predictions_by_day=base_predictions,
                )
                if base_scorer
                else None
            )
            base_test_metrics = (
                evaluate_directional_predictions(
                    examples=test_symbol_examples,
                    predictions_by_day=base_predictions,
                )
                if base_scorer
                else None
            )
            finetuned_validation_metrics = evaluate_directional_predictions(
                examples=validation_symbol_examples,
                predictions_by_day=finetuned_predictions,
            )
            finetuned_test_metrics = evaluate_directional_predictions(
                examples=test_symbol_examples,
                predictions_by_day=finetuned_predictions,
            )

            if base_scorer:
                base_definition, _, base_validation_trade_metrics = select_best_directional_trade_definition(
                    bars=validation_bars,
                    examples=validation_symbol_examples,
                    predictions_by_day=base_predictions,
                    risk_limits=self._risk_limits,
                    definition_prefix=f"{symbol.lower()}_deepseek_directional_base",
                )
            else:
                base_definition = None
                base_validation_trade_metrics = None
            finetuned_definition, _, finetuned_validation_trade_metrics = select_best_directional_trade_definition(
                bars=validation_bars,
                examples=validation_symbol_examples,
                predictions_by_day=finetuned_predictions,
                risk_limits=self._risk_limits,
                definition_prefix=f"{symbol.lower()}_deepseek_directional_finetuned",
            )
            base_test_trace = (
                build_directional_trade_trace(
                    bars=test_bars,
                    predictions_by_day=base_predictions,
                    definition=base_definition,
                    risk_limits=self._risk_limits,
                )
                if base_definition is not None
                else None
            )
            finetuned_test_trace = build_directional_trade_trace(
                bars=test_bars,
                predictions_by_day=finetuned_predictions,
                definition=finetuned_definition,
                risk_limits=self._risk_limits,
            )
            base_replay = run_execution_aware_replay(test_bars, base_test_trace, assumptions) if base_test_trace else None
            finetuned_replay = run_execution_aware_replay(test_bars, finetuned_test_trace, assumptions)

            comparison_rows.append(
                {
                    "symbol": symbol,
                    "validation": {
                        "base_metrics": base_validation_metrics,
                        "base_trade_definition": asdict(base_definition) if base_definition is not None else None,
                        "base_trade_signal_metrics": base_validation_trade_metrics,
                        "finetuned_metrics": finetuned_validation_metrics,
                        "finetuned_trade_definition": asdict(finetuned_definition),
                        "finetuned_trade_signal_metrics": finetuned_validation_trade_metrics,
                    },
                    "test": {
                        "base_metrics": base_test_metrics,
                        "base_trade_summary": base_replay["summary"] if base_replay is not None else None,
                        "finetuned_metrics": finetuned_test_metrics,
                        "finetuned_trade_summary": finetuned_replay["summary"],
                    },
                }
            )

        result = {
            "plan": {
                "objective": "Train DeepSeek for future trend direction correctness first, then map predictions to trades second.",
                "workflow": [
                    "Create future-direction labels from later realized returns with a volatility-scaled neutral band.",
                    "Fine-tune DeepSeek on those directional labels with chronological train data only.",
                    "Score validation and test on classification correctness before any trading metric.",
                    "Select trade gating rules on validation using prediction hit-rate and coverage.",
                    "Replay the chosen directional policy with the same execution-aware assumptions as the other branches.",
                ],
            },
            "training_symbols": train_symbols,
            "evaluation_symbols": eval_symbols,
            "base_model_path": base_model_path,
            "max_bars": max_bars,
            "training_config": {
                "num_train_epochs": num_train_epochs,
                "max_train_samples": max_train_samples,
                "max_eval_samples": max_eval_samples,
                "include_base_model": include_base_model,
                "scoring_batch_size": scoring_batch_size,
            },
            "date_splits": self._split_summary(split_by_day),
            "dataset_sizes": {
                "train_examples": len(train_examples),
                "validation_examples": len(validation_examples),
                "test_examples": len(test_examples),
            },
            "training_result": training_result,
            "comparison_rows": comparison_rows,
            "aggregates": self._aggregate(comparison_rows),
        }
        output_path = output_dir / "deepseek_directional_run.json"
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _load_bars(self, *, all_symbols: list[str], max_bars: int) -> dict[str, list[BacktestBar]]:
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

    @staticmethod
    def _subset_bars(bars: list[BacktestBar], split_by_day: dict[str, str], split: str) -> list[BacktestBar]:
        return [
            bar
            for bar in bars
            if split_by_day.get(datetime.fromisoformat(bar.timestamp).date().isoformat()) == split
        ]

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
    def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
        finetuned_direction_acc = [float(row["test"]["finetuned_metrics"]["direction_accuracy"]) for row in rows]
        finetuned_f1 = [float(row["test"]["finetuned_metrics"]["direction_macro_f1"]) for row in rows]
        finetuned_returns = [float(row["test"]["finetuned_trade_summary"]["total_return"]) for row in rows]
        finetuned_drawdowns = [float(row["test"]["finetuned_trade_summary"]["max_drawdown"]) for row in rows]
        base_rows = [row for row in rows if row["test"].get("base_metrics") and row["test"].get("base_trade_summary")]
        payload = {
            "finetuned_direction_accuracy_mean": round(sum(finetuned_direction_acc) / len(finetuned_direction_acc), 6),
            "finetuned_direction_macro_f1_mean": round(sum(finetuned_f1) / len(finetuned_f1), 6),
            "finetuned_trade_return_mean": round(sum(finetuned_returns) / len(finetuned_returns), 6),
            "finetuned_trade_drawdown_mean": round(sum(finetuned_drawdowns) / len(finetuned_drawdowns), 6),
        }
        if base_rows:
            base_direction_acc = [float(row["test"]["base_metrics"]["direction_accuracy"]) for row in base_rows]
            base_f1 = [float(row["test"]["base_metrics"]["direction_macro_f1"]) for row in base_rows]
            base_returns = [float(row["test"]["base_trade_summary"]["total_return"]) for row in base_rows]
            base_drawdowns = [float(row["test"]["base_trade_summary"]["max_drawdown"]) for row in base_rows]
            payload.update(
                {
                    "base_direction_accuracy_mean": round(sum(base_direction_acc) / len(base_direction_acc), 6),
                    "base_direction_macro_f1_mean": round(sum(base_f1) / len(base_f1), 6),
                    "base_trade_return_mean": round(sum(base_returns) / len(base_returns), 6),
                    "base_trade_drawdown_mean": round(sum(base_drawdowns) / len(base_drawdowns), 6),
                    "finetuned_beats_base_accuracy": sum(
                        1
                        for row in base_rows
                        if float(row["test"]["finetuned_metrics"]["direction_accuracy"]) > float(row["test"]["base_metrics"]["direction_accuracy"])
                    ),
                    "finetuned_beats_base_return": sum(
                        1
                        for row in base_rows
                        if float(row["test"]["finetuned_trade_summary"]["total_return"]) > float(row["test"]["base_trade_summary"]["total_return"])
                    ),
                }
            )
        return payload
