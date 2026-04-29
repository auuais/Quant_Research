from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from algoding.data.historical import AlpacaHistoricalClient
from algoding.data.news import AlpacaNewsHistoricalClient
from algoding.execution.deepseek_news_research import TOP_US_STOCK_SYMBOLS
from algoding.execution.llm_research import DEFAULT_BROAD_STOCK_SYMBOLS
from algoding.portfolio.risk import RiskLimits
from algoding.research.deepseek_directional_model import (
    DeepseekDirectionalScorer,
    DirectionalTradeDefinition,
    build_directional_trade_trace,
    evaluate_directional_predictions,
)
from algoding.research.deepseek_lora_trainer import DeepseekLoraTrainer, DeepseekLoraTrainingConfig
from algoding.research.historical_backtest import BacktestBar
from algoding.research.news_directional_hourly_labeler import (
    build_hourly_bundle_metadata,
    build_hourly_directional_examples,
    build_hourly_directional_prompt,
)
from algoding.settings import Settings


class DeepseekDirectionalV3HourlyLab:
    def __init__(self, settings: Settings) -> None:
        if not settings.has_alpaca_credentials:
            raise RuntimeError("Alpaca credentials are required in .env for DeepSeek hourly directional research.")
        self._settings = settings
        self._bars_client = AlpacaHistoricalClient(settings)
        self._news_client = AlpacaNewsHistoricalClient(settings)
        self._risk_limits = RiskLimits.from_file().with_overrides(stop_loss_pct=0.05, trailing_stop_pct=0.01)

    def run(
        self,
        *,
        training_symbols: list[str] | None = None,
        evaluation_symbols: list[str] | None = None,
        base_model_path: str = r"E:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B",
        output_root: str = "reports/research/deepseek_directional_v3_hourly",
        max_bars: int = 3500,
        num_train_epochs: float = 1.0,
        max_train_samples: int = 1200,
        max_eval_samples: int = 300,
        scoring_batch_size: int = 4,
    ) -> dict[str, object]:
        train_symbols = training_symbols or DEFAULT_BROAD_STOCK_SYMBOLS
        eval_symbols = evaluation_symbols or TOP_US_STOCK_SYMBOLS
        all_symbols = sorted({*train_symbols, *eval_symbols, "SPY"})
        output_dir = Path(output_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        bars_by_symbol = self._load_bars(all_symbols=all_symbols, max_bars=max_bars)
        split_by_key = self._build_timestamp_splits(bars_by_symbol["SPY"])
        benchmark_bars = bars_by_symbol["SPY"]
        bundles_by_symbol: dict[str, list[object]] = {}
        examples_by_symbol: dict[str, list[object]] = {}
        all_examples: list[object] = []

        for symbol in sorted(set(all_symbols) - {"SPY"}):
            bars = bars_by_symbol[symbol]
            articles = self._load_news(symbol=symbol, bars=bars)
            bar_timestamps = [datetime.fromisoformat(bar.timestamp) for bar in bars]
            bundles = build_hourly_bundle_metadata(
                symbol=symbol,
                articles=articles,
                bar_timestamps=bar_timestamps,
            )
            examples = build_hourly_directional_examples(
                symbol=symbol,
                bundles=bundles,
                bars=bars,
                benchmark_bars=benchmark_bars,
            )
            bundles_by_symbol[symbol] = bundles
            examples_by_symbol[symbol] = examples
            all_examples.extend(examples)

        train_examples = [example for example in all_examples if split_by_key.get(example.trading_day) == "train"]
        validation_examples = [example for example in all_examples if split_by_key.get(example.trading_day) == "validation"]
        test_examples = [example for example in all_examples if split_by_key.get(example.trading_day) == "test"]

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
                "train_examples": min(len(train_examples), max_train_samples),
                "eval_examples": min(len(validation_examples), max_eval_samples),
                "train_runtime_seconds": 0.0,
                "train_loss": 0.0,
                "reused_existing_adapter": True,
            }
        else:
            training_result = trainer.fit(
                train_records=[asdict(example) for example in train_examples],
                eval_records=[asdict(example) for example in validation_examples],
            )

        scorer = DeepseekDirectionalScorer(
            base_model_path=base_model_path,
            adapter_path=training_result["output_dir"],
            cache_path=output_dir / "finetuned_directional_scores.jsonl",
        )

        comparison_rows: list[dict[str, object]] = []
        for symbol in eval_symbols:
            bundles = [bundle for bundle in bundles_by_symbol[symbol] if split_by_key.get(bundle.trading_key) in {"validation", "test"}]
            predictions = scorer.score_bundles(
                bundles,
                batch_size=scoring_batch_size,
                prompt_builder=build_hourly_directional_prompt,
            )
            test_symbol_examples = [example for example in examples_by_symbol[symbol] if split_by_key.get(example.trading_day) == "test"]
            test_symbol_bars = [bar for bar in bars_by_symbol[symbol] if split_by_key.get(bar.timestamp) == "test"]
            test_metrics = evaluate_directional_predictions(
                examples=test_symbol_examples,
                predictions_by_day=predictions,
            )
            trace = build_directional_trade_trace(
                bars=test_symbol_bars,
                predictions_by_day=predictions,
                definition=DirectionalTradeDefinition(
                    name=f"{symbol.lower()}_v3_5_hour",
                    minimum_strength="medium",
                    require_relative_confirmation=False,
                    require_momentum_confirmation=False,
                ),
                risk_limits=self._risk_limits,
                no_same_day_reentry=True,
                prediction_key_mode="timestamp",
            )
            comparison_rows.append(
                {
                    "symbol": symbol,
                    "test_metrics": test_metrics,
                    "test_trade_summary": trace["summary"],
                }
            )

        result = {
            "version": "V3-5-Hourly",
            "timeframe": "hour",
            "policy": {
                "minimum_strength": "medium",
                "require_relative_confirmation": False,
                "require_momentum_confirmation": False,
                "stop_loss_pct": 0.05,
                "trailing_stop_pct": 0.01,
                "no_same_day_reentry": True,
            },
            "training_symbols": train_symbols,
            "evaluation_symbols": eval_symbols,
            "base_model_path": base_model_path,
            "max_bars": max_bars,
            "training_config": {
                "num_train_epochs": num_train_epochs,
                "max_train_samples": max_train_samples,
                "max_eval_samples": max_eval_samples,
                "scoring_batch_size": scoring_batch_size,
            },
            "date_splits": self._split_summary(split_by_key),
            "dataset_sizes": {
                "train_examples": len(train_examples),
                "validation_examples": len(validation_examples),
                "test_examples": len(test_examples),
            },
            "training_result": training_result,
            "comparison_rows": comparison_rows,
            "aggregate": self._aggregate(comparison_rows),
        }
        output_path = output_dir / "deepseek_directional_hourly_run.json"
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        summary_path = output_dir / "deepseek_directional_hourly_summary.md"
        summary_path.write_text(self._build_summary(result), encoding="utf-8")
        return result

    def _load_bars(self, *, all_symbols: list[str], max_bars: int) -> dict[str, list[BacktestBar]]:
        stock_bars = self._bars_client.get_recent_bars(all_symbols, max_bars, timeframe="hour")
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
    def _build_timestamp_splits(bars: list[BacktestBar]) -> dict[str, str]:
        keys = sorted(dict.fromkeys(bar.timestamp for bar in bars))
        train_end = int(len(keys) * 0.6)
        validation_end = int(len(keys) * 0.8)
        split_by_key: dict[str, str] = {}
        for index, key in enumerate(keys):
            if index < train_end:
                split_by_key[key] = "train"
            elif index < validation_end:
                split_by_key[key] = "validation"
            else:
                split_by_key[key] = "test"
        return split_by_key

    @staticmethod
    def _split_summary(split_by_key: dict[str, str]) -> dict[str, str]:
        by_split: dict[str, list[str]] = defaultdict(list)
        for key, split in split_by_key.items():
            by_split[split].append(key)
        return {
            split: f"{keys[0]} -> {keys[-1]} ({len(keys)} bars)"
            for split, keys in by_split.items()
        }

    @staticmethod
    def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
        accuracies = [float(row["test_metrics"]["direction_accuracy"]) for row in rows]
        f1s = [float(row["test_metrics"]["direction_macro_f1"]) for row in rows]
        returns = [float(row["test_trade_summary"]["total_return"]) for row in rows]
        drawdowns = [float(row["test_trade_summary"]["max_drawdown"]) for row in rows]
        trades = [float(row["test_trade_summary"]["trades"]) for row in rows]
        return {
            "mean_direction_accuracy": round(sum(accuracies) / len(accuracies), 6),
            "mean_direction_macro_f1": round(sum(f1s) / len(f1s), 6),
            "mean_total_return": round(sum(returns) / len(returns), 6),
            "mean_max_drawdown": round(sum(drawdowns) / len(drawdowns), 6),
            "mean_trades": round(sum(trades) / len(trades), 2),
        }

    @staticmethod
    def _build_summary(result: dict[str, object]) -> str:
        lines = [
            "# V3-5 Hourly Directional Run",
            "",
            "Policy:",
            "",
            "- timeframe: `hour`",
            "- minimum strength: `medium`",
            "- relative confirmation: `false`",
            "- momentum confirmation: `false`",
            "- stop loss: `5%`",
            "- trailing stop: `1.0%`",
            "- no same-day re-entry: `true`",
            "",
            "## Per Symbol",
            "",
            "| Symbol | Direction accuracy | Macro F1 | Total return | Max drawdown | Trades |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in result["comparison_rows"]:
            lines.append(
                f"| `{row['symbol']}` | `{row['test_metrics']['direction_accuracy']:.2%}` | "
                f"`{row['test_metrics']['direction_macro_f1']:.4f}` | "
                f"`{row['test_trade_summary']['total_return']:.2%}` | "
                f"`{row['test_trade_summary']['max_drawdown']:.2%}` | "
                f"`{row['test_trade_summary']['trades']}` |"
            )
        lines.extend(
            [
                "",
                "## Aggregate",
                "",
                f"- mean direction accuracy: `{result['aggregate']['mean_direction_accuracy']:.2%}`",
                f"- mean macro F1: `{result['aggregate']['mean_direction_macro_f1']:.4f}`",
                f"- mean total return: `{result['aggregate']['mean_total_return']:.2%}`",
                f"- mean max drawdown: `{result['aggregate']['mean_max_drawdown']:.2%}`",
                f"- mean trades: `{result['aggregate']['mean_trades']}`",
            ]
        )
        return "\n".join(lines) + "\n"
