from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from algoding.data.historical import AlpacaHistoricalClient
from algoding.data.news import AlpacaNewsHistoricalClient
from algoding.execution.storage import OrderStore
from algoding.portfolio.risk import RiskLimits
from algoding.research.cost_aware_backtest import ExecutionAssumptions, run_execution_aware_replay
from algoding.research.deepseek_directional_model import (
    DeepseekDirectionalScorer,
    DirectionalTradeDefinition,
    build_directional_trade_trace,
    evaluate_directional_predictions,
)
from algoding.research.deepseek_lora_trainer import DeepseekLoraTrainer, DeepseekLoraTrainingConfig
from algoding.research.historical_backtest import BacktestBar
from algoding.research.news_directional_labeler import build_directional_examples
from algoding.research.news_labeler import build_bundle_metadata
from algoding.settings import Settings


DEFAULT_CRYPTO_TRAIN_SYMBOLS = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "DOGE/USD",
    "LTC/USD",
    "BCH/USD",
    "AVAX/USD",
    "LINK/USD",
    "UNI/USD",
    "AAVE/USD",
    "XRP/USD",
]

DEFAULT_CRYPTO_EVAL_SYMBOLS = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "DOGE/USD",
    "XRP/USD",
]


class DeepseekDirectionalCryptoResearchLab:
    def __init__(self, settings: Settings) -> None:
        if not settings.has_alpaca_credentials:
            raise RuntimeError("Alpaca credentials are required in .env for crypto directional research.")
        self._settings = settings
        self._bars_client = AlpacaHistoricalClient(settings)
        self._news_client = AlpacaNewsHistoricalClient(settings)
        self._store = OrderStore(settings)
        self._risk_limits = RiskLimits.from_file().with_overrides(
            stop_loss_pct=0.05,
            trailing_stop_pct=0.01,
        )

    def run(
        self,
        *,
        training_symbols: list[str] | None = None,
        evaluation_symbols: list[str] | None = None,
        base_model_path: str = r"E:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B",
        output_root: str = "reports/research/deepseek_directional_crypto_cv0",
        max_bars: int = 800,
        num_train_epochs: float = 1.0,
        max_train_samples: int | None = 1200,
        max_eval_samples: int | None = 300,
        scoring_batch_size: int = 4,
    ) -> dict[str, object]:
        train_symbols = training_symbols or DEFAULT_CRYPTO_TRAIN_SYMBOLS
        eval_symbols = evaluation_symbols or DEFAULT_CRYPTO_EVAL_SYMBOLS
        all_symbols = sorted({*train_symbols, *eval_symbols})
        if "BTC/USD" not in all_symbols:
            all_symbols.append("BTC/USD")
        output_dir = Path(output_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._store.ensure_schema()

        bars_by_symbol = self._load_bars(all_symbols=all_symbols, max_bars=max_bars)
        benchmark_bars = bars_by_symbol["BTC/USD"]
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

        base_scorer = DeepseekDirectionalScorer(
            base_model_path=base_model_path,
            adapter_path=None,
            cache_path=output_dir / "base_directional_scores.jsonl",
        )
        finetuned_scorer = DeepseekDirectionalScorer(
            base_model_path=base_model_path,
            adapter_path=training_result["output_dir"],
            cache_path=output_dir / "finetuned_directional_scores.jsonl",
        )
        assumptions = self._execution_assumptions()

        comparison_rows: list[dict[str, object]] = []
        for symbol in eval_symbols:
            bundles = [
                bundle
                for bundle in bundles_by_symbol[symbol]
                if split_by_day.get(bundle.trading_day) in {"validation", "test"}
            ]
            base_predictions = base_scorer.score_bundles(bundles, batch_size=scoring_batch_size)
            finetuned_predictions = finetuned_scorer.score_bundles(bundles, batch_size=scoring_batch_size)
            validation_symbol_examples = [
                example for example in examples_by_symbol[symbol] if split_by_day.get(example.trading_day) == "validation"
            ]
            test_symbol_examples = [
                example for example in examples_by_symbol[symbol] if split_by_day.get(example.trading_day) == "test"
            ]
            validation_bars = self._subset_bars(bars_by_symbol[symbol], split_by_day, "validation")
            test_bars = self._subset_bars(bars_by_symbol[symbol], split_by_day, "test")

            base_validation_metrics = evaluate_directional_predictions(
                examples=validation_symbol_examples,
                predictions_by_day=base_predictions,
            )
            base_test_metrics = evaluate_directional_predictions(
                examples=test_symbol_examples,
                predictions_by_day=base_predictions,
            )
            finetuned_validation_metrics = evaluate_directional_predictions(
                examples=validation_symbol_examples,
                predictions_by_day=finetuned_predictions,
            )
            finetuned_test_metrics = evaluate_directional_predictions(
                examples=test_symbol_examples,
                predictions_by_day=finetuned_predictions,
            )

            definition = DirectionalTradeDefinition(
                name=f"{symbol.lower().replace('/', '_')}_c_v_0",
                minimum_strength="medium",
                require_relative_confirmation=False,
                require_momentum_confirmation=False,
            )
            base_trace = build_directional_trade_trace(
                bars=test_bars,
                predictions_by_day=base_predictions,
                definition=definition,
                risk_limits=self._risk_limits,
                no_same_day_reentry=True,
            )
            finetuned_trace = build_directional_trade_trace(
                bars=test_bars,
                predictions_by_day=finetuned_predictions,
                definition=definition,
                risk_limits=self._risk_limits,
                no_same_day_reentry=True,
            )
            base_replay = run_execution_aware_replay(test_bars, base_trace, assumptions)
            finetuned_replay = run_execution_aware_replay(test_bars, finetuned_trace, assumptions)
            base_replay["summary"]["market"] = "crypto"
            finetuned_replay["summary"]["market"] = "crypto"
            base_replay["summary"]["llm_model_name"] = base_model_path
            finetuned_replay["summary"]["llm_model_name"] = base_model_path
            self._store.record_strategy_run(base_replay["summary"])
            self._store.record_strategy_run(finetuned_replay["summary"])

            comparison_rows.append(
                {
                    "symbol": symbol,
                    "validation": {
                        "base_metrics": base_validation_metrics,
                        "finetuned_metrics": finetuned_validation_metrics,
                    },
                    "test": {
                        "base_metrics": base_test_metrics,
                        "base_trade_summary": base_replay["summary"],
                        "finetuned_metrics": finetuned_test_metrics,
                        "finetuned_trade_summary": finetuned_replay["summary"],
                    },
                }
            )

        result = {
            "version": "C_V-0",
            "plan": {
                "objective": "Train a crypto directional DeepSeek branch similar to V3-5 and evaluate it with the same loose-medium strength trade policy.",
                "workflow": [
                    "Use a broad crypto basket for train/validation/test splits on daily bars.",
                    "Use BTC/USD as the crypto-market benchmark for relative labeling.",
                    "Fine-tune DeepSeek on crypto news bundles and future directional labels.",
                    "Evaluate directional accuracy first, then replay C_V-0 with V3-5-style risk rules.",
                    "Compare base vs fine-tuned DeepSeek on the top crypto evaluation basket.",
                ],
            },
            "training_symbols": train_symbols,
            "evaluation_symbols": eval_symbols,
            "base_model_path": base_model_path,
            "max_bars": max_bars,
            "policy": {
                "minimum_strength": "medium",
                "require_relative_confirmation": False,
                "require_momentum_confirmation": False,
                "stop_loss_pct": self._risk_limits.stop_loss_pct,
                "trailing_stop_pct": self._risk_limits.trailing_stop_pct,
                "no_same_day_reentry": True,
            },
            "training_config": {
                "num_train_epochs": num_train_epochs,
                "max_train_samples": max_train_samples,
                "max_eval_samples": max_eval_samples,
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
        output_path = output_dir / "deepseek_directional_crypto_run.json"
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        (output_dir / "deepseek_directional_crypto_summary.md").write_text(
            self._build_summary_markdown(result),
            encoding="utf-8",
        )
        return result

    def _load_bars(self, *, all_symbols: list[str], max_bars: int) -> dict[str, list[BacktestBar]]:
        crypto_bars = self._bars_client.get_recent_crypto_bars(all_symbols, max_bars, timeframe="day")
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
            for symbol, values in crypto_bars.items()
            if values
        }

    def _load_news(self, *, symbol: str, bars: list[BacktestBar]) -> list[object]:
        start = datetime.fromisoformat(bars[0].timestamp) - timedelta(days=3)
        end = datetime.fromisoformat(bars[-1].timestamp) + timedelta(days=1)
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
            split: (
                f"{days[0]} -> {days[-1]} ({len(days)} days)" if days else ""
            )
            for split, days in by_split.items()
        }

    @staticmethod
    def _execution_assumptions() -> ExecutionAssumptions:
        return ExecutionAssumptions(
            starting_capital=100_000.0,
            commission_per_order=0.0,
            buy_fee_bps=25.0,
            sell_fee_bps=25.0,
            quoted_spread_bps=3.0,
            market_impact_bps=2.0,
            stop_extra_slippage_bps=6.0,
            max_bar_participation_rate=0.001,
            max_bar_shares=None,
            sec_fee_per_million_sell=0.0,
            finra_taf_per_share_sell=0.0,
            finra_taf_cap_per_trade=0.0,
        )

    @staticmethod
    def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
        base_accuracy = []
        finetuned_accuracy = []
        base_returns = []
        finetuned_returns = []
        base_drawdowns = []
        finetuned_drawdowns = []
        wins = 0
        for row in rows:
            base_accuracy.append(float(row["test"]["base_metrics"]["direction_accuracy"]))
            finetuned_accuracy.append(float(row["test"]["finetuned_metrics"]["direction_accuracy"]))
            base_returns.append(float(row["test"]["base_trade_summary"]["total_return"]))
            finetuned_returns.append(float(row["test"]["finetuned_trade_summary"]["total_return"]))
            base_drawdowns.append(float(row["test"]["base_trade_summary"]["max_drawdown"]))
            finetuned_drawdowns.append(float(row["test"]["finetuned_trade_summary"]["max_drawdown"]))
            if float(row["test"]["finetuned_trade_summary"]["total_return"]) > float(row["test"]["base_trade_summary"]["total_return"]):
                wins += 1
        return {
            "symbols": len(rows),
            "base_mean_direction_accuracy": round(sum(base_accuracy) / len(base_accuracy), 6) if base_accuracy else 0.0,
            "finetuned_mean_direction_accuracy": round(sum(finetuned_accuracy) / len(finetuned_accuracy), 6) if finetuned_accuracy else 0.0,
            "base_mean_return": round(sum(base_returns) / len(base_returns), 6) if base_returns else 0.0,
            "finetuned_mean_return": round(sum(finetuned_returns) / len(finetuned_returns), 6) if finetuned_returns else 0.0,
            "base_mean_max_drawdown": round(sum(base_drawdowns) / len(base_drawdowns), 6) if base_drawdowns else 0.0,
            "finetuned_mean_max_drawdown": round(sum(finetuned_drawdowns) / len(finetuned_drawdowns), 6) if finetuned_drawdowns else 0.0,
            "finetuned_beats_base_on_return": wins,
        }

    @staticmethod
    def _build_summary_markdown(result: dict[str, object]) -> str:
        lines = [
            "# C_V-0 Crypto Directional Run",
            "",
            "Policy:",
            "",
            "- minimum strength: `medium`",
            "- relative confirmation: `false`",
            "- momentum confirmation: `false`",
            "- stop loss: `5%`",
            "- trailing stop: `1.0%`",
            "- no same-day re-entry: `true`",
            "",
            "## Per Symbol",
            "",
            "| Symbol | Base accuracy | Fine-tuned accuracy | Base return | Fine-tuned return | Base max drawdown | Fine-tuned max drawdown |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in result["comparison_rows"]:
            lines.append(
                f"| `{row['symbol']}` | "
                f"`{float(row['test']['base_metrics']['direction_accuracy']):.2%}` | "
                f"`{float(row['test']['finetuned_metrics']['direction_accuracy']):.2%}` | "
                f"`{float(row['test']['base_trade_summary']['total_return']):.2%}` | "
                f"`{float(row['test']['finetuned_trade_summary']['total_return']):.2%}` | "
                f"`{float(row['test']['base_trade_summary']['max_drawdown']):.2%}` | "
                f"`{float(row['test']['finetuned_trade_summary']['max_drawdown']):.2%}` |"
            )
        lines.extend(
            [
                "",
                "## Aggregate",
                "",
                f"- base mean direction accuracy: `{float(result['aggregates']['base_mean_direction_accuracy']):.2%}`",
                f"- fine-tuned mean direction accuracy: `{float(result['aggregates']['finetuned_mean_direction_accuracy']):.2%}`",
                f"- base mean return: `{float(result['aggregates']['base_mean_return']):.2%}`",
                f"- fine-tuned mean return: `{float(result['aggregates']['finetuned_mean_return']):.2%}`",
                f"- base mean max drawdown: `{float(result['aggregates']['base_mean_max_drawdown']):.2%}`",
                f"- fine-tuned mean max drawdown: `{float(result['aggregates']['finetuned_mean_max_drawdown']):.2%}`",
                f"- fine-tuned beats base on return: `{result['aggregates']['finetuned_beats_base_on_return']}/{result['aggregates']['symbols']}`",
            ]
        )
        return "\n".join(lines) + "\n"
