from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from algoding.execution.deepseek_news_research import DeepseekNewsResearchLab, TOP_US_STOCK_SYMBOLS
from algoding.execution.llm_research import LlmNewsResearchLab
from algoding.portfolio.risk import RiskLimits
from algoding.research.cost_aware_backtest import run_execution_aware_replay
from algoding.research.llm_overlay_backtest import LlmOverlayDefinition, build_llm_overlay_trace
from algoding.settings import Settings


PREVIOUS_REPORT_PATH = Path("reports/research/llm_news_sentiment_deepseek_stocks_report_full.json")
DEFAULT_META_ARTIFACT_PATH = Path("reports/research/deepseek_news_price_meta_full/deepseek_news_price_meta_run.json")
DEFAULT_OUTPUT_PATH = Path("reports/research/deepseek_top5_strict_exact_vs_meta.json")


@dataclass(frozen=True)
class ExactVariant:
    symbol: str
    strategy_name: str
    profitable_windows: int
    avg_total_return: float
    avg_max_drawdown: float
    lookback: int
    threshold: float
    entry_sentiment_min: float | None
    exit_sentiment_max: float | None


class DeepseekStrictComparisonLab:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._deepseek_lab = DeepseekNewsResearchLab(settings)
        self._risk_limits = RiskLimits.from_file().with_overrides(
            stop_loss_pct=0.05,
            trailing_stop_pct=0.015,
        )

    def run(
        self,
        *,
        previous_report_path: str = str(PREVIOUS_REPORT_PATH),
        meta_artifact_path: str = str(DEFAULT_META_ARTIFACT_PATH),
        output_path: str = str(DEFAULT_OUTPUT_PATH),
        max_bars: int = 800,
    ) -> dict[str, object]:
        previous_payload = json.loads(Path(previous_report_path).read_text(encoding="utf-8"))
        meta_payload = json.loads(Path(meta_artifact_path).read_text(encoding="utf-8"))
        exact_variants = self._extract_exact_variants(previous_payload)

        bars_by_symbol = self._deepseek_lab._load_bars(all_symbols=TOP_US_STOCK_SYMBOLS + ["SPY"], max_bars=max_bars)
        benchmark_bars = bars_by_symbol["SPY"]
        split_by_day = self._deepseek_lab._build_date_splits(benchmark_bars)
        assumptions = LlmNewsResearchLab._execution_assumptions(
            market="stocks",
            timeframe="day",
            regular_hours_only=False,
        )
        sentiment_engine = self._deepseek_lab._build_baseline_engine()

        exact_rows: list[dict[str, object]] = []
        for symbol in TOP_US_STOCK_SYMBOLS:
            bars = bars_by_symbol[symbol]
            articles = self._deepseek_lab._load_news(symbol=symbol, bars=bars)
            trading_days = [datetime.fromisoformat(bar.timestamp).date() for bar in bars]
            bundles = sentiment_engine.build_trading_day_bundles(
                symbol=symbol,
                articles=articles,
                trading_days=trading_days,
            )
            bundle_scores = sentiment_engine.score_bundles(bundles, batch_size=4)
            filled_scores = sentiment_engine.fill_missing_days(trading_days, bundle_scores, symbol)
            strict_test_days = {
                datetime.fromisoformat(bar.timestamp).date().isoformat()
                for bar in self._deepseek_lab._subset_bars(bars, split_by_day, "test")
            }
            test_bars = self._deepseek_lab._subset_bars(bars, split_by_day, "test")
            test_scores = [score for score in filled_scores if score.trading_day in strict_test_days]
            variant = exact_variants[symbol]
            definition = LlmOverlayDefinition(
                name=variant.strategy_name,
                lookback=variant.lookback,
                threshold=variant.threshold,
                entry_sentiment_min=variant.entry_sentiment_min,
                exit_sentiment_max=variant.exit_sentiment_max,
            )
            trace = build_llm_overlay_trace(
                bars=test_bars,
                sentiment_scores=test_scores,
                definition=definition,
                symbol=symbol,
                market="stocks",
                window_name="strict_test_exact_variant",
                risk_limits=self._risk_limits,
            )
            replay = run_execution_aware_replay(test_bars, trace, assumptions)
            exact_rows.append(
                {
                    "symbol": symbol,
                    "previous_window_average": {
                        "strategy_name": variant.strategy_name,
                        "profitable_windows": variant.profitable_windows,
                        "avg_total_return": round(variant.avg_total_return, 6),
                        "avg_max_drawdown": round(variant.avg_max_drawdown, 6),
                        "lookback": variant.lookback,
                        "threshold": variant.threshold,
                        "entry_sentiment_min": variant.entry_sentiment_min,
                        "exit_sentiment_max": variant.exit_sentiment_max,
                    },
                    "strict_exact_prompt_variant": replay["summary"],
                }
            )

        meta_by_symbol = {
            str(row["symbol"]): row["meta_strategy"]
            for row in meta_payload.get("comparison_rows", [])
        }
        comparison_rows = []
        for row in exact_rows:
            symbol = str(row["symbol"])
            comparison_rows.append(
                {
                    **row,
                    "strict_meta_model": meta_by_symbol[symbol],
                }
            )

        result = {
            "source_previous_report": str(Path(previous_report_path).resolve()),
            "source_meta_artifact": str(Path(meta_artifact_path).resolve()),
            "symbols": TOP_US_STOCK_SYMBOLS,
            "comparison_rows": comparison_rows,
            "aggregates": self._build_aggregates(comparison_rows),
        }
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _extract_exact_variants(self, previous_payload: dict[str, object]) -> dict[str, ExactVariant]:
        leaderboard = previous_payload["leaderboard"]
        latest_runs = previous_payload["latest_runs"]
        targets = {
            "AVGO": "avgo_momentum_llm_entry_filter",
            "NVDA": "nvda_momentum_llm_exit_filter",
            "TSLA": "tsla_momentum_llm_exit_filter",
            "GOOGL": "googl_momentum_llm_exit_filter",
            "XOM": "xom_momentum_llm_exit_filter",
        }
        extracted: dict[str, ExactVariant] = {}
        for symbol, strategy_name in targets.items():
            leaderboard_row = next(
                row for row in leaderboard
                if str(row["symbol"]) == symbol and str(row["strategy_name"]) == strategy_name
            )
            matching_runs = [
                row for row in latest_runs
                if str(row["symbol"]) == symbol and str(row["strategy_name"]) == strategy_name
            ]
            grouped: dict[tuple[float | None, float | None], list[dict[str, object]]] = {}
            for run in matching_runs:
                risk = run.get("risk", {})
                key = (
                    self._normalize_optional_float(risk.get("entry_sentiment_min")),
                    self._normalize_optional_float(risk.get("exit_sentiment_max")),
                )
                grouped.setdefault(key, []).append(run)
            target_avg = float(leaderboard_row["avg_total_return"])
            best_key = min(
                grouped,
                key=lambda key: abs(
                    (sum(float(run["total_return"]) for run in grouped[key]) / len(grouped[key])) - target_avg
                ),
            )
            sample = grouped[best_key][0]
            extracted[symbol] = ExactVariant(
                symbol=symbol,
                strategy_name=strategy_name,
                profitable_windows=int(leaderboard_row["profitable_windows"]),
                avg_total_return=float(leaderboard_row["avg_total_return"]),
                avg_max_drawdown=float(leaderboard_row["avg_max_drawdown"]),
                lookback=int(sample["lookback"]),
                threshold=float(sample["threshold"]),
                entry_sentiment_min=best_key[0],
                exit_sentiment_max=best_key[1],
            )
        return extracted

    @staticmethod
    def _normalize_optional_float(value: object) -> float | None:
        if value is None:
            return None
        return round(float(value), 6)

    @staticmethod
    def _build_aggregates(rows: list[dict[str, object]]) -> dict[str, object]:
        previous_returns = [float(row["previous_window_average"]["avg_total_return"]) for row in rows]
        previous_drawdowns = [float(row["previous_window_average"]["avg_max_drawdown"]) for row in rows]
        exact_returns = [float(row["strict_exact_prompt_variant"]["total_return"]) for row in rows]
        exact_drawdowns = [float(row["strict_exact_prompt_variant"]["max_drawdown"]) for row in rows]
        meta_returns = [float(row["strict_meta_model"]["total_return"]) for row in rows]
        meta_drawdowns = [float(row["strict_meta_model"]["max_drawdown"]) for row in rows]
        return {
            "previous_window_average": {
                "mean_total_return": round(sum(previous_returns) / len(previous_returns), 6),
                "mean_max_drawdown": round(sum(previous_drawdowns) / len(previous_drawdowns), 6),
            },
            "strict_exact_prompt_variant": {
                "mean_total_return": round(sum(exact_returns) / len(exact_returns), 6),
                "mean_max_drawdown": round(sum(exact_drawdowns) / len(exact_drawdowns), 6),
            },
            "strict_meta_model": {
                "mean_total_return": round(sum(meta_returns) / len(meta_returns), 6),
                "mean_max_drawdown": round(sum(meta_drawdowns) / len(meta_drawdowns), 6),
            },
            "strict_prompt_beats_meta": sum(
                1
                for row in rows
                if float(row["strict_exact_prompt_variant"]["total_return"]) > float(row["strict_meta_model"]["total_return"])
            ),
        }
