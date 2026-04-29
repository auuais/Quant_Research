from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import datetime

from algoding.execution.llm_research import LlmNewsResearchLab
from algoding.research.cost_aware_backtest import run_execution_aware_replay
from algoding.research.llm_overlay_backtest import LlmOverlayDefinition, build_llm_overlay_strategies, build_llm_overlay_trace
from algoding.settings import Settings


TOP_US_DAILY_CONTENDERS: dict[str, str] = {
    "AVGO": "entry_filter",
    "NVDA": "exit_filter",
    "TSLA": "exit_filter",
    "GOOGL": "exit_filter",
    "XOM": "exit_filter",
}


class LlmRiskExperimentLab:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lab = LlmNewsResearchLab(settings)

    def run_top5_us_daily(
        self,
        *,
        windows: list[str] | None = None,
        stop_loss_pct: float = 0.05,
        trailing_stop_pct: float = 0.015,
    ) -> dict[str, object]:
        resolved_windows = windows or ["1y", "2y", "3y"]
        risk_limits = self._lab._risk_limits.with_overrides(
            stop_loss_pct=stop_loss_pct,
            trailing_stop_pct=trailing_stop_pct,
        )
        assumptions = self._lab._execution_assumptions(
            market="stocks",
            timeframe="day",
            regular_hours_only=False,
        )
        max_request_bars = max(self._lab._request_bar_count(window, "day") for window in resolved_windows)
        bars_by_symbol = self._lab._load_bars(
            "stocks",
            list(TOP_US_DAILY_CONTENDERS.keys()),
            max_bars=max_request_bars,
            timeframe="day",
            regular_hours_only=False,
        )

        all_results: list[dict[str, object]] = []
        per_symbol_runs: dict[str, list[dict[str, object]]] = defaultdict(list)
        coverage: dict[str, dict[str, object]] = {}

        for symbol, overlay_mode in TOP_US_DAILY_CONTENDERS.items():
            bars = bars_by_symbol[symbol]
            articles = self._lab._load_news(market="stocks", symbol=symbol, bars=bars)
            trading_days = [datetime.fromisoformat(bar.timestamp).date() for bar in bars]
            unique_trading_days = list(dict.fromkeys(trading_days))
            bundles = self._lab._sentiment_engine.build_trading_day_bundles(
                symbol=symbol,
                articles=articles,
                trading_days=unique_trading_days,
            )
            scores_by_day = self._lab._sentiment_engine.score_bundles(
                bundles,
                batch_size=self._lab._sentiment_batch_size(market="stocks"),
            )
            filled_daily_scores = self._lab._sentiment_engine.fill_missing_days(
                unique_trading_days,
                scores_by_day,
                symbol,
            )
            daily_score_map = {item.trading_day: item for item in filled_daily_scores}
            aligned_scores = [daily_score_map[day.isoformat()] for day in trading_days]
            entry_threshold, exit_threshold = self._lab._calibrate_sentiment_thresholds(filled_daily_scores)
            coverage[symbol] = {
                "articles": len(articles),
                "news_days": len(bundles),
                "entry_threshold": round(entry_threshold, 6),
                "exit_threshold": round(exit_threshold, 6),
            }

            definitions = build_llm_overlay_strategies(
                symbol.lower(),
                entry_sentiment_min=entry_threshold,
                exit_sentiment_max=exit_threshold,
            )
            base_name = f"{symbol.lower()}_momentum_llm_{overlay_mode}"
            base_definition = next(
                definition for definition in definitions if definition.name == base_name
            )
            variants = self._build_risk_variants(base_definition, trailing_stop_pct=trailing_stop_pct)

            for window_name in resolved_windows:
                window_size = self._lab._analysis_bar_count(window_name, "day", False)
                window_bars = bars[-window_size:]
                window_scores = aligned_scores[-window_size:]
                for variant_label, definition in variants:
                    trace = build_llm_overlay_trace(
                        bars=window_bars,
                        sentiment_scores=window_scores,
                        definition=definition,
                        symbol=symbol,
                        market="stocks",
                        window_name=f"{window_name}_day",
                        risk_limits=risk_limits,
                    )
                    replay = run_execution_aware_replay(window_bars, trace, assumptions)
                    replay["summary"]["run_type"] = "historical_llm_risk_experiment_execution_aware"
                    replay["summary"]["llm_model_name"] = self._settings.llm_news_model_name
                    replay["summary"]["base_strategy_name"] = base_name
                    replay["summary"]["experiment_variant"] = variant_label
                    replay["summary"]["news_days"] = coverage[symbol]["news_days"]
                    replay["summary"]["news_articles"] = coverage[symbol]["articles"]
                    all_results.append(replay["summary"])
                    per_symbol_runs[symbol].append(replay["summary"])

        leaderboard = self._build_leaderboard(all_results)
        variant_summary = self._build_variant_summary(all_results)
        comparison = self._build_comparison(per_symbol_runs)
        return {
            "symbols": list(TOP_US_DAILY_CONTENDERS.keys()),
            "base_modes": dict(TOP_US_DAILY_CONTENDERS),
            "windows": resolved_windows,
            "timeframe": "day",
            "llm_model_name": self._settings.llm_news_model_name,
            "execution_aware": True,
            "stop_loss_pct": stop_loss_pct,
            "trailing_stop_pct": trailing_stop_pct,
            "runs_recorded": len(all_results),
            "news_coverage": coverage,
            "leaderboard": leaderboard,
            "variant_summary": variant_summary,
            "comparison": comparison,
            "top_runs": sorted(all_results, key=lambda item: float(item["score"]), reverse=True)[:20],
        }

    def run_top5_us_daily_rule_grid(self) -> dict[str, object]:
        resolved_windows = ["1y", "2y", "3y"]
        symbols = list(TOP_US_DAILY_CONTENDERS.keys())
        max_request_bars = max(self._lab._request_bar_count(window, "day") for window in resolved_windows)
        bars_by_symbol = self._lab._load_bars(
            "stocks",
            symbols,
            max_bars=max_request_bars,
            timeframe="day",
            regular_hours_only=False,
        )
        all_results: list[dict[str, object]] = []
        coverage: dict[str, dict[str, object]] = {}
        per_symbol_runs: dict[str, list[dict[str, object]]] = defaultdict(list)

        parameter_grid = self._build_rule_grid()
        for symbol, overlay_mode in TOP_US_DAILY_CONTENDERS.items():
            bars = bars_by_symbol[symbol]
            articles = self._lab._load_news(market="stocks", symbol=symbol, bars=bars)
            trading_days = [datetime.fromisoformat(bar.timestamp).date() for bar in bars]
            unique_trading_days = list(dict.fromkeys(trading_days))
            bundles = self._lab._sentiment_engine.build_trading_day_bundles(
                symbol=symbol,
                articles=articles,
                trading_days=unique_trading_days,
            )
            scores_by_day = self._lab._sentiment_engine.score_bundles(
                bundles,
                batch_size=self._lab._sentiment_batch_size(market="stocks"),
            )
            base_filled_scores = self._lab._sentiment_engine.fill_missing_days(
                unique_trading_days,
                scores_by_day,
                symbol,
                carry_days=3,
            )
            entry_threshold, exit_threshold = self._lab._calibrate_sentiment_thresholds(base_filled_scores)
            strict_entry_threshold = min(1.0, max(entry_threshold, entry_threshold * 1.25))
            coverage[symbol] = {
                "articles": len(articles),
                "news_days": len(bundles),
                "entry_threshold": round(entry_threshold, 6),
                "strict_entry_threshold": round(strict_entry_threshold, 6),
                "exit_threshold": round(exit_threshold, 6),
            }

            definitions = build_llm_overlay_strategies(
                symbol.lower(),
                entry_sentiment_min=entry_threshold,
                exit_sentiment_max=exit_threshold,
            )
            base_name = f"{symbol.lower()}_momentum_llm_{overlay_mode}"
            base_definition = next(
                definition for definition in definitions if definition.name == base_name
            )

            for freshness_days in (1, 2, 3):
                filled_scores = self._lab._sentiment_engine.fill_missing_days(
                    unique_trading_days,
                    scores_by_day,
                    symbol,
                    carry_days=freshness_days,
                )
                daily_score_map = {item.trading_day: item for item in filled_scores}
                aligned_scores = [daily_score_map[day.isoformat()] for day in trading_days]
                for config in parameter_grid:
                    entry_min = (
                        strict_entry_threshold
                        if config["asymmetric_threshold_enabled"]
                        else entry_threshold
                    )
                    definition = replace(
                        base_definition,
                        name=f"{base_definition.name}__{config['label']}",
                        entry_sentiment_min=entry_min,
                        exit_sentiment_max=exit_threshold,
                        break_even_trigger_pct=0.03 if config["break_even_enabled"] else None,
                        time_stop_bars=config["time_stop_bars"],
                        reentry_cooldown_bars=config["reentry_cooldown_bars"],
                        sentiment_freshness_days=freshness_days,
                        no_trade_weak_neutral_cluster=config["weak_cluster_block"],
                    )
                    risk_limits = self._lab._risk_limits.with_overrides(
                        stop_loss_pct=config["stop_loss_pct"],
                        trailing_stop_pct=config["trailing_stop_pct"],
                    )
                    assumptions = self._lab._execution_assumptions(
                        market="stocks",
                        timeframe="day",
                        regular_hours_only=False,
                    )
                    for window_name in resolved_windows:
                        window_size = self._lab._analysis_bar_count(window_name, "day", False)
                        window_bars = bars[-window_size:]
                        window_scores = aligned_scores[-window_size:]
                        trace = build_llm_overlay_trace(
                            bars=window_bars,
                            sentiment_scores=window_scores,
                            definition=definition,
                            symbol=symbol,
                            market="stocks",
                            window_name=f"{window_name}_day",
                            risk_limits=risk_limits,
                        )
                        replay = run_execution_aware_replay(window_bars, trace, assumptions)
                        replay["summary"]["run_type"] = "historical_llm_rule_grid_execution_aware"
                        replay["summary"]["llm_model_name"] = self._settings.llm_news_model_name
                        replay["summary"]["base_strategy_name"] = base_name
                        replay["summary"]["experiment_variant"] = config["label"]
                        replay["summary"]["stop_loss_pct"] = config["stop_loss_pct"]
                        replay["summary"]["trailing_stop_pct"] = config["trailing_stop_pct"]
                        replay["summary"]["break_even_trigger_pct"] = 0.03 if config["break_even_enabled"] else None
                        replay["summary"]["break_even_enabled"] = config["break_even_enabled"]
                        replay["summary"]["time_stop_bars"] = config["time_stop_bars"]
                        replay["summary"]["reentry_cooldown_bars"] = config["reentry_cooldown_bars"]
                        replay["summary"]["sentiment_freshness_days"] = freshness_days
                        replay["summary"]["asymmetric_threshold_enabled"] = config["asymmetric_threshold_enabled"]
                        replay["summary"]["weak_cluster_block"] = config["weak_cluster_block"]
                        replay["summary"]["news_days"] = coverage[symbol]["news_days"]
                        replay["summary"]["news_articles"] = coverage[symbol]["articles"]
                        all_results.append(replay["summary"])
                        per_symbol_runs[symbol].append(replay["summary"])

        leaderboard = self._build_rule_grid_leaderboard(all_results)
        parameter_scores = self._build_parameter_scores(all_results)
        best_by_symbol = self._build_best_rule_grid_by_symbol(per_symbol_runs)
        return {
            "symbols": symbols,
            "base_modes": dict(TOP_US_DAILY_CONTENDERS),
            "windows": resolved_windows,
            "timeframe": "day",
            "llm_model_name": self._settings.llm_news_model_name,
            "execution_aware": True,
            "runs_recorded": len(all_results),
            "news_coverage": coverage,
            "leaderboard": leaderboard,
            "parameter_scores": parameter_scores,
            "best_by_symbol": best_by_symbol,
            "top_runs": sorted(all_results, key=lambda item: float(item["score"]), reverse=True)[:25],
        }

    @staticmethod
    def _build_risk_variants(
        base_definition: LlmOverlayDefinition,
        *,
        trailing_stop_pct: float,
    ) -> list[tuple[str, LlmOverlayDefinition]]:
        variants: list[tuple[str, LlmOverlayDefinition]] = [
            (
                "current_best",
                replace(base_definition, name=f"{base_definition.name}__current_best"),
            ),
            (
                "profit_lock_only",
                replace(
                    base_definition,
                    name=f"{base_definition.name}__profit_lock_only",
                    profit_lock_trigger_pct=0.05,
                    profit_lock_trailing_stop_pct=0.01,
                ),
            ),
        ]
        for fraction in (0.25, 0.5):
            for trigger in (0.05, 0.08):
                fraction_label = int(fraction * 100)
                trigger_label = int(trigger * 100)
                variants.append(
                    (
                        f"profit_lock_partial_{fraction_label}_at_{trigger_label}",
                        replace(
                            base_definition,
                            name=f"{base_definition.name}__profit_lock_partial_{fraction_label}_at_{trigger_label}",
                            profit_lock_trigger_pct=0.05,
                            profit_lock_trailing_stop_pct=0.01,
                            partial_take_profit_fraction=fraction,
                            partial_take_profit_trigger_pct=trigger,
                        ),
                    )
                )
        return variants

    @staticmethod
    def _build_leaderboard(runs: list[dict[str, object]]) -> list[dict[str, object]]:
        grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for run in runs:
            key = (str(run["symbol"]), str(run["experiment_variant"]))
            grouped[key].append(run)

        leaderboard: list[dict[str, object]] = []
        for (symbol, variant_label), items in grouped.items():
            avg_total_return = sum(float(item["total_return"]) for item in items) / len(items)
            avg_annualized_return = sum(float(item.get("annualized_return", 0.0)) for item in items) / len(items)
            avg_max_drawdown = sum(float(item["max_drawdown"]) for item in items) / len(items)
            avg_win_rate = sum(float(item["win_rate"]) for item in items) / len(items)
            avg_profit_factor = sum(float(item.get("profit_factor", 0.0)) for item in items) / len(items)
            avg_score = sum(float(item["score"]) for item in items) / len(items)
            profitable_windows = sum(1 for item in items if float(item["total_return"]) > 0)
            profitable_ratio = profitable_windows / len(items)
            leaderboard.append(
                {
                    "symbol": symbol,
                    "base_strategy_name": str(items[0]["base_strategy_name"]),
                    "experiment_variant": variant_label,
                    "windows_tested": sorted({str(item["window_name"]) for item in items}),
                    "profitable_windows": profitable_windows,
                    "avg_total_return": round(avg_total_return, 6),
                    "avg_annualized_return": round(avg_annualized_return, 6),
                    "avg_max_drawdown": round(avg_max_drawdown, 6),
                    "avg_win_rate": round(avg_win_rate, 6),
                    "avg_profit_factor": round(avg_profit_factor, 6),
                    "avg_score": round(avg_score, 6),
                    "composite_score": round(avg_score + (profitable_ratio * 0.1), 6),
                    "best_window_return": round(max(float(item["total_return"]) for item in items), 6),
                    "worst_window_return": round(min(float(item["total_return"]) for item in items), 6),
                }
            )
        leaderboard.sort(key=lambda item: float(item["composite_score"]), reverse=True)
        return leaderboard

    @staticmethod
    def _build_variant_summary(runs: list[dict[str, object]]) -> list[dict[str, object]]:
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for run in runs:
            grouped[str(run["experiment_variant"])].append(run)
        summary: list[dict[str, object]] = []
        for variant_label, items in grouped.items():
            avg_total_return = sum(float(item["total_return"]) for item in items) / len(items)
            avg_max_drawdown = sum(float(item["max_drawdown"]) for item in items) / len(items)
            avg_score = sum(float(item["score"]) for item in items) / len(items)
            profitable_windows = sum(1 for item in items if float(item["total_return"]) > 0)
            summary.append(
                {
                    "experiment_variant": variant_label,
                    "avg_total_return": round(avg_total_return, 6),
                    "avg_max_drawdown": round(avg_max_drawdown, 6),
                    "avg_score": round(avg_score, 6),
                    "profitable_windows": profitable_windows,
                    "windows_tested": len(items),
                }
            )
        summary.sort(key=lambda item: float(item["avg_score"]), reverse=True)
        return summary

    @staticmethod
    def _build_comparison(per_symbol_runs: dict[str, list[dict[str, object]]]) -> dict[str, object]:
        by_symbol: dict[str, list[dict[str, object]]] = {}
        for symbol, items in per_symbol_runs.items():
            grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
            for item in items:
                grouped[str(item["experiment_variant"])].append(item)
            rows: list[dict[str, object]] = []
            current = grouped["current_best"]
            current_avg_return = sum(float(item["total_return"]) for item in current) / len(current)
            current_avg_drawdown = sum(float(item["max_drawdown"]) for item in current) / len(current)
            for variant_label, variant_items in grouped.items():
                avg_total_return = sum(float(item["total_return"]) for item in variant_items) / len(variant_items)
                avg_max_drawdown = sum(float(item["max_drawdown"]) for item in variant_items) / len(variant_items)
                rows.append(
                    {
                        "experiment_variant": variant_label,
                        "avg_total_return": round(avg_total_return, 6),
                        "avg_max_drawdown": round(avg_max_drawdown, 6),
                        "delta_vs_current_return": round(avg_total_return - current_avg_return, 6),
                        "delta_vs_current_drawdown": round(avg_max_drawdown - current_avg_drawdown, 6),
                    }
                )
            rows.sort(key=lambda item: float(item["avg_total_return"]), reverse=True)
            by_symbol[symbol] = rows
        return {"by_symbol": by_symbol}

    @staticmethod
    def _build_rule_grid() -> list[dict[str, object]]:
        grid: list[dict[str, object]] = []
        for stop_loss_pct in (0.04, 0.05, 0.06):
            for trailing_stop_pct in (0.01, 0.015, 0.02):
                for time_stop_bars in (10, 20, 30):
                    for reentry_cooldown_bars in (1, 2):
                        for break_even_enabled in (False, True):
                            for asymmetric_threshold_enabled in (False, True):
                                for weak_cluster_block in (False, True):
                                    label = (
                                        f"sl_{int(stop_loss_pct * 100)}"
                                        f"_trail_{str(trailing_stop_pct * 100).replace('.0', '')}"
                                        f"_be_{'3' if break_even_enabled else 'off'}"
                                        f"_time_{time_stop_bars}"
                                        f"_cool_{reentry_cooldown_bars}"
                                        f"_asym_{'on' if asymmetric_threshold_enabled else 'off'}"
                                        f"_weak_{'on' if weak_cluster_block else 'off'}"
                                    )
                                    grid.append(
                                        {
                                            "label": label,
                                            "stop_loss_pct": stop_loss_pct,
                                            "trailing_stop_pct": trailing_stop_pct,
                                            "time_stop_bars": time_stop_bars,
                                            "reentry_cooldown_bars": reentry_cooldown_bars,
                                            "break_even_enabled": break_even_enabled,
                                            "asymmetric_threshold_enabled": asymmetric_threshold_enabled,
                                            "weak_cluster_block": weak_cluster_block,
                                        }
                                    )
        return grid

    @staticmethod
    def _build_rule_grid_leaderboard(runs: list[dict[str, object]]) -> list[dict[str, object]]:
        grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
        for run in runs:
            key = (
                str(run["symbol"]),
                str(run["experiment_variant"]),
                int(run["sentiment_freshness_days"]),
            )
            grouped[key].append(run)

        leaderboard: list[dict[str, object]] = []
        for (symbol, variant_label, freshness_days), items in grouped.items():
            avg_total_return = sum(float(item["total_return"]) for item in items) / len(items)
            avg_annualized_return = sum(float(item.get("annualized_return", 0.0)) for item in items) / len(items)
            avg_max_drawdown = sum(float(item["max_drawdown"]) for item in items) / len(items)
            avg_win_rate = sum(float(item["win_rate"]) for item in items) / len(items)
            avg_profit_factor = sum(float(item.get("profit_factor", 0.0)) for item in items) / len(items)
            avg_score = sum(float(item["score"]) for item in items) / len(items)
            profitable_windows = sum(1 for item in items if float(item["total_return"]) > 0)
            profitable_ratio = profitable_windows / len(items)
            leaderboard.append(
                {
                    "symbol": symbol,
                    "base_strategy_name": str(items[0]["base_strategy_name"]),
                    "experiment_variant": variant_label,
                    "sentiment_freshness_days": freshness_days,
                    "stop_loss_pct": items[0]["stop_loss_pct"],
                    "trailing_stop_pct": items[0]["trailing_stop_pct"],
                    "time_stop_bars": items[0]["time_stop_bars"],
                    "reentry_cooldown_bars": items[0]["reentry_cooldown_bars"],
                    "break_even_enabled": items[0]["break_even_enabled"],
                    "asymmetric_threshold_enabled": items[0]["asymmetric_threshold_enabled"],
                    "weak_cluster_block": items[0]["weak_cluster_block"],
                    "windows_tested": sorted({str(item["window_name"]) for item in items}),
                    "profitable_windows": profitable_windows,
                    "avg_total_return": round(avg_total_return, 6),
                    "avg_annualized_return": round(avg_annualized_return, 6),
                    "avg_max_drawdown": round(avg_max_drawdown, 6),
                    "avg_win_rate": round(avg_win_rate, 6),
                    "avg_profit_factor": round(avg_profit_factor, 6),
                    "avg_score": round(avg_score, 6),
                    "composite_score": round(avg_score + (profitable_ratio * 0.1), 6),
                }
            )
        leaderboard.sort(key=lambda item: float(item["composite_score"]), reverse=True)
        return leaderboard

    @staticmethod
    def _build_parameter_scores(runs: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
        output: dict[str, list[dict[str, object]]] = {}
        for key in (
            "stop_loss_pct",
            "trailing_stop_pct",
            "time_stop_bars",
            "reentry_cooldown_bars",
            "sentiment_freshness_days",
            "break_even_enabled",
            "asymmetric_threshold_enabled",
            "weak_cluster_block",
        ):
            grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
            for run in runs:
                grouped[str(run[key])].append(run)
            rows: list[dict[str, object]] = []
            for value, items in grouped.items():
                avg_total_return = sum(float(item["total_return"]) for item in items) / len(items)
                avg_max_drawdown = sum(float(item["max_drawdown"]) for item in items) / len(items)
                avg_score = sum(float(item["score"]) for item in items) / len(items)
                rows.append(
                    {
                        "value": value,
                        "avg_total_return": round(avg_total_return, 6),
                        "avg_max_drawdown": round(avg_max_drawdown, 6),
                        "avg_score": round(avg_score, 6),
                        "runs": len(items),
                    }
                )
            rows.sort(key=lambda item: float(item["avg_score"]), reverse=True)
            output[key] = rows
        return output

    @staticmethod
    def _build_best_rule_grid_by_symbol(per_symbol_runs: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for symbol, items in per_symbol_runs.items():
            grouped: dict[tuple[str, int, bool, bool, bool], list[dict[str, object]]] = defaultdict(list)
            for item in items:
                key = (
                    str(item["experiment_variant"]),
                    int(item["sentiment_freshness_days"]),
                    bool(item["break_even_enabled"]),
                    bool(item["asymmetric_threshold_enabled"]),
                    bool(item["weak_cluster_block"]),
                )
                grouped[key].append(item)
            best_row: dict[str, object] | None = None
            for (variant_label, freshness_days, break_even_enabled, asymmetric_threshold_enabled, weak_cluster_block), variant_items in grouped.items():
                avg_total_return = sum(float(item["total_return"]) for item in variant_items) / len(variant_items)
                avg_max_drawdown = sum(float(item["max_drawdown"]) for item in variant_items) / len(variant_items)
                avg_score = sum(float(item["score"]) for item in variant_items) / len(variant_items)
                candidate = {
                    "symbol": symbol,
                    "base_strategy_name": str(variant_items[0]["base_strategy_name"]),
                    "experiment_variant": variant_label,
                    "sentiment_freshness_days": freshness_days,
                    "stop_loss_pct": variant_items[0]["stop_loss_pct"],
                    "trailing_stop_pct": variant_items[0]["trailing_stop_pct"],
                    "time_stop_bars": variant_items[0]["time_stop_bars"],
                    "reentry_cooldown_bars": variant_items[0]["reentry_cooldown_bars"],
                    "break_even_enabled": break_even_enabled,
                    "asymmetric_threshold_enabled": asymmetric_threshold_enabled,
                    "weak_cluster_block": weak_cluster_block,
                    "avg_total_return": round(avg_total_return, 6),
                    "avg_max_drawdown": round(avg_max_drawdown, 6),
                    "avg_score": round(avg_score, 6),
                }
                if best_row is None or float(candidate["avg_score"]) > float(best_row["avg_score"]):
                    best_row = candidate
            if best_row is not None:
                rows.append(best_row)
        rows.sort(key=lambda item: float(item["avg_score"]), reverse=True)
        return rows
