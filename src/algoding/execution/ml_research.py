from __future__ import annotations

from collections import defaultdict
from zoneinfo import ZoneInfo

from algoding.data.fx import FrankfurterFxHistoricalClient
from algoding.data.historical import AlpacaHistoricalClient
from algoding.execution.historical_research import MARKET_GROUPS, WINDOW_BAR_COUNTS
from algoding.execution.storage import OrderStore
from algoding.portfolio.risk import RiskLimits
from algoding.research.cost_aware_backtest import ExecutionAssumptions, run_execution_aware_replay
from algoding.research.historical_backtest import BacktestBar
from algoding.research.ml_historical import build_ml_dataset, default_ml_models, run_walk_forward_ml_trace
from algoding.settings import Settings


class MlResearchLab:
    def __init__(self, settings: Settings) -> None:
        if not settings.has_alpaca_credentials:
            raise RuntimeError("Alpaca credentials are required in .env for ML historical research.")
        self._settings = settings
        self._stock_history = AlpacaHistoricalClient(settings)
        self._fx_history = FrankfurterFxHistoricalClient()
        self._store = OrderStore(settings)
        self._risk_limits = RiskLimits.from_file()

    def run(
        self,
        markets: list[str],
        windows: list[str],
        models: list[str] | None = None,
        *,
        timeframe: str = "day",
        regular_hours_only: bool = False,
        no_same_day_reentry: bool = False,
        stop_loss_pct: float | None = None,
        trailing_stop_pct: float | None = None,
    ) -> dict[str, object]:
        resolved_windows = self._resolve_windows(windows)
        resolved_markets = self._resolve_markets(markets)
        resolved_models = self._resolve_models(models)
        max_bars = max(self._lookback_bars(window, timeframe) for window in resolved_windows)
        assumptions = self._execution_assumptions(timeframe=timeframe, regular_hours_only=regular_hours_only)
        risk_limits = self._risk_limits.with_overrides(
            stop_loss_pct=stop_loss_pct,
            trailing_stop_pct=trailing_stop_pct,
        )
        self._store.ensure_schema()

        all_results: list[dict[str, object]] = []
        for market in resolved_markets:
            series = self._load_market_series(
                market=market,
                max_bars=max_bars,
                timeframe=timeframe,
                regular_hours_only=regular_hours_only,
            )
            for symbol, bars in series.items():
                for window_name in resolved_windows:
                    window_bars = bars[-self._lookback_bars(window_name, timeframe) :]
                    dataset = build_ml_dataset(window_bars)
                    for model in resolved_models:
                        trace = run_walk_forward_ml_trace(
                            dataset=dataset,
                            model_definition=model,
                            symbol=symbol,
                            market=market,
                            window_name=f"{window_name}_{timeframe}{'_rth' if regular_hours_only else ''}",
                            risk_limits=risk_limits,
                            no_same_day_reentry=no_same_day_reentry,
                        )
                        replay = run_execution_aware_replay(dataset.bars, trace, assumptions)
                        self._store.record_strategy_run(replay["summary"])
                        all_results.append(replay["summary"])

        leaderboard = self._build_leaderboard(all_results)
        return {
            "markets": resolved_markets,
            "windows": resolved_windows,
            "models": [model.name for model in resolved_models],
            "timeframe": timeframe,
            "regular_hours_only": regular_hours_only,
            "no_same_day_reentry": no_same_day_reentry,
            "stop_loss_pct": risk_limits.stop_loss_pct,
            "trailing_stop_pct": risk_limits.trailing_stop_pct,
            "runs_recorded": len(all_results),
            "leaderboard": leaderboard[:50],
            "top_runs": sorted(all_results, key=lambda item: float(item["score"]), reverse=True)[:20],
        }

    def report(self, limit: int = 50) -> dict[str, object]:
        self._store.ensure_schema()
        runs = self._store.list_all_strategy_runs(run_type="historical_ml_execution_aware", limit=5000)
        leaderboard = self._build_leaderboard(runs)
        latest_runs = sorted(runs, key=lambda item: str(item["created_at"]), reverse=True)[:limit]
        return {"leaderboard": leaderboard[:limit], "latest_runs": latest_runs}

    def _load_market_series(
        self,
        *,
        market: str,
        max_bars: int,
        timeframe: str,
        regular_hours_only: bool,
    ) -> dict[str, list[BacktestBar]]:
        symbols = MARKET_GROUPS[market]
        if market == "fx":
            if timeframe.strip().lower() != "day":
                raise ValueError("FX ML research currently supports daily bars only.")
            fx_bars = self._fx_history.get_recent_daily_bars(symbols, max_bars)
            return {
                symbol: [
                    BacktestBar(timestamp=bar.timestamp, close=bar.close, high=bar.close, low=bar.close)
                    for bar in symbol_bars
                ]
                for symbol, symbol_bars in fx_bars.items()
                if symbol_bars
            }

        stock_bars = self._stock_history.get_recent_bars(symbols, max_bars, timeframe=timeframe)
        series: dict[str, list[BacktestBar]] = {}
        for symbol, symbol_bars in stock_bars.items():
            filtered_bars = symbol_bars
            if regular_hours_only and timeframe.strip().lower() == "hour":
                filtered_bars = [bar for bar in symbol_bars if self._is_regular_session_hour(bar.timestamp)]
            if not filtered_bars:
                continue
            series[symbol] = [
                BacktestBar(
                    timestamp=bar.timestamp.isoformat(),
                    open=float(bar.open),
                    close=float(bar.close),
                    high=float(bar.high),
                    low=float(bar.low),
                    volume=float(bar.volume) if getattr(bar, "volume", None) is not None else None,
                )
                for bar in filtered_bars
            ]
        return series

    @staticmethod
    def _resolve_windows(windows: list[str]) -> list[str]:
        resolved = [window.strip().lower() for window in windows if window.strip()]
        invalid = [window for window in resolved if window not in WINDOW_BAR_COUNTS]
        if invalid:
            raise ValueError(f"Unsupported windows: {', '.join(invalid)}")
        return resolved or ["1y", "2y", "3y"]

    @staticmethod
    def _resolve_markets(markets: list[str]) -> list[str]:
        resolved = [market.strip().lower() for market in markets if market.strip()]
        invalid = [market for market in resolved if market not in MARKET_GROUPS]
        if invalid:
            raise ValueError(f"Unsupported markets: {', '.join(invalid)}")
        return resolved or ["commodities", "etfs", "stocks"]

    @staticmethod
    def _resolve_models(models: list[str] | None) -> list[object]:
        available = {model.name: model for model in default_ml_models()}
        if not models:
            return list(available.values())
        resolved: list[object] = []
        invalid: list[str] = []
        for model_name in models:
            normalized = model_name.strip().lower()
            model = available.get(normalized)
            if model is None:
                invalid.append(model_name)
                continue
            resolved.append(model)
        if invalid:
            raise ValueError(f"Unsupported ML models: {', '.join(invalid)}")
        return resolved

    @staticmethod
    def _lookback_bars(window_name: str, timeframe: str) -> int:
        base = WINDOW_BAR_COUNTS[window_name]
        if timeframe.strip().lower() == "hour":
            return base * 7
        return base

    @staticmethod
    def _is_regular_session_hour(timestamp) -> bool:
        return 9 <= timestamp.astimezone(ZoneInfo("America/New_York")).hour <= 15

    @staticmethod
    def _execution_assumptions(*, timeframe: str, regular_hours_only: bool) -> ExecutionAssumptions:
        max_bar_shares = 4000.0 if timeframe.strip().lower() == "hour" and regular_hours_only else None
        return ExecutionAssumptions(
            starting_capital=100_000.0,
            commission_per_order=0.0,
            quoted_spread_bps=1.0,
            market_impact_bps=1.0,
            stop_extra_slippage_bps=2.0,
            max_bar_participation_rate=0.0005,
            max_bar_shares=max_bar_shares,
            sec_fee_per_million_sell=0.0,
            finra_taf_per_share_sell=0.000195,
            finra_taf_cap_per_trade=9.79,
        )

    @staticmethod
    def _build_leaderboard(runs: list[dict[str, object]]) -> list[dict[str, object]]:
        latest_by_window: dict[tuple[str, str, str, str], dict[str, object]] = {}
        for run in runs:
            key = (
                str(run.get("market")),
                str(run.get("symbol")),
                str(run.get("strategy_name")),
                str(run.get("window_name")),
            )
            current = latest_by_window.get(key)
            if current is None or str(run.get("created_at", "")) > str(current.get("created_at", "")):
                latest_by_window[key] = run

        grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
        for run in latest_by_window.values():
            key = (str(run.get("market")), str(run.get("symbol")), str(run.get("strategy_name")))
            grouped[key].append(run)

        leaderboard: list[dict[str, object]] = []
        for (market, symbol, strategy_name), items in grouped.items():
            avg_total_return = sum(float(item["total_return"]) for item in items) / len(items)
            avg_annualized_return = sum(float(item.get("annualized_return", 0.0)) for item in items) / len(items)
            avg_max_drawdown = sum(float(item["max_drawdown"]) for item in items) / len(items)
            avg_win_rate = sum(float(item["win_rate"]) for item in items) / len(items)
            avg_profit_factor = sum(float(item.get("profit_factor", 0.0)) for item in items) / len(items)
            avg_score = sum(float(item["score"]) for item in items) / len(items)
            profitable_windows = sum(1 for item in items if float(item["total_return"]) > 0)
            profitable_ratio = profitable_windows / len(items)
            composite_score = avg_score + (profitable_ratio * 0.1)
            leaderboard.append(
                {
                    "market": market,
                    "symbol": symbol,
                    "strategy_name": strategy_name,
                    "strategy_family": str(items[0].get("strategy_family", "")),
                    "model_name": str(items[0].get("model_name", "")),
                    "windows_tested": sorted({str(item.get("window_name")) for item in items}),
                    "profitable_windows": profitable_windows,
                    "avg_total_return": round(avg_total_return, 6),
                    "avg_annualized_return": round(avg_annualized_return, 6),
                    "avg_max_drawdown": round(avg_max_drawdown, 6),
                    "avg_win_rate": round(avg_win_rate, 6),
                    "avg_profit_factor": round(avg_profit_factor, 6),
                    "avg_score": round(avg_score, 6),
                    "composite_score": round(composite_score, 6),
                    "best_window_return": round(max(float(item["total_return"]) for item in items), 6),
                    "worst_window_return": round(min(float(item["total_return"]) for item in items), 6),
                    "latest_signal": str(items[-1].get("latest_signal", "flat")),
                }
            )

        leaderboard.sort(key=lambda item: float(item["composite_score"]), reverse=True)
        return leaderboard
