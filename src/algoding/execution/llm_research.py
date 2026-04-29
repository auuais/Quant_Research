from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import floor
from zoneinfo import ZoneInfo

from algoding.data.historical import AlpacaHistoricalClient
from algoding.data.news import AlpacaNewsHistoricalClient
from algoding.data.yahoo import YahooHistoricalClient
from algoding.execution.storage import OrderStore
from algoding.portfolio.risk import RiskLimits
from algoding.research.cost_aware_backtest import ExecutionAssumptions, run_execution_aware_replay
from algoding.research.historical_backtest import BacktestBar
from algoding.research.llm_overlay_backtest import build_llm_overlay_strategies, build_llm_overlay_trace
from algoding.research.llm_sentiment import LocalLlmNewsSentimentEngine
from algoding.settings import Settings


DEFAULT_BROAD_STOCK_SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "AVGO",
    "AMD",
    "NFLX",
    "JPM",
    "XOM",
    "ORCL",
    "CRM",
    "WMT",
    "COST",
]

DEFAULT_COMMODITY_SYMBOLS = [
    "GLD",
    "SLV",
    "USO",
    "UNG",
]

DEFAULT_BROAD_FX_SYMBOLS = [
    "EURUSD",
    "USDJPY",
    "GBPUSD",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "EURJPY",
]

FX_YAHOO_TICKERS = {
    "EURUSD": "EURUSD=X",
    "USDJPY": "JPY=X",
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
    "USDCHF": "CHF=X",
    "EURJPY": "EURJPY=X",
}

FX_NEWS_PROXIES = {
    "EURUSD": ["FXE", "UUP"],
    "USDJPY": ["FXY", "UUP"],
    "GBPUSD": ["FXB", "UUP"],
    "AUDUSD": ["FXA", "UUP"],
    "USDCAD": ["FXC", "UUP"],
    "USDCHF": ["FXF", "UUP"],
    "EURJPY": ["FXE", "FXY"],
}

LLM_NEWS_SYMBOLS = {
    "commodities": DEFAULT_COMMODITY_SYMBOLS,
    "stocks": DEFAULT_BROAD_STOCK_SYMBOLS,
    "fx": DEFAULT_BROAD_FX_SYMBOLS,
}

WINDOW_BAR_COUNTS = {
    "1y": 252,
    "2y": 504,
    "3y": 756,
}


class LlmNewsResearchLab:
    def __init__(self, settings: Settings) -> None:
        if not settings.has_alpaca_credentials:
            raise RuntimeError("Alpaca credentials are required in .env for LLM news research.")
        self._settings = settings
        self._bars_client = AlpacaHistoricalClient(settings)
        self._yahoo_client = YahooHistoricalClient()
        self._news_client = AlpacaNewsHistoricalClient(settings)
        self._sentiment_engine = LocalLlmNewsSentimentEngine(settings)
        self._store = OrderStore(settings)
        self._risk_limits = RiskLimits.from_file()

    def run(
        self,
        *,
        windows: list[str],
        markets: list[str] | None = None,
        commodity_symbols: list[str] | None = None,
        stock_symbols: list[str] | None = None,
        fx_symbols: list[str] | None = None,
        stocks_only: bool = False,
        fx_only: bool = False,
        timeframe: str = "day",
        regular_hours_only: bool = False,
        stop_loss_pct: float | None = 0.05,
        trailing_stop_pct: float | None = 0.015,
    ) -> dict[str, object]:
        resolved_windows = self._resolve_windows(windows)
        normalized_timeframe = timeframe.strip().lower()
        max_request_bars = max(
            self._request_bar_count(window, normalized_timeframe) for window in resolved_windows
        )
        risk_limits = self._risk_limits.with_overrides(
            stop_loss_pct=stop_loss_pct,
            trailing_stop_pct=trailing_stop_pct,
        )
        self._store.ensure_schema()

        all_results: list[dict[str, object]] = []
        symbol_results: dict[str, list[dict[str, object]]] = {}
        coverage: dict[str, dict[str, object]] = {}

        market_symbols = self._resolve_symbol_sets(
            markets=markets,
            commodity_symbols=commodity_symbols,
            stock_symbols=stock_symbols,
            fx_symbols=fx_symbols,
            stocks_only=stocks_only,
            fx_only=fx_only,
        )
        for market, symbols in market_symbols.items():
            assumptions = self._execution_assumptions(
                market=market,
                timeframe=normalized_timeframe,
                regular_hours_only=regular_hours_only,
            )
            bars_by_symbol = self._load_bars(
                market,
                symbols,
                max_bars=max_request_bars,
                timeframe=normalized_timeframe,
                regular_hours_only=regular_hours_only,
            )
            for symbol, bars in bars_by_symbol.items():
                articles = self._load_news(market=market, symbol=symbol, bars=bars)
                trading_days = [datetime.fromisoformat(bar.timestamp).date() for bar in bars]
                unique_trading_days = list(dict.fromkeys(trading_days))
                bundles = self._sentiment_engine.build_trading_day_bundles(
                    symbol=symbol,
                    articles=articles,
                    trading_days=unique_trading_days,
                )
                scores_by_day = self._sentiment_engine.score_bundles(
                    bundles,
                    batch_size=self._sentiment_batch_size(market=market),
                )
                filled_daily_scores = self._sentiment_engine.fill_missing_days(
                    unique_trading_days,
                    scores_by_day,
                    symbol,
                )
                daily_score_map = {item.trading_day: item for item in filled_daily_scores}
                aligned_scores = [daily_score_map[day.isoformat()] for day in trading_days]
                entry_threshold, exit_threshold = self._calibrate_sentiment_thresholds(filled_daily_scores)
                coverage[symbol] = {
                    "articles": len(articles),
                    "news_days": len(bundles),
                    "bullish_days": sum(1 for item in filled_daily_scores if item.label == "bullish"),
                    "bearish_days": sum(1 for item in filled_daily_scores if item.label == "bearish"),
                    "neutral_days": sum(1 for item in filled_daily_scores if item.label == "neutral"),
                    "entry_threshold": round(entry_threshold, 6),
                    "exit_threshold": round(exit_threshold, 6),
                    "news_symbols": self._news_symbols_for_market(market=market, symbol=symbol),
                }

                symbol_runs: list[dict[str, object]] = []
                for window_name in resolved_windows:
                    window_size = self._analysis_bar_count(
                        window_name,
                        normalized_timeframe,
                        regular_hours_only,
                    )
                    window_bars = bars[-window_size:]
                    window_scores = aligned_scores[-window_size:]
                    for definition in build_llm_overlay_strategies(
                        symbol.lower(),
                        entry_sentiment_min=entry_threshold,
                        exit_sentiment_max=exit_threshold,
                    ):
                        trace = build_llm_overlay_trace(
                            bars=window_bars,
                            sentiment_scores=window_scores,
                            definition=definition,
                            symbol=symbol,
                            market=market,
                            window_name=(
                                f"{window_name}_{normalized_timeframe}"
                                f"{'_rth' if regular_hours_only and normalized_timeframe == 'hour' else ''}"
                            ),
                            risk_limits=risk_limits,
                        )
                        replay = run_execution_aware_replay(window_bars, trace, assumptions)
                        replay["summary"]["llm_model_name"] = self._settings.llm_news_model_name
                        replay["summary"]["news_days"] = coverage[symbol]["news_days"]
                        replay["summary"]["news_articles"] = coverage[symbol]["articles"]
                        self._store.record_strategy_run(replay["summary"])
                        all_results.append(replay["summary"])
                        symbol_runs.append(replay["summary"])
                symbol_results[symbol] = symbol_runs

        leaderboard = self._build_leaderboard(all_results)
        improvement = self._baseline_improvement(symbol_results)
        return {
            "markets": list(market_symbols.keys()),
            "symbols": [symbol for values in market_symbols.values() for symbol in values],
            "windows": resolved_windows,
            "llm_model_name": self._settings.llm_news_model_name,
            "timeframe": normalized_timeframe,
            "regular_hours_only": regular_hours_only,
            "execution_aware": True,
            "stop_loss_pct": risk_limits.stop_loss_pct,
            "trailing_stop_pct": risk_limits.trailing_stop_pct,
            "runs_recorded": len(all_results),
            "news_coverage": coverage,
            "leaderboard": leaderboard,
            "baseline_vs_overlay": improvement,
            "top_runs": sorted(all_results, key=lambda item: float(item["score"]), reverse=True)[:20],
        }

    def report(self, limit: int = 50) -> dict[str, object]:
        self._store.ensure_schema()
        runs = self._store.list_all_strategy_runs(run_type="historical_llm_news_execution_aware", limit=5000)
        leaderboard = self._build_leaderboard(runs)
        latest_runs = sorted(runs, key=lambda item: str(item.get("created_at", "")), reverse=True)[:limit]
        return {"leaderboard": leaderboard[:limit], "latest_runs": latest_runs}

    def _load_bars(
        self,
        market: str,
        symbols: list[str],
        *,
        max_bars: int,
        timeframe: str,
        regular_hours_only: bool,
    ) -> dict[str, list[BacktestBar]]:
        if market == "fx":
            if timeframe != "day":
                raise ValueError("FX LLM news research currently supports daily bars only.")
            yahoo_symbols = [FX_YAHOO_TICKERS[symbol] for symbol in symbols if symbol in FX_YAHOO_TICKERS]
            yahoo_bars = self._yahoo_client.get_recent_daily_bars(yahoo_symbols, max_bars)
            bars_by_symbol: dict[str, list[BacktestBar]] = {}
            for symbol in symbols:
                yahoo_symbol = FX_YAHOO_TICKERS.get(symbol)
                if yahoo_symbol is None:
                    continue
                symbol_bars = yahoo_bars.get(yahoo_symbol, [])
                if not symbol_bars:
                    continue
                bars_by_symbol[symbol] = [
                    BacktestBar(
                        timestamp=bar.timestamp.isoformat(),
                        open=float(bar.open),
                        close=float(bar.close),
                        high=float(bar.high),
                        low=float(bar.low),
                        volume=float(bar.volume) if bar.volume is not None else None,
                    )
                    for bar in symbol_bars
                ]
            return bars_by_symbol

        stock_bars = self._bars_client.get_recent_bars(symbols, max_bars, timeframe=timeframe)
        return {
            symbol: [
                BacktestBar(
                    timestamp=bar.timestamp.isoformat(),
                    open=float(bar.open),
                    close=float(bar.close),
                    high=float(bar.high),
                    low=float(bar.low),
                    volume=float(bar.volume) if getattr(bar, "volume", None) is not None else None,
                )
                for bar in (
                    [
                        value
                        for value in symbol_bars
                        if not (regular_hours_only and timeframe == "hour" and not self._is_regular_session_hour(value.timestamp))
                    ]
                )
            ]
            for symbol, symbol_bars in stock_bars.items()
            if symbol_bars
        }

    def _load_news(self, *, market: str, symbol: str, bars: list[BacktestBar]) -> list[object]:
        start = datetime.fromisoformat(bars[0].timestamp).astimezone(timezone.utc) - timedelta(days=3)
        end = datetime.fromisoformat(bars[-1].timestamp).astimezone(timezone.utc) + timedelta(days=1)
        proxy_symbols = self._news_symbols_for_market(market=market, symbol=symbol)
        articles_by_id: dict[int, object] = {}
        for proxy_symbol in proxy_symbols:
            for article in self._news_client.get_news_articles(
                symbol=proxy_symbol,
                start=start,
                end=end,
                include_content=False,
            ):
                articles_by_id[int(article.article_id)] = article
        return sorted(articles_by_id.values(), key=lambda item: str(item.created_at))

    @staticmethod
    def _resolve_symbol_sets(
        *,
        markets: list[str] | None,
        commodity_symbols: list[str] | None,
        stock_symbols: list[str] | None,
        fx_symbols: list[str] | None,
        stocks_only: bool,
        fx_only: bool,
    ) -> dict[str, list[str]]:
        symbols = dict(LLM_NEWS_SYMBOLS)
        if markets:
            normalized_markets = [value.strip().lower() for value in markets if value.strip()]
            invalid = [value for value in normalized_markets if value not in symbols]
            if invalid:
                raise ValueError(f"Unsupported LLM news markets: {', '.join(invalid)}")
            symbols = {market: symbols[market] for market in normalized_markets}
        if commodity_symbols:
            symbols["commodities"] = [value.strip().upper() for value in commodity_symbols if value.strip()]
        if stock_symbols:
            symbols["stocks"] = [value.strip().upper() for value in stock_symbols if value.strip()]
        if fx_symbols:
            symbols["fx"] = [value.strip().upper() for value in fx_symbols if value.strip()]
        if stocks_only:
            return {"stocks": symbols["stocks"]}
        if fx_only:
            return {"fx": symbols["fx"]}
        return symbols

    @staticmethod
    def _resolve_windows(windows: list[str]) -> list[str]:
        resolved = [window.strip().lower() for window in windows if window.strip()]
        invalid = [window for window in resolved if window not in WINDOW_BAR_COUNTS]
        if invalid:
            raise ValueError(f"Unsupported windows: {', '.join(invalid)}")
        return resolved or ["1y", "2y", "3y"]

    @staticmethod
    def _execution_assumptions(*, market: str, timeframe: str, regular_hours_only: bool) -> ExecutionAssumptions:
        if market == "fx":
            return ExecutionAssumptions(
                starting_capital=100_000.0,
                commission_per_order=0.0,
                quoted_spread_bps=1.5,
                market_impact_bps=0.5,
                stop_extra_slippage_bps=1.5,
                max_bar_participation_rate=0.005,
                max_bar_shares=None,
                sec_fee_per_million_sell=0.0,
                finra_taf_per_share_sell=0.0,
                finra_taf_cap_per_trade=0.0,
            )
        max_bar_shares = 4000.0 if timeframe == "hour" and regular_hours_only else None
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
    def _sentiment_batch_size(*, market: str) -> int:
        if market == "fx":
            return 2
        return 8

    @staticmethod
    def _news_symbols_for_market(*, market: str, symbol: str) -> list[str]:
        if market == "fx":
            return FX_NEWS_PROXIES.get(symbol, [])
        return [symbol]

    @staticmethod
    def _request_bar_count(window_name: str, timeframe: str) -> int:
        base = WINDOW_BAR_COUNTS[window_name]
        if timeframe == "hour":
            return base * 24
        return base

    @staticmethod
    def _analysis_bar_count(window_name: str, timeframe: str, regular_hours_only: bool) -> int:
        base = WINDOW_BAR_COUNTS[window_name]
        if timeframe == "hour":
            return base * (7 if regular_hours_only else 24)
        return base

    @staticmethod
    def _is_regular_session_hour(timestamp) -> bool:
        return 9 <= timestamp.astimezone(ZoneInfo("America/New_York")).hour <= 15

    @staticmethod
    def _calibrate_sentiment_thresholds(scores: list[object]) -> tuple[float, float]:
        values = sorted(float(item.score) for item in scores if abs(float(item.score)) > 1e-9)
        if not values:
            return 0.05, 0.02
        entry_index = min(len(values) - 1, floor(0.65 * (len(values) - 1)))
        exit_index = min(len(values) - 1, floor(0.35 * (len(values) - 1)))
        entry_threshold = max(0.03, values[entry_index])
        exit_threshold = min(entry_threshold - 1e-6, max(0.0, values[exit_index]))
        return entry_threshold, exit_threshold

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
                    "llm_model_name": str(items[0].get("llm_model_name", "")),
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

    @staticmethod
    def _baseline_improvement(symbol_runs: dict[str, list[dict[str, object]]]) -> dict[str, object]:
        totals = {"symbol_window_pairs": 0, "entry_filter_better": 0, "exit_filter_better": 0, "combo_better": 0}
        details: list[dict[str, object]] = []
        for symbol, runs in symbol_runs.items():
            by_window: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
            for run in runs:
                by_window[str(run["window_name"])][str(run["strategy_name"])] = run
            for window_name, strategies in by_window.items():
                baseline = strategies.get(f"{symbol.lower()}_momentum_baseline")
                if baseline is None:
                    continue
                totals["symbol_window_pairs"] += 1
                row = {
                    "symbol": symbol,
                    "window_name": window_name,
                    "baseline_return": baseline["total_return"],
                }
                for label, key in (
                    ("entry_filter", f"{symbol.lower()}_momentum_llm_entry_filter"),
                    ("exit_filter", f"{symbol.lower()}_momentum_llm_exit_filter"),
                    ("combo", f"{symbol.lower()}_momentum_llm_combo"),
                ):
                    candidate = strategies.get(key)
                    if candidate is None:
                        continue
                    better = float(candidate["total_return"]) > float(baseline["total_return"])
                    row[f"{label}_return"] = candidate["total_return"]
                    row[f"{label}_better"] = better
                    if better:
                        totals[f"{label}_better"] += 1
                details.append(row)
        return {"summary": totals, "details": details}
