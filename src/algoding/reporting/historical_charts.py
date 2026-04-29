from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from algoding.data.fx import FrankfurterFxHistoricalClient
from algoding.data.historical import AlpacaHistoricalClient
from algoding.execution.historical_research import WINDOW_BAR_COUNTS
from algoding.portfolio.risk import RiskLimits
from algoding.research.historical_backtest import BacktestBar, build_backtest_trace
from algoding.research.parallel_strategies import StrategyDefinition, build_market_strategies
from algoding.settings import Settings


@dataclass(frozen=True)
class HistoricalChartResult:
    output_path: str
    market: str
    symbol: str
    strategy_name: str
    window_name: str
    summary: dict[str, object]


class HistoricalChartBuilder:
    def __init__(self, settings: Settings, risk_limits: RiskLimits | None = None) -> None:
        self._settings = settings
        self._stock_history = AlpacaHistoricalClient(settings)
        self._fx_history = FrankfurterFxHistoricalClient()
        self._risk_limits = risk_limits or RiskLimits.from_file()

    def build_chart(
        self,
        market: str,
        symbol: str,
        strategy_name: str,
        window_name: str,
        output_path: str | None = None,
        risk_limits: RiskLimits | None = None,
        timeframe: str = "day",
        regular_hours_only: bool = False,
        no_same_day_reentry: bool = False,
    ) -> HistoricalChartResult:
        normalized_market, normalized_symbol, normalized_window, bars, trace = self.build_trace(
            market=market,
            symbol=symbol,
            strategy_name=strategy_name,
            window_name=window_name,
            risk_limits=risk_limits,
            timeframe=timeframe,
            regular_hours_only=regular_hours_only,
            no_same_day_reentry=no_same_day_reentry,
        )

        destination = Path(output_path) if output_path else Path(self._default_output_path(
            normalized_symbol,
            strategy_name,
            normalized_window,
            timeframe,
            interactive=False,
        ))
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._render_chart(
            bars=bars,
            trace=trace,
            strategy_name=strategy_name,
            symbol=normalized_symbol,
            market=normalized_market,
            window_name=normalized_window,
            output_path=destination,
        )
        return HistoricalChartResult(
            output_path=str(destination),
            market=normalized_market,
            symbol=normalized_symbol,
            strategy_name=strategy_name,
            window_name=normalized_window,
            summary=trace["summary"],
        )

    def build_interactive_chart(
        self,
        market: str,
        symbol: str,
        strategy_name: str,
        window_name: str,
        output_path: str | None = None,
        risk_limits: RiskLimits | None = None,
        timeframe: str = "day",
        regular_hours_only: bool = False,
        no_same_day_reentry: bool = False,
    ) -> HistoricalChartResult:
        normalized_market, normalized_symbol, normalized_window, bars, trace = self.build_trace(
            market=market,
            symbol=symbol,
            strategy_name=strategy_name,
            window_name=window_name,
            risk_limits=risk_limits,
            timeframe=timeframe,
            regular_hours_only=regular_hours_only,
            no_same_day_reentry=no_same_day_reentry,
        )

        destination = Path(output_path) if output_path else Path(self._default_output_path(
            normalized_symbol,
            strategy_name,
            normalized_window,
            timeframe,
            interactive=True,
        ))
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._render_interactive_chart(
            bars=bars,
            trace=trace,
            strategy_name=strategy_name,
            symbol=normalized_symbol,
            market=normalized_market,
            window_name=normalized_window,
            output_path=destination,
        )
        return HistoricalChartResult(
            output_path=str(destination),
            market=normalized_market,
            symbol=normalized_symbol,
            strategy_name=strategy_name,
            window_name=normalized_window,
            summary=trace["summary"],
        )

    def build_trace(
        self,
        market: str,
        symbol: str,
        strategy_name: str,
        window_name: str,
        risk_limits: RiskLimits | None = None,
        timeframe: str = "day",
        regular_hours_only: bool = False,
        no_same_day_reentry: bool = False,
    ) -> tuple[str, str, str, list[BacktestBar], dict[str, object]]:
        normalized_market = market.strip().lower()
        normalized_symbol = symbol.strip().upper()
        normalized_window = window_name.strip().lower()
        if normalized_window not in WINDOW_BAR_COUNTS:
            raise ValueError(f"Unsupported window: {window_name}")

        definition = self._resolve_strategy(normalized_symbol, strategy_name)
        bars = self._load_bars(
            normalized_market,
            normalized_symbol,
            self._lookback_bars_for_timeframe(normalized_window, timeframe),
            timeframe=timeframe,
            regular_hours_only=regular_hours_only,
        )
        trace = build_backtest_trace(
            bars=bars,
            definition=definition,
            symbol=normalized_symbol,
            market=normalized_market,
            window_name=f"{normalized_window}_{timeframe}{'_rth' if regular_hours_only else ''}",
            risk_limits=risk_limits or self._risk_limits,
            no_same_day_reentry=no_same_day_reentry,
        )
        return normalized_market, normalized_symbol, normalized_window, bars, trace

    def _resolve_strategy(self, symbol: str, strategy_name: str) -> StrategyDefinition:
        for definition in build_market_strategies(symbol.lower()):
            if definition.name == strategy_name:
                return definition
        raise ValueError(f"Strategy not found for {symbol}: {strategy_name}")

    def _load_bars(
        self,
        market: str,
        symbol: str,
        lookback_bars: int,
        timeframe: str = "day",
        regular_hours_only: bool = False,
    ) -> list[BacktestBar]:
        if market == "fx":
            fx_bars = self._fx_history.get_recent_daily_bars([symbol], lookback_bars)
            symbol_bars = fx_bars.get(symbol, [])
            if not symbol_bars:
                raise ValueError(f"No historical bars returned for {symbol}.")
            return [
                BacktestBar(
                    timestamp=bar.timestamp,
                    open=bar.close,
                    close=bar.close,
                    high=bar.close,
                    low=bar.close,
                    volume=None,
                )
                for bar in symbol_bars
            ]

        if market == "crypto":
            crypto_bars = self._stock_history.get_recent_crypto_bars([symbol], lookback_bars, timeframe=timeframe)
            symbol_bars = crypto_bars.get(symbol, [])
            if not symbol_bars:
                raise ValueError(f"No historical bars returned for {symbol}.")
            return [
                BacktestBar(
                    timestamp=bar.timestamp.isoformat(),
                    open=float(bar.open),
                    close=float(bar.close),
                    high=float(bar.high),
                    low=float(bar.low),
                    volume=float(bar.volume) if getattr(bar, "volume", None) is not None else None,
                )
                for bar in symbol_bars
            ]

        stock_bars = self._stock_history.get_recent_bars([symbol], lookback_bars, timeframe=timeframe)
        symbol_bars = stock_bars.get(symbol, [])
        if regular_hours_only and timeframe.strip().lower() == "hour":
            symbol_bars = [
                bar
                for bar in symbol_bars
                if self._is_regular_session_hour(bar.timestamp)
            ]
        if not symbol_bars:
            raise ValueError(f"No historical bars returned for {symbol}.")
        return [
            BacktestBar(
                timestamp=bar.timestamp.isoformat(),
                open=float(bar.open),
                close=float(bar.close),
                high=float(bar.high),
                low=float(bar.low),
                volume=float(bar.volume) if getattr(bar, "volume", None) is not None else None,
            )
            for bar in symbol_bars
        ]

    @staticmethod
    def _is_regular_session_hour(timestamp: datetime) -> bool:
        market_time = timestamp.astimezone(ZoneInfo("America/New_York"))
        return 9 <= market_time.hour <= 15

    @staticmethod
    def _lookback_bars_for_timeframe(window_name: str, timeframe: str) -> int:
        base = WINDOW_BAR_COUNTS[window_name]
        if timeframe.strip().lower() == "hour":
            return base * 7
        return base

    @staticmethod
    def _default_output_path(
        symbol: str,
        strategy_name: str,
        window_name: str,
        timeframe: str,
        interactive: bool,
    ) -> str:
        normalized_timeframe = timeframe.strip().lower()
        timeframe_suffix = "" if normalized_timeframe == "day" else f"_{normalized_timeframe}"
        extension = "html" if interactive else "png"
        interactive_suffix = "_interactive" if interactive else ""
        return (
            f"reports/charts/{symbol.lower()}_{strategy_name}_{window_name}{timeframe_suffix}"
            f"{interactive_suffix}_dark.{extension}"
        )

    def _render_chart(
        self,
        bars: list[BacktestBar],
        trace: dict[str, object],
        strategy_name: str,
        symbol: str,
        market: str,
        window_name: str,
        output_path: Path,
    ) -> None:
        dates = [datetime.fromisoformat(bar.timestamp) for bar in bars]
        close_prices = [bar.close for bar in bars]
        trace_points = trace["trace_points"]
        summary = trace["summary"]

        indicator_dates = []
        indicator_values = []
        threshold_values = []
        long_dates = []
        long_values = []
        for point in trace_points:
            dt = datetime.fromisoformat(str(point["timestamp"]))
            indicator = point.get("indicator_value")
            threshold = point.get("threshold_value")
            if indicator is not None:
                indicator_dates.append(dt)
                indicator_values.append(float(indicator))
                threshold_values.append(float(threshold or 0.0))
            if bool(point.get("in_position")):
                long_dates.append(dt)
                long_values.append(float(point["close"]))

        buy_dates = [datetime.fromisoformat(str(event["timestamp"])) for event in trace["events"] if event["event"] == "buy"]
        buy_prices = [float(event["price"]) for event in trace["events"] if event["event"] == "buy"]
        sell_dates = [datetime.fromisoformat(str(event["timestamp"])) for event in trace["events"] if event["event"] == "sell"]
        sell_prices = [float(event["price"]) for event in trace["events"] if event["event"] == "sell"]

        plt.style.use("dark_background")
        fig, (ax_price, ax_indicator) = plt.subplots(
            2,
            1,
            figsize=(15, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1.4]},
        )
        fig.patch.set_facecolor("#0b1020")
        ax_price.set_facecolor("#111827")
        ax_indicator.set_facecolor("#111827")

        ax_price.plot(dates, close_prices, color="#8ab4f8", linewidth=1.9, label="Close")
        if long_dates:
            ax_price.fill_between(
                long_dates,
                long_values,
                [min(close_prices) * 0.985] * len(long_values),
                color="#22c55e",
                alpha=0.08,
                label="In Position",
            )
        if buy_dates:
            ax_price.scatter(buy_dates, buy_prices, color="#22c55e", marker="^", s=90, label="Buy")
        if sell_dates:
            ax_price.scatter(sell_dates, sell_prices, color="#ef4444", marker="v", s=90, label="Sell")

        ax_price.set_title(
            f"{symbol} {strategy_name} ({window_name}) | total return {float(summary['total_return']):.2%} | "
            f"max DD {float(summary['max_drawdown']):.2%} | trades {summary['trades']}",
            fontsize=14,
            color="#f8fafc",
            pad=14,
        )
        ax_price.set_ylabel("Price", color="#cbd5e1")
        ax_price.grid(color="#334155", alpha=0.35, linewidth=0.6)
        ax_price.legend(loc="upper left", frameon=False)

        ax_indicator.axhline(0.0, color="#64748b", linewidth=1.0, alpha=0.75)
        if indicator_dates:
            ax_indicator.plot(indicator_dates, indicator_values, color="#f59e0b", linewidth=1.6, label="Momentum Gap")
            ax_indicator.plot(
                indicator_dates,
                threshold_values,
                color="#22d3ee",
                linewidth=1.2,
                linestyle="--",
                label="Threshold",
            )
            ax_indicator.fill_between(
                indicator_dates,
                indicator_values,
                threshold_values,
                where=[value >= threshold for value, threshold in zip(indicator_values, threshold_values)],
                color="#22c55e",
                alpha=0.16,
            )
        ax_indicator.set_ylabel("Indicator", color="#cbd5e1")
        ax_indicator.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax_indicator.grid(color="#334155", alpha=0.35, linewidth=0.6)
        ax_indicator.legend(loc="upper left", frameon=False)
        ax_indicator.set_xlabel(f"{market} | dark research chart", color="#cbd5e1")

        for axis in (ax_price, ax_indicator):
            axis.tick_params(colors="#cbd5e1")
            for spine in axis.spines.values():
                spine.set_color("#334155")

        fig.tight_layout()
        fig.savefig(output_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    def _render_interactive_chart(
        self,
        bars: list[BacktestBar],
        trace: dict[str, object],
        strategy_name: str,
        symbol: str,
        market: str,
        window_name: str,
        output_path: Path,
    ) -> None:
        dates = [datetime.fromisoformat(bar.timestamp) for bar in bars]
        close_prices = [bar.close for bar in bars]
        trace_points = trace["trace_points"]
        summary = trace["summary"]

        indicator_dates = []
        indicator_values = []
        threshold_values = []
        position_fill = []
        effective_stops = []
        fixed_stops = []
        trailing_stops = []
        for point in trace_points:
            dt = datetime.fromisoformat(str(point["timestamp"]))
            indicator_dates.append(dt)
            indicator_values.append(point.get("indicator_value"))
            threshold_values.append(point.get("threshold_value"))
            position_fill.append(point["close"] if bool(point.get("in_position")) else None)
            effective_stops.append(point.get("effective_stop_price"))
            fixed_stops.append(point.get("fixed_stop_price"))
            trailing_stops.append(point.get("trailing_stop_price"))

        buy_events = [event for event in trace["events"] if event["event"] == "buy"]
        sell_events = [event for event in trace["events"] if event["event"] == "sell"]
        buy_dates = [datetime.fromisoformat(str(event["timestamp"])) for event in buy_events]
        buy_prices = [float(event["price"]) for event in buy_events]
        sell_dates = [datetime.fromisoformat(str(event["timestamp"])) for event in sell_events]
        sell_prices = [float(event["price"]) for event in sell_events]

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.72, 0.28],
            subplot_titles=(
                f"{symbol} {strategy_name} ({window_name})",
                "Signal Indicator",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=close_prices,
                mode="lines",
                name="Close",
                line={"color": "#8ab4f8", "width": 2},
                hovertemplate="%{x|%Y-%m-%d}<br>Close: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=position_fill,
                mode="lines",
                name="In Position",
                line={"color": "#22c55e", "width": 0},
                fill="tozeroy",
                fillcolor="rgba(34,197,94,0.10)",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=effective_stops,
                mode="lines",
                name="Effective Stop",
                line={"color": "#f97316", "width": 1.5, "dash": "dot"},
                hovertemplate="%{x|%Y-%m-%d}<br>Effective stop: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=fixed_stops,
                mode="lines",
                name="Fixed Stop",
                line={"color": "#ef4444", "width": 1, "dash": "dash"},
                visible="legendonly",
                hovertemplate="%{x|%Y-%m-%d}<br>Fixed stop: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=trailing_stops,
                mode="lines",
                name="Trailing Stop",
                line={"color": "#f59e0b", "width": 1, "dash": "dash"},
                visible="legendonly",
                hovertemplate="%{x|%Y-%m-%d}<br>Trailing stop: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        if buy_events:
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode="markers",
                    name="Buy",
                    marker={"color": "#22c55e", "size": 10, "symbol": "triangle-up"},
                    text=[event["reason"] for event in buy_events],
                    hovertemplate="%{x|%Y-%m-%d}<br>Buy: %{y:.2f}<br>%{text}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        if sell_events:
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode="markers",
                    name="Sell",
                    marker={"color": "#ef4444", "size": 10, "symbol": "triangle-down"},
                    text=[event["reason"] for event in sell_events],
                    hovertemplate="%{x|%Y-%m-%d}<br>Sell: %{y:.2f}<br>%{text}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=indicator_dates,
                y=indicator_values,
                mode="lines",
                name="Indicator",
                line={"color": "#f59e0b", "width": 1.8},
                hovertemplate="%{x|%Y-%m-%d}<br>Indicator: %{y:.2%}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=indicator_dates,
                y=threshold_values,
                mode="lines",
                name="Threshold",
                line={"color": "#22d3ee", "width": 1.4, "dash": "dash"},
                hovertemplate="%{x|%Y-%m-%d}<br>Threshold: %{y:.2%}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1020",
            plot_bgcolor="#111827",
            font={"color": "#e5e7eb"},
            title={
                "text": (
                    f"{symbol} {strategy_name} | total return {float(summary['total_return']):.2%} | "
                    f"max DD {float(summary['max_drawdown']):.2%} | trades {summary['trades']}"
                ),
                "x": 0.03,
            },
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.01},
            hovermode="x unified",
            margin={"l": 60, "r": 30, "t": 90, "b": 60},
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            rangeslider={"visible": True, "thickness": 0.07},
            row=2,
            col=1,
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)", row=1, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)", title_text="Price", row=1, col=1)
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            title_text="Indicator",
            tickformat=".0%",
            row=2,
            col=1,
        )
        fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
