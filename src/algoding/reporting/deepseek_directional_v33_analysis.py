from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr, linregress

from algoding.data.historical import AlpacaHistoricalClient
from algoding.data.news import AlpacaNewsHistoricalClient
from algoding.execution.deepseek_directional_research import DeepseekDirectionalResearchLab
from algoding.execution.llm_research import LlmNewsResearchLab
from algoding.portfolio.risk import RiskLimits
from algoding.research.cost_aware_backtest import run_execution_aware_replay
from algoding.research.deepseek_directional_model import (
    DirectionalPrediction,
    DirectionalTradeDefinition,
    build_directional_trade_trace,
)
from algoding.research.historical_backtest import BacktestBar
from algoding.research.news_labeler import build_bundle_metadata
from algoding.settings import Settings


@dataclass(frozen=True)
class SentimentPoint:
    trading_day: str
    score: float
    label: str


def run_v33_analysis(
    *,
    settings: Settings,
    symbols: list[str],
    output_root: str = "reports/research/deepseek_directional_v3/v33_sentiment_analysis",
    max_bars: int = 1400,
) -> dict[str, object]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_client = AlpacaHistoricalClient(settings)
    news_client = AlpacaNewsHistoricalClient(settings)
    split_helper = DeepseekDirectionalResearchLab(settings)
    risk_limits = RiskLimits.from_file().with_overrides(stop_loss_pct=0.05, trailing_stop_pct=0.01)
    assumptions = LlmNewsResearchLab._execution_assumptions(market="stocks", timeframe="day", regular_hours_only=False)

    raw_sentiment_cache = _load_sentiment_cache(
        Path("cache/llm_news_deepseek_v4/daily_bundle_scores.jsonl"),
        symbols=symbols,
    )
    directional_cache = _load_directional_cache(
        Path("reports/research/deepseek_directional_v3/finetuned_directional_scores.jsonl"),
        symbols=symbols,
    )

    all_symbols = sorted(set(symbols + ["SPY"]))
    bars_by_symbol_raw = history_client.get_recent_bars(all_symbols, max_bars, timeframe="day")
    bars_by_symbol = {
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
        for symbol, values in bars_by_symbol_raw.items()
    }
    split_by_day = split_helper._build_date_splits(bars_by_symbol["SPY"])

    rows = []
    for symbol in symbols:
        bars = bars_by_symbol[symbol]
        test_bars = [
            bar
            for bar in bars
            if split_by_day.get(datetime.fromisoformat(bar.timestamp).date().isoformat()) == "test"
        ]
        trading_days = [datetime.fromisoformat(bar.timestamp).date() for bar in bars]
        articles = news_client.get_news_articles(
            symbol=symbol,
            start=datetime.fromisoformat(bars[0].timestamp) - timedelta(days=3),
            end=datetime.fromisoformat(bars[-1].timestamp) + timedelta(days=1),
            include_content=False,
        )
        bundles = build_bundle_metadata(symbol=symbol, articles=articles, trading_days=trading_days)
        sentiment_series = _fill_missing_sentiment_days(
            trading_days=trading_days,
            raw_scores=raw_sentiment_cache.get(symbol, {}),
            symbol=symbol,
        )
        test_sentiment_series = {
            day: point
            for day, point in sentiment_series.items()
            if split_by_day.get(day) == "test"
        }
        predictions = {
            day: directional_cache[symbol][day]
            for day in (datetime.fromisoformat(bar.timestamp).date().isoformat() for bar in test_bars)
            if day in directional_cache.get(symbol, {})
        }
        trace = build_directional_trade_trace(
            bars=test_bars,
            predictions_by_day=predictions,
            definition=DirectionalTradeDefinition(
                name=f"{symbol.lower()}_v3_3_trailing_1pct",
                minimum_strength="medium",
                require_relative_confirmation=False,
                require_momentum_confirmation=False,
            ),
            risk_limits=risk_limits,
        )
        replay = run_execution_aware_replay(test_bars, trace, assumptions)
        chart_path = output_dir / f"{symbol.lower()}_v33_trailing_1pct_sentiment_chart.html"
        _render_chart(
            symbol=symbol,
            bars=test_bars,
            trace=trace,
            sentiment_series=test_sentiment_series,
            output_path=chart_path,
        )
        stats = _compute_stats(
            bars=test_bars,
            sentiment_series=test_sentiment_series,
        )
        rows.append(
            {
                "symbol": symbol,
                "test_period": f"{datetime.fromisoformat(test_bars[0].timestamp).date().isoformat()} -> {datetime.fromisoformat(test_bars[-1].timestamp).date().isoformat()} ({len(test_bars)} days)",
                "trade_summary": replay["summary"],
                "chart_path": str(chart_path),
                "stats": stats,
            }
        )

    aggregate = _aggregate(rows)
    result = {
        "version": "V3-3",
        "policy": {
            "minimum_strength": "medium",
            "require_relative_confirmation": False,
            "require_momentum_confirmation": False,
            "stop_loss_pct": 0.05,
            "trailing_stop_pct": 0.01,
        },
        "rows": rows,
        "aggregate": aggregate,
    }
    result_path = output_dir / "v33_sentiment_analysis.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary_path = output_dir / "v33_sentiment_analysis_summary.md"
    summary_path.write_text(_build_summary_markdown(result), encoding="utf-8")
    return result


def _load_sentiment_cache(path: Path, *, symbols: list[str]) -> dict[str, dict[str, SentimentPoint]]:
    allowed = set(symbols)
    results: dict[str, dict[str, SentimentPoint]] = {symbol: {} for symbol in symbols}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        symbol = str(payload.get("symbol", "")).upper()
        if symbol not in allowed:
            continue
        results[symbol][str(payload["trading_day"])] = SentimentPoint(
            trading_day=str(payload["trading_day"]),
            score=float(payload["score"]),
            label=str(payload["label"]),
        )
    return results


def _load_directional_cache(path: Path, *, symbols: list[str]) -> dict[str, dict[str, DirectionalPrediction]]:
    allowed = set(symbols)
    results: dict[str, dict[str, DirectionalPrediction]] = {symbol: {} for symbol in symbols}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        symbol = str(payload.get("symbol", "")).upper()
        if symbol not in allowed:
            continue
        results[symbol][str(payload["trading_day"])] = DirectionalPrediction(
            symbol=symbol,
            trading_day=str(payload["trading_day"]),
            direction=str(payload["direction"]),
            strength=str(payload["strength"]),
            relative_to_spy=str(payload["relative_to_spy"]),
            event_type=str(payload.get("event_type", "other")),
            session=str(payload.get("session", "mixed")),
            raw_completion=str(payload.get("raw_completion", "")),
        )
    return results


def _fill_missing_sentiment_days(
    *,
    trading_days: list[date],
    raw_scores: dict[str, SentimentPoint],
    symbol: str,
    carry_days: int = 3,
    decay: float = 0.6,
) -> dict[str, SentimentPoint]:
    resolved: dict[str, SentimentPoint] = {}
    recent_scores: list[float] = []
    for trading_day in trading_days:
        key = trading_day.isoformat()
        sentiment = raw_scores.get(key)
        if sentiment is None:
            carried = 0.0
            if recent_scores:
                carried = sum(value * (decay ** idx) for idx, value in enumerate(reversed(recent_scores[-carry_days:])))
            label = "neutral" if abs(carried) < 0.25 else ("bullish" if carried > 0 else "bearish")
            sentiment = SentimentPoint(trading_day=key, score=max(-1.0, min(1.0, carried)), label=label)
        if sentiment.score != 0.0:
            recent_scores.append(float(sentiment.score))
        resolved[key] = sentiment
    return resolved


def _render_chart(
    *,
    symbol: str,
    bars: list[BacktestBar],
    trace: dict[str, object],
    sentiment_series: dict[str, SentimentPoint],
    output_path: Path,
) -> None:
    dates = [datetime.fromisoformat(bar.timestamp) for bar in bars]
    closes = [bar.close for bar in bars]
    sentiment_scores = [float(sentiment_series[dt.date().isoformat()].score) for dt in dates]
    sentiment_roll = _rolling_mean(sentiment_scores, 5)

    buy_events = [event for event in trace["events"] if event["event"] == "buy"]
    sell_events = [event for event in trace["events"] if event["event"] == "sell"]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.58, 0.22, 0.20],
        subplot_titles=(
            f"{symbol} price and V3-3 trades",
            "Raw daily sentiment score",
            "Normalized price vs sentiment (5d mean)",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=closes,
            mode="lines",
            name="Close",
            line={"color": "#8ab4f8", "width": 2},
        ),
        row=1,
        col=1,
    )
    if buy_events:
        fig.add_trace(
            go.Scatter(
                x=[datetime.fromisoformat(str(event["timestamp"])) for event in buy_events],
                y=[float(event["price"]) for event in buy_events],
                mode="markers",
                name="Buy",
                marker={"color": "#22c55e", "size": 9, "symbol": "triangle-up"},
                text=[event.get("reason", "buy") for event in buy_events],
                hovertemplate="%{x|%Y-%m-%d}<br>Buy %{y:.2f}<br>%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if sell_events:
        fig.add_trace(
            go.Scatter(
                x=[datetime.fromisoformat(str(event["timestamp"])) for event in sell_events],
                y=[float(event["price"]) for event in sell_events],
                mode="markers",
                name="Sell",
                marker={"color": "#ef4444", "size": 9, "symbol": "triangle-down"},
                text=[event.get("reason", "sell") for event in sell_events],
                hovertemplate="%{x|%Y-%m-%d}<br>Sell %{y:.2f}<br>%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    marker_colors = [
        "#22c55e" if value > 0.1 else "#ef4444" if value < -0.1 else "#94a3b8"
        for value in sentiment_scores
    ]
    fig.add_trace(
        go.Bar(
            x=dates,
            y=sentiment_scores,
            name="Sentiment",
            marker={"color": marker_colors},
            opacity=0.75,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=sentiment_roll,
            mode="lines",
            name="5d sentiment mean",
            line={"color": "#f59e0b", "width": 2},
        ),
        row=2,
        col=1,
    )

    norm_price = _normalize_series(closes)
    norm_sent = _normalize_series(sentiment_roll)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=norm_price,
            mode="lines",
            name="Normalized price",
            line={"color": "#8ab4f8", "width": 2},
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=norm_sent,
            mode="lines",
            name="Normalized sentiment",
            line={"color": "#f59e0b", "width": 2},
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1020",
        plot_bgcolor="#111827",
        font={"color": "#e5e7eb"},
        title={"text": f"{symbol} V3-3 trades with raw DeepSeek sentiment", "x": 0.03},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.01},
        hovermode="x unified",
        margin={"l": 60, "r": 30, "t": 90, "b": 60},
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", row=2, col=1)
    fig.update_yaxes(title_text="Normalized", row=3, col=1)
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        rangeslider={"visible": True, "thickness": 0.06},
        row=3,
        col=1,
    )
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)


def _compute_stats(*, bars: list[BacktestBar], sentiment_series: dict[str, SentimentPoint]) -> dict[str, object]:
    dates = [datetime.fromisoformat(bar.timestamp).date().isoformat() for bar in bars]
    closes = np.array([bar.close for bar in bars], dtype=float)
    sentiment = np.array([float(sentiment_series[day].score) for day in dates], dtype=float)
    price_norm = np.array(_normalize_series(closes.tolist()), dtype=float)
    sentiment_roll = np.array(_rolling_mean(sentiment.tolist(), 5), dtype=float)

    returns_1d = np.zeros_like(closes)
    returns_1d[1:] = closes[1:] / closes[:-1] - 1.0
    forward_1d = np.zeros_like(closes)
    forward_1d[:-1] = closes[1:] / closes[:-1] - 1.0
    forward_3d = _forward_return(closes, 3)
    forward_5d = _forward_return(closes, 5)

    level_corr = _corr_stats(sentiment_roll, price_norm)
    same_day_return_corr = _corr_stats(sentiment, returns_1d)
    next_day_corr = _corr_stats(sentiment[:-1], forward_1d[:-1])
    next_3d_corr = _corr_stats(sentiment[:-3], forward_3d[:-3])
    next_5d_corr = _corr_stats(sentiment[:-5], forward_5d[:-5])
    lag_table = []
    for lag in range(0, 6):
        if lag == 0:
            x = sentiment
            y = forward_1d
        else:
            x = sentiment[:-lag]
            y = forward_1d[lag:]
        lag_table.append({"lag_days": lag, **_corr_stats(x, y)})

    bullish_mask = sentiment > 0.1
    bearish_mask = sentiment < -0.1
    neutral_mask = (~bullish_mask) & (~bearish_mask)

    return {
        "sentiment_vs_price_level": level_corr,
        "sentiment_vs_same_day_return": same_day_return_corr,
        "sentiment_vs_next_day_return": next_day_corr,
        "sentiment_vs_next_3d_return": next_3d_corr,
        "sentiment_vs_next_5d_return": next_5d_corr,
        "lagged_next_day_return_correlations": lag_table,
        "conditional_forward_returns": {
            "bullish_count": int(bullish_mask.sum()),
            "neutral_count": int(neutral_mask.sum()),
            "bearish_count": int(bearish_mask.sum()),
            "bullish_mean_1d": float(np.nanmean(forward_1d[bullish_mask])) if bullish_mask.any() else 0.0,
            "neutral_mean_1d": float(np.nanmean(forward_1d[neutral_mask])) if neutral_mask.any() else 0.0,
            "bearish_mean_1d": float(np.nanmean(forward_1d[bearish_mask])) if bearish_mask.any() else 0.0,
            "bullish_mean_5d": float(np.nanmean(forward_5d[bullish_mask])) if bullish_mask.any() else 0.0,
            "neutral_mean_5d": float(np.nanmean(forward_5d[neutral_mask])) if neutral_mask.any() else 0.0,
            "bearish_mean_5d": float(np.nanmean(forward_5d[bearish_mask])) if bearish_mask.any() else 0.0,
        },
    }


def _corr_stats(x, y) -> dict[str, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 3 or np.allclose(x_arr.std(), 0.0) or np.allclose(y_arr.std(), 0.0):
        return {"pearson": 0.0, "pearson_p": 1.0, "spearman": 0.0, "spearman_p": 1.0, "beta": 0.0, "r2": 0.0}
    pear = pearsonr(x_arr, y_arr)
    spear = spearmanr(x_arr, y_arr)
    reg = linregress(x_arr, y_arr)
    return {
        "pearson": float(pear.statistic),
        "pearson_p": float(pear.pvalue),
        "spearman": float(spear.statistic),
        "spearman_p": float(spear.pvalue),
        "beta": float(reg.slope),
        "r2": float(reg.rvalue ** 2),
    }


def _forward_return(closes: np.ndarray, horizon: int) -> np.ndarray:
    result = np.full_like(closes, np.nan, dtype=float)
    if horizon <= 0:
        return result
    result[:-horizon] = closes[horizon:] / closes[:-horizon] - 1.0
    return result


def _rolling_mean(values: list[float], window: int) -> list[float]:
    out = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        subset = values[start : index + 1]
        out.append(sum(subset) / len(subset))
    return out


def _normalize_series(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return []
    first = arr[0] if arr[0] != 0 else 1.0
    return (arr / first).tolist()


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    def mean_for(path: str) -> float:
        values = []
        for row in rows:
            current = row
            for key in path.split("."):
                current = current[key]
            values.append(float(current))
        return float(np.mean(values))

    return {
        "mean_trade_return": mean_for("trade_summary.total_return"),
        "mean_trade_drawdown": mean_for("trade_summary.max_drawdown"),
        "mean_level_pearson": mean_for("stats.sentiment_vs_price_level.pearson"),
        "mean_next_day_pearson": mean_for("stats.sentiment_vs_next_day_return.pearson"),
        "mean_next_3d_pearson": mean_for("stats.sentiment_vs_next_3d_return.pearson"),
        "mean_next_5d_pearson": mean_for("stats.sentiment_vs_next_5d_return.pearson"),
    }


def _build_summary_markdown(result: dict[str, object]) -> str:
    lines = [
        "# V3-3 Sentiment Analysis",
        "",
        "## Aggregate",
        "",
        f"- mean trade return: `{result['aggregate']['mean_trade_return']:.4f}`",
        f"- mean trade drawdown: `{result['aggregate']['mean_trade_drawdown']:.4f}`",
        f"- mean sentiment vs price-level Pearson: `{result['aggregate']['mean_level_pearson']:.4f}`",
        f"- mean sentiment vs next-day Pearson: `{result['aggregate']['mean_next_day_pearson']:.4f}`",
        f"- mean sentiment vs next-3d Pearson: `{result['aggregate']['mean_next_3d_pearson']:.4f}`",
        f"- mean sentiment vs next-5d Pearson: `{result['aggregate']['mean_next_5d_pearson']:.4f}`",
        "",
        "## Per Symbol",
        "",
    ]
    for row in result["rows"]:
        stats = row["stats"]
        lines.extend(
            [
                f"### {row['symbol']}",
                "",
                f"- chart: `{row['chart_path']}`",
                f"- trade return: `{row['trade_summary']['total_return']:.4f}`",
                f"- max drawdown: `{row['trade_summary']['max_drawdown']:.4f}`",
                f"- sentiment vs price-level Pearson: `{stats['sentiment_vs_price_level']['pearson']:.4f}`",
                f"- sentiment vs next-day Pearson: `{stats['sentiment_vs_next_day_return']['pearson']:.4f}`",
                f"- sentiment vs next-3d Pearson: `{stats['sentiment_vs_next_3d_return']['pearson']:.4f}`",
                f"- sentiment vs next-5d Pearson: `{stats['sentiment_vs_next_5d_return']['pearson']:.4f}`",
                (
                    f"- bullish/neutral/bearish mean 5d returns: "
                    f"`{stats['conditional_forward_returns']['bullish_mean_5d']:.4f}` / "
                    f"`{stats['conditional_forward_returns']['neutral_mean_5d']:.4f}` / "
                    f"`{stats['conditional_forward_returns']['bearish_mean_5d']:.4f}`"
                ),
                "",
            ]
        )
    return "\n".join(lines)
