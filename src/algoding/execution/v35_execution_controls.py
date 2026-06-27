"""Price-aware controls for the V3-5 directional trade mapper.

Everything here runs off the cached DeepSeek V3-5 directional scores (no GPU) and the same
execution-aware cost model as the rest of the repo, on a single shared test window. It exists to
answer one question: how much of V3-5's headline return is *entry edge* versus a trailing-stop /
zero-cost / idealized-fill artifact?

Controls produced (all gross AND net of costs, same window, same symbols):
  - buy_and_hold .............. honest single-name benchmark
  - always_long_1pct ......... no signal, just the 1% trailing-stop mechanic
  - random_entry_1pct ........ random entries at V3-5's own entry rate (mean over seeds)
  - v35_model @ trailing grid . model entries at 1/2/3/5%/none + model-exit-only (no stop)
  - entry_expectancy ......... forward-5d return of bullish entry days vs unconditional
  - no_same_day_reentry check . demonstrates the flag is a no-op on daily bars
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from algoding.data.historical import AlpacaHistoricalClient
from algoding.execution.deepseek_directional_research import DeepseekDirectionalResearchLab
from algoding.execution.llm_research import LlmNewsResearchLab
from algoding.execution.qwen_embedding_directional_research import (
    TOP5_SYMBOLS,
    _apply_embargo,
    _bar_day,
    _cache_max_day,
    _load_sentiment_cache,
)
from algoding.research.trade_execution import relabel_stop_events
from algoding.portfolio.risk import RiskLimits
from algoding.research.cost_aware_backtest import run_execution_aware_replay
from algoding.research.deepseek_directional_model import (
    DirectionalPrediction,
    DirectionalTradeDefinition,
    build_directional_trade_trace,
)
from algoding.research.historical_backtest import BacktestBar
from algoding.settings import Settings


NO_TRAIL = 10.0  # trailing/stop pct so large the band is never hit (disables that stop)
TRAILING_GRID = [0.01, 0.02, 0.03, 0.05, NO_TRAIL]
LABEL_HORIZON_BARS = 5


def run_v35_execution_controls(
    settings: Settings,
    *,
    output_root: str = "reports/research/qwen_embedding_v3_q3e/controls",
    v35_scores_path: str = "reports/research/deepseek_directional_v3/finetuned_directional_scores.jsonl",
    symbols: list = None,
    max_bars: int = 1400,
    end_date: str = None,
    seeds: int = 25,
    stop_loss_pct: float = 0.05,
    participation_rate: float = 0.0005,
    volume_scale: float = 30.0,
    sec_fee_per_million_sell: float = 27.8,
) -> dict:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_symbols = symbols or TOP5_SYMBOLS

    sentiment_cache = _load_sentiment_cache(Path(v35_scores_path))
    if end_date is None:
        end_date = _cache_max_day(sentiment_cache)

    client = AlpacaHistoricalClient(settings)
    bars_by_symbol = _load_bars(client, sorted({*eval_symbols, "SPY"}), max_bars)
    if end_date is not None:
        bars_by_symbol = {
            symbol: [bar for bar in bars if _bar_day(bar) <= end_date]
            for symbol, bars in bars_by_symbol.items()
        }
    split_by_day = _apply_embargo(
        DeepseekDirectionalResearchLab._build_date_splits(bars_by_symbol["SPY"]),
        embargo_days=LABEL_HORIZON_BARS,
    )
    # The default IEX feed reports only ~1/30th of consolidated volume, which makes the
    # 0.05%-of-volume participation cap bind on tiny $100k orders. Scale volume back up to an
    # approximate consolidated figure so the (retained) cap reflects real liquidity, and pass
    # through the SEC Section 31 sell fee for completeness.
    assumptions = replace(
        LlmNewsResearchLab._execution_assumptions(market="stocks", timeframe="day", regular_hours_only=False),
        max_bar_participation_rate=participation_rate,
        sec_fee_per_million_sell=sec_fee_per_million_sell,
    )
    # Levers #2 + #3: a market-on-close profile -- low spread/impact, no stop-gap slippage.
    moc_assumptions = replace(assumptions, quoted_spread_bps=0.5, market_impact_bps=0.5, stop_extra_slippage_bps=0.0)

    moc_keys = ["v35_model_trail_1pct", "v35_model_trail_2pct", "v35_model_trail_3pct",
                "v35_model_trail_5pct", "v35_model_trail_none", "v35_model_exit_only"]
    per_symbol = {}
    entry_rate_by_symbol = {}
    reentry_check = {}
    median_dollar_volume = {}
    for symbol in eval_symbols:
        test_bars = _scale_volume(_test_bars(bars_by_symbol[symbol], split_by_day), volume_scale)
        median_dollar_volume[symbol] = _median([bar.volume * bar.close for bar in test_bars]) if test_bars else 0.0
        cache = sentiment_cache.get(symbol, {})
        v35_predictions = {
            _bar_day(bar): cache[_bar_day(bar)] for bar in test_bars if _bar_day(bar) in cache
        }
        entry_rate = _entry_rate(test_bars, v35_predictions)
        entry_rate_by_symbol[symbol] = round(entry_rate, 6)

        results = {}
        results["buy_and_hold"] = _buy_and_hold(test_bars, assumptions)
        results["always_long_1pct"] = _trade_result(
            test_bars, _constant_predictions(test_bars, symbol), assumptions,
            trailing_pct=0.01, stop_pct=stop_loss_pct,
        )
        results["random_entry_1pct"] = _random_entry_result(
            test_bars, symbol, entry_rate, assumptions, trailing_pct=0.01, stop_pct=stop_loss_pct, seeds=seeds,
        )
        for trailing in TRAILING_GRID:
            label = "none" if trailing >= NO_TRAIL else f"{int(trailing * 100)}pct"
            results[f"v35_model_trail_{label}"] = _trade_result(
                test_bars, v35_predictions, assumptions, trailing_pct=trailing, stop_pct=stop_loss_pct,
            )
        # model entries, exit only when the model stops being bullish (no stops at all)
        results["v35_model_exit_only"] = _trade_result(
            test_bars, v35_predictions, assumptions, trailing_pct=NO_TRAIL, stop_pct=NO_TRAIL,
        )
        results["entry_expectancy"] = _entry_expectancy(test_bars, v35_predictions)

        # MOC / close-execution version of the model strategies (levers #2 + #3).
        moc = {}
        for trailing in TRAILING_GRID:
            label = "none" if trailing >= NO_TRAIL else f"{int(trailing * 100)}pct"
            moc[f"v35_model_trail_{label}"] = _close_execution_result(
                test_bars, v35_predictions, moc_assumptions, trailing_pct=trailing, stop_pct=stop_loss_pct,
            )
        moc["v35_model_exit_only"] = _close_execution_result(
            test_bars, v35_predictions, moc_assumptions, trailing_pct=NO_TRAIL, stop_pct=NO_TRAIL,
        )
        results["moc"] = moc

        reentry_check[symbol] = _reentry_noop_check(test_bars, v35_predictions, assumptions, stop_loss_pct)
        per_symbol[symbol] = results

    strategy_keys = [
        "buy_and_hold",
        "always_long_1pct",
        "random_entry_1pct",
        "v35_model_trail_1pct",
        "v35_model_trail_2pct",
        "v35_model_trail_3pct",
        "v35_model_trail_5pct",
        "v35_model_trail_none",
        "v35_model_exit_only",
    ]
    aggregate = {key: _aggregate_strategy(per_symbol, key) for key in strategy_keys}
    moc_aggregate = {key: _aggregate_moc(per_symbol, key) for key in moc_keys}
    expectancy_aggregate = _aggregate_expectancy(per_symbol)
    execution_comparison = {
        key: {
            "stop_market_net": aggregate[key]["mean_net_return"],
            "moc_close_net": moc_aggregate[key]["mean_net_return"],
            "stop_market_trades": aggregate[key]["mean_trades"],
            "moc_trades": moc_aggregate[key]["mean_trades"],
        }
        for key in moc_keys
    }

    result = {
        "window": split_by_day_summary(split_by_day),
        "symbols": eval_symbols,
        "evaluation_window_end": end_date,
        "cost_model": {
            "commission_per_order": assumptions.commission_per_order,
            "quoted_spread_bps": assumptions.quoted_spread_bps,
            "market_impact_bps": assumptions.market_impact_bps,
            "stop_extra_slippage_bps": assumptions.stop_extra_slippage_bps,
            "finra_taf_per_share_sell": assumptions.finra_taf_per_share_sell,
            "finra_taf_cap_per_trade": assumptions.finra_taf_cap_per_trade,
            "sec_fee_per_million_sell": assumptions.sec_fee_per_million_sell,
            "max_bar_participation_rate": assumptions.max_bar_participation_rate,
            "iex_volume_scale": volume_scale,
            "data_feed": getattr(settings, "alpaca_data_feed", "unknown"),
            "median_test_dollar_volume_scaled": {sym: round(val, 2) for sym, val in median_dollar_volume.items()},
            "note": (
                "Commission-free (Alpaca). Volume scaled x%g to undo IEX under-reporting so the 0.05%% "
                "participation cap reflects consolidated liquidity; SEC Section 31 sell fee included."
            ) % volume_scale,
        },
        "fixed_stop_loss_pct": stop_loss_pct,
        "random_seeds": seeds,
        "entry_rate_by_symbol": entry_rate_by_symbol,
        "per_symbol": per_symbol,
        "aggregate": aggregate,
        "moc_aggregate": moc_aggregate,
        "execution_comparison": execution_comparison,
        "moc_cost_profile": {
            "quoted_spread_bps": moc_assumptions.quoted_spread_bps,
            "market_impact_bps": moc_assumptions.market_impact_bps,
            "stop_extra_slippage_bps": moc_assumptions.stop_extra_slippage_bps,
            "note": "close-to-close decisions, fills at the close auction, no intraday stop-gap",
        },
        "entry_expectancy_aggregate": expectancy_aggregate,
        "no_same_day_reentry_check": reentry_check,
        "interpretation": _interpretation(aggregate, expectancy_aggregate, reentry_check),
    }
    (output_dir / "controls_run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "controls_summary.md").write_text(_summary_markdown(result), encoding="utf-8")
    _write_chart(output_dir, aggregate)
    _write_execution_chart(output_dir, execution_comparison)
    return result


# ------------------------------------------------------------------ data utils


def _load_bars(client: AlpacaHistoricalClient, symbols: list, max_bars: int) -> dict:
    raw = client.get_recent_bars(symbols, max_bars, timeframe="day")
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
        for symbol, values in raw.items()
    }


def _test_bars(bars: list, split_by_day: dict) -> list:
    return [bar for bar in bars if split_by_day.get(_bar_day(bar)) == "test"]


def _scale_volume(bars: list, scale: float) -> list:
    if scale == 1.0:
        return bars
    return [replace(bar, volume=(bar.volume or 0.0) * scale) for bar in bars]


def _median(values) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0


def _entry_rate(test_bars: list, predictions: dict) -> float:
    if not test_bars:
        return 0.0
    entries = sum(
        1
        for bar in test_bars
        if (pred := predictions.get(_bar_day(bar))) is not None
        and pred.direction == "bullish"
        and pred.strength in {"medium", "high"}
    )
    return entries / len(test_bars)


def _constant_predictions(test_bars: list, symbol: str) -> dict:
    return {
        _bar_day(bar): DirectionalPrediction(
            symbol=symbol, trading_day=_bar_day(bar), direction="bullish", strength="high",
            relative_to_spy="outperform", event_type="other", session="mixed", raw_completion="",
        )
        for bar in test_bars
    }


# --------------------------------------------------------------- trade running


def _trade_result(test_bars, predictions, assumptions, *, trailing_pct, stop_pct, minimum_strength="medium",
                  no_same_day_reentry=True) -> dict:
    risk = RiskLimits.from_file().with_overrides(stop_loss_pct=stop_pct, trailing_stop_pct=trailing_pct)
    trace = build_directional_trade_trace(
        bars=test_bars,
        predictions_by_day=predictions,
        definition=DirectionalTradeDefinition(
            name="control", minimum_strength=minimum_strength,
            require_relative_confirmation=False, require_momentum_confirmation=False,
        ),
        risk_limits=risk,
        no_same_day_reentry=no_same_day_reentry,
    )
    gross = trace["summary"]
    net = run_execution_aware_replay(test_bars, relabel_stop_events(trace), assumptions)["summary"]
    return {
        "gross_return": round(float(gross["total_return"]), 6),
        "net_return": round(float(net["total_return"]), 6),
        "gross_max_drawdown": round(float(gross["max_drawdown"]), 6),
        "net_max_drawdown": round(float(net["max_drawdown"]), 6),
        "trades": int(net["trades"]),
        "win_rate": round(float(net["win_rate"]), 6),
        "total_fees_paid": round(float(net.get("total_fees_paid", 0.0)), 4),
        "total_slippage_cost": round(float(net.get("total_slippage_cost", 0.0)), 4),
        "partial_fill_events": int(net.get("partial_fill_events", 0)),
    }


_STRENGTH_ORDER = {"low": 1, "medium": 2, "high": 3}


def _meets(prediction, minimum_strength: str) -> bool:
    return (
        prediction.direction == "bullish"
        and _STRENGTH_ORDER.get(prediction.strength, 0) >= _STRENGTH_ORDER.get(minimum_strength, 0)
    )


def _close_execution_result(test_bars, predictions, moc_assumptions, *, trailing_pct, stop_pct, minimum_strength="medium") -> dict:
    """Levers #2 + #3: decide and fill only at the close (MOC), evaluate the trailing/fixed stop
    on close-to-close prices. No intrabar high/low -> no look-ahead, no stop-gap; fills at the close
    auction with a low-impact cost profile."""
    equity_curve = [1.0]
    cash, units = 1.0, 0.0
    in_position = False
    entry_close = high_water = trade_start = None
    events = []
    for bar in test_bars:
        day = _bar_day(bar)
        close = bar.close
        prediction = predictions.get(day)
        wants_long = prediction is not None and _meets(prediction, minimum_strength)
        exited = False
        if in_position:
            high_water = max(high_water, close)
            effective_stop = max(entry_close * (1 - stop_pct), high_water * (1 - trailing_pct))
            if close <= effective_stop or not wants_long:
                cash += units * close
                events.append({"timestamp": bar.timestamp, "event": "sell", "price": round(close, 6),
                               "reason": "close_exit", "fraction": 1.0})
                in_position, entry_close, high_water, trade_start, units, exited = False, None, None, None, 0.0, True
        if not in_position and not exited and wants_long:
            trade_start, entry_close, high_water = cash, close, close
            units, cash = (cash / close if close > 0 else 0.0), 0.0
            in_position = True
            events.append({"timestamp": bar.timestamp, "event": "buy", "price": round(close, 6),
                           "reason": "directional_entry", "fraction": 1.0})
        equity_curve.append(cash + units * close)

    gross_return = equity_curve[-1] - 1.0
    peak, gross_dd = equity_curve[0], 0.0
    for value in equity_curve:
        peak = max(peak, value)
        gross_dd = min(gross_dd, (value / peak) - 1.0 if peak > 0 else 0.0)
    trace = {"summary": {"run_type": "close_execution", "strategy_name": "moc"}, "events": events}
    net = run_execution_aware_replay(test_bars, trace, moc_assumptions)["summary"]
    return {
        "gross_return": round(gross_return, 6),
        "net_return": round(float(net["total_return"]), 6),
        "gross_max_drawdown": round(gross_dd, 6),
        "net_max_drawdown": round(float(net["max_drawdown"]), 6),
        "trades": int(net["trades"]),
        "total_fees_paid": round(float(net.get("total_fees_paid", 0.0)), 4),
        "total_slippage_cost": round(float(net.get("total_slippage_cost", 0.0)), 4),
        "partial_fill_events": int(net.get("partial_fill_events", 0)),
    }


def _random_entry_result(test_bars, symbol, entry_rate, assumptions, *, trailing_pct, stop_pct, seeds) -> dict:
    nets, grosses, net_dds, trades = [], [], [], []
    for seed in range(seeds):
        rng = random.Random(seed)
        predictions = {}
        for bar in test_bars:
            if rng.random() < entry_rate:
                day = _bar_day(bar)
                predictions[day] = DirectionalPrediction(
                    symbol=symbol, trading_day=day, direction="bullish", strength="medium",
                    relative_to_spy="inline", event_type="other", session="mixed", raw_completion="",
                )
        outcome = _trade_result(test_bars, predictions, assumptions, trailing_pct=trailing_pct, stop_pct=stop_pct)
        nets.append(outcome["net_return"])
        grosses.append(outcome["gross_return"])
        net_dds.append(outcome["net_max_drawdown"])
        trades.append(outcome["trades"])
    return {
        "gross_return": round(_mean(grosses), 6),
        "net_return": round(_mean(nets), 6),
        "net_return_std": round(_std(nets), 6),
        "net_return_p05": round(sorted(nets)[max(0, int(0.05 * len(nets)) - 1)], 6),
        "net_return_p95": round(sorted(nets)[min(len(nets) - 1, int(0.95 * len(nets)))], 6),
        "net_max_drawdown": round(_mean(net_dds), 6),
        "trades": round(_mean([float(value) for value in trades]), 2),
        "seeds": seeds,
    }


def _buy_and_hold(test_bars, assumptions) -> dict:
    if not test_bars:
        return {"gross_return": 0.0, "net_return": 0.0, "gross_max_drawdown": 0.0, "net_max_drawdown": 0.0, "trades": 0}
    # all-bullish + disabled stops => one entry on the first bar, held to the last.
    predictions = _constant_predictions(test_bars, test_bars[0].timestamp)
    return _trade_result(test_bars, predictions, assumptions, trailing_pct=NO_TRAIL, stop_pct=NO_TRAIL)


def _entry_expectancy(test_bars, predictions) -> dict:
    closes = [bar.close for bar in test_bars]
    eligible = range(0, len(test_bars) - LABEL_HORIZON_BARS)
    entry_fwd, all_fwd = [], []
    for index in eligible:
        fwd = (closes[index + LABEL_HORIZON_BARS] / closes[index]) - 1.0 if closes[index] > 0 else 0.0
        all_fwd.append(fwd)
        pred = predictions.get(_bar_day(test_bars[index]))
        if pred is not None and pred.direction == "bullish" and pred.strength in {"medium", "high"}:
            entry_fwd.append(fwd)
    return {
        "entry_days": len(entry_fwd),
        "eligible_days": len(all_fwd),
        "mean_fwd5_on_entry": round(_mean(entry_fwd), 6) if entry_fwd else None,
        "mean_fwd5_unconditional": round(_mean(all_fwd), 6) if all_fwd else None,
        "edge_fwd5": round(_mean(entry_fwd) - _mean(all_fwd), 6) if entry_fwd and all_fwd else None,
        "hit_rate_on_entry": round(_mean([1.0 if value > 0 else 0.0 for value in entry_fwd]), 6) if entry_fwd else None,
        "hit_rate_unconditional": round(_mean([1.0 if value > 0 else 0.0 for value in all_fwd]), 6) if all_fwd else None,
    }


def _reentry_noop_check(test_bars, predictions, assumptions, stop_pct) -> dict:
    with_flag = _trade_result(test_bars, predictions, assumptions, trailing_pct=0.01, stop_pct=stop_pct, no_same_day_reentry=True)
    without_flag = _trade_result(test_bars, predictions, assumptions, trailing_pct=0.01, stop_pct=stop_pct, no_same_day_reentry=False)
    return {
        "net_return_with_flag": with_flag["net_return"],
        "net_return_without_flag": without_flag["net_return"],
        "trades_with_flag": with_flag["trades"],
        "trades_without_flag": without_flag["trades"],
        "identical": with_flag["net_return"] == without_flag["net_return"] and with_flag["trades"] == without_flag["trades"],
    }


# ---------------------------------------------------------------- aggregation


def _aggregate_strategy(per_symbol: dict, key: str) -> dict:
    rows = [per_symbol[symbol][key] for symbol in per_symbol]
    return {
        "mean_gross_return": round(_mean([row["gross_return"] for row in rows]), 6),
        "mean_net_return": round(_mean([row["net_return"] for row in rows]), 6),
        "mean_net_max_drawdown": round(_mean([row.get("net_max_drawdown", 0.0) for row in rows]), 6),
        "mean_trades": round(_mean([float(row.get("trades", 0)) for row in rows]), 2),
        "total_partial_fills": sum(int(row.get("partial_fill_events", 0)) for row in rows),
    }


def _aggregate_moc(per_symbol: dict, key: str) -> dict:
    rows = [per_symbol[symbol]["moc"][key] for symbol in per_symbol]
    return {
        "mean_gross_return": round(_mean([row["gross_return"] for row in rows]), 6),
        "mean_net_return": round(_mean([row["net_return"] for row in rows]), 6),
        "mean_net_max_drawdown": round(_mean([row["net_max_drawdown"] for row in rows]), 6),
        "mean_trades": round(_mean([float(row["trades"]) for row in rows]), 2),
    }


def _aggregate_expectancy(per_symbol: dict) -> dict:
    rows = [per_symbol[symbol]["entry_expectancy"] for symbol in per_symbol]
    edges = [row["edge_fwd5"] for row in rows if row["edge_fwd5"] is not None]
    return {
        "mean_edge_fwd5": round(_mean(edges), 6) if edges else None,
        "symbols_with_positive_edge": sum(1 for value in edges if value > 0),
        "per_symbol_edge": {symbol: per_symbol[symbol]["entry_expectancy"]["edge_fwd5"] for symbol in per_symbol},
    }


def _interpretation(aggregate, expectancy, reentry_check) -> dict:
    v35 = aggregate["v35_model_trail_1pct"]
    bh = aggregate["buy_and_hold"]
    rnd = aggregate["random_entry_1pct"]
    always = aggregate["always_long_1pct"]
    gross_grid = {key: aggregate[key]["mean_gross_return"] for key in aggregate if key.startswith("v35_model_trail")}
    net_grid = {key: aggregate[key]["mean_net_return"] for key in aggregate if key.startswith("v35_model_trail")}
    return {
        "v35_1pct_gross_vs_net": [v35["mean_gross_return"], v35["mean_net_return"]],
        "model_beats_random_net": v35["mean_net_return"] > rnd["mean_net_return"],
        "model_beats_buy_and_hold_net": v35["mean_net_return"] > bh["mean_net_return"],
        "model_beats_always_long_net": v35["mean_net_return"] > always["mean_net_return"],
        "gross_best_trailing": max(gross_grid, key=gross_grid.get),
        "net_best_trailing": max(net_grid, key=net_grid.get),
        "entry_edge_positive": expectancy["mean_edge_fwd5"] is not None and expectancy["mean_edge_fwd5"] > 0,
        "no_same_day_reentry_is_noop_on_daily": all(row["identical"] for row in reentry_check.values()),
    }


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _std(values) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def split_by_day_summary(split_by_day: dict) -> dict:
    from collections import defaultdict

    by_split = defaultdict(list)
    for day, split in sorted(split_by_day.items()):
        by_split[split].append(day)
    return {split: f"{days[0]} -> {days[-1]} ({len(days)} days)" for split, days in by_split.items() if days}


# -------------------------------------------------------------------- reporting


def _write_chart(output_dir: Path, aggregate: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = ["buy_and_hold", "always_long_1pct", "random_entry_1pct", "v35_model_trail_1pct",
              "v35_model_trail_2pct", "v35_model_trail_3pct", "v35_model_trail_5pct", "v35_model_trail_none"]
    net = [aggregate[key]["mean_net_return"] * 100 for key in labels]
    gross = [aggregate[key]["mean_gross_return"] * 100 for key in labels]
    figure, axis = plt.subplots(figsize=(11, 5))
    x = range(len(labels))
    axis.bar([i - 0.2 for i in x], gross, width=0.4, label="gross", color="#9aa7c7")
    axis.bar([i + 0.2 for i in x], net, width=0.4, label="net of costs", color="#4C72B0")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("mean return (percent)")
    axis.set_title("V3-5 trade-mapper controls: gross vs net, same window")
    axis.set_xticks(list(x))
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "controls_gross_vs_net.png", dpi=130)
    plt.close(figure)


def _write_execution_chart(output_dir: Path, execution_comparison: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = list(execution_comparison.keys())
    stop_market = [execution_comparison[key]["stop_market_net"] * 100 for key in labels]
    moc = [execution_comparison[key]["moc_close_net"] * 100 for key in labels]
    figure, axis = plt.subplots(figsize=(11, 5))
    x = range(len(labels))
    axis.bar([i - 0.2 for i in x], stop_market, width=0.4, label="stop-market net", color="#C44E52")
    axis.bar([i + 0.2 for i in x], moc, width=0.4, label="MOC/close net", color="#55A868")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("mean net return (percent)")
    axis.set_title("Execution method: stop-market vs MOC/close (net of costs, same window)")
    axis.set_xticks(list(x))
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "controls_execution_method.png", dpi=130)
    plt.close(figure)


def _summary_markdown(result: dict) -> str:
    agg = result["aggregate"]
    lines = [
        "# V3-5 price-aware execution controls",
        "",
        f"- window (test): `{result['window'].get('test', 'n/a')}`",
        f"- symbols: `{', '.join(result['symbols'])}`",
        f"- fixed stop: `{result['fixed_stop_loss_pct']:.0%}` · random seeds: `{result['random_seeds']}`",
        "",
        "## Cost model (Alpaca-aligned)",
        "",
        f"- commission `${result['cost_model']['commission_per_order']:.2f}` (commission-free), "
        f"data feed `{result['cost_model']['data_feed']}`",
        f"- slippage `{result['cost_model']['quoted_spread_bps']:.1f}`bps spread + "
        f"`{result['cost_model']['market_impact_bps']:.1f}`bps impact (+`{result['cost_model']['stop_extra_slippage_bps']:.1f}`bps on stops)",
        f"- FINRA TAF `${result['cost_model']['finra_taf_per_share_sell']}`/sh cap `${result['cost_model']['finra_taf_cap_per_trade']}`, "
        f"SEC 31 `${result['cost_model']['sec_fee_per_million_sell']}`/$1M sold",
        f"- participation cap `{result['cost_model']['max_bar_participation_rate']:.4%}` of $-volume, "
        f"volume scaled `x{result['cost_model']['iex_volume_scale']:.0f}` to undo IEX under-reporting",
        "",
        "## Strategy comparison (mean over symbols)",
        "",
        "| Strategy | Gross return | **Net return** | Net max DD | Trades | Partial fills |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    order = ["buy_and_hold", "always_long_1pct", "random_entry_1pct", "v35_model_trail_1pct",
             "v35_model_trail_2pct", "v35_model_trail_3pct", "v35_model_trail_5pct",
             "v35_model_trail_none", "v35_model_exit_only"]
    for key in order:
        row = agg[key]
        lines.append(
            f"| `{key}` | `{row['mean_gross_return']:.2%}` | `{row['mean_net_return']:.2%}` | "
            f"`{row['mean_net_max_drawdown']:.2%}` | `{row['mean_trades']}` | `{row.get('total_partial_fills', 0)}` |"
        )
    comp = result["execution_comparison"]
    moc = result["moc_cost_profile"]
    lines.extend([
        "",
        "## Cheaper execution: stop-market vs MOC/close (levers #2 + #3)",
        "",
        f"MOC profile: `{moc['quoted_spread_bps']:.1f}`bps spread + `{moc['market_impact_bps']:.1f}`bps impact, "
        f"`{moc['stop_extra_slippage_bps']:.1f}`bps stop-extra ({moc['note']}).",
        "",
        "| Model strategy | Stop-market net | Stop-mkt trades | **MOC/close net** | MOC trades |",
        "|---|---:|---:|---:|---:|",
    ])
    for key in ["v35_model_trail_1pct", "v35_model_trail_2pct", "v35_model_trail_3pct",
                "v35_model_trail_5pct", "v35_model_trail_none", "v35_model_exit_only"]:
        row = comp[key]
        lines.append(
            f"| `{key}` | `{row['stop_market_net']:.2%}` | `{row['stop_market_trades']}` | "
            f"`{row['moc_close_net']:.2%}` | `{row['moc_trades']}` |"
        )
    exp = result["entry_expectancy_aggregate"]
    interp = result["interpretation"]
    lines.extend([
        "",
        "## Entry edge (forward 5-day return on bullish entry days vs unconditional)",
        "",
        f"- mean edge (entry − unconditional): `{_fmt_pct(exp['mean_edge_fwd5'])}`",
        f"- symbols with positive entry edge: `{exp['symbols_with_positive_edge']}/{len(result['symbols'])}`",
        f"- per-symbol edge: `{ {s: _fmt_pct(v) for s, v in exp['per_symbol_edge'].items()} }`",
        "",
        "## Verdicts",
        "",
        f"- V3-5 (1% trailing) gross vs net: `{interp['v35_1pct_gross_vs_net'][0]:.2%}` -> `{interp['v35_1pct_gross_vs_net'][1]:.2%}`",
        f"- model entries beat **random** entries (same rate), net: `{interp['model_beats_random_net']}`",
        f"- model entries beat **buy & hold**, net: `{interp['model_beats_buy_and_hold_net']}`",
        f"- model entries beat **always-long** (stop only), net: `{interp['model_beats_always_long_net']}`",
        f"- gross-best trailing width: `{interp['gross_best_trailing']}` · net-best: `{interp['net_best_trailing']}`",
        f"- entry edge positive: `{interp['entry_edge_positive']}`",
        f"- no-same-day-reentry is a no-op on daily bars: `{interp['no_same_day_reentry_is_noop_on_daily']}`",
        "",
    ])
    return "\n".join(lines) + "\n"


def _fmt_pct(value) -> str:
    return "n/a" if value is None else f"{value:.2%}"
