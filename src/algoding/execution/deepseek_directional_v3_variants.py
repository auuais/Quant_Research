from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from algoding.data.historical import AlpacaHistoricalClient
from algoding.execution.deepseek_directional_research import DeepseekDirectionalResearchLab
from algoding.portfolio.risk import RiskLimits
from algoding.research.deepseek_directional_model import (
    DirectionalPrediction,
    DirectionalTradeDefinition,
    build_directional_trade_trace,
)
from algoding.research.historical_backtest import BacktestBar
from algoding.settings import Settings


def run_v3_top5_trailing_no_same_day_reentry(
    *,
    settings: Settings,
    output_root: str = "reports/research/deepseek_directional_v3",
    max_bars: int = 1400,
) -> dict[str, object]:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    version = _next_version(output_dir)
    stem = f"{version.lower().replace('-', '_')}_top5_trailing_1pct_no_same_day_reentry"

    prior_run = json.loads((output_dir / "deepseek_directional_run.json").read_text(encoding="utf-8"))
    evaluation_symbols = [str(value).upper() for value in prior_run["evaluation_symbols"]]
    metrics_by_symbol = {
        str(row["symbol"]).upper(): {
            "direction_accuracy": float(row["test"]["finetuned_metrics"]["direction_accuracy"]),
            "direction_macro_f1": float(row["test"]["finetuned_metrics"]["direction_macro_f1"]),
        }
        for row in prior_run["comparison_rows"]
    }

    history_client = AlpacaHistoricalClient(settings)
    split_helper = DeepseekDirectionalResearchLab(settings)
    directional_cache = _load_directional_cache(
        output_dir / "finetuned_directional_scores.jsonl",
        symbols=evaluation_symbols,
    )
    bars_by_symbol_raw = history_client.get_recent_bars(sorted(set(evaluation_symbols + ["SPY"])), max_bars, timeframe="day")
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
    risk_limits = RiskLimits.from_file().with_overrides(stop_loss_pct=0.05, trailing_stop_pct=0.01)

    rows: list[dict[str, object]] = []
    for symbol in evaluation_symbols:
        bars = bars_by_symbol[symbol]
        test_bars = [
            bar
            for bar in bars
            if split_by_day.get(datetime.fromisoformat(bar.timestamp).date().isoformat()) == "test"
        ]
        predictions = {
            trading_day: directional_cache[symbol][trading_day]
            for trading_day in (
                datetime.fromisoformat(bar.timestamp).date().isoformat()
                for bar in test_bars
            )
            if trading_day in directional_cache.get(symbol, {})
        }
        trace = build_directional_trade_trace(
            bars=test_bars,
            predictions_by_day=predictions,
            definition=DirectionalTradeDefinition(
                name=f"{symbol.lower()}_{stem}",
                minimum_strength="medium",
                require_relative_confirmation=False,
                require_momentum_confirmation=False,
            ),
            risk_limits=risk_limits,
            no_same_day_reentry=True,
        )
        summary = trace["summary"]
        rows.append(
            {
                "symbol": symbol,
                "test_period": (
                    f"{datetime.fromisoformat(test_bars[0].timestamp).date().isoformat()} -> "
                    f"{datetime.fromisoformat(test_bars[-1].timestamp).date().isoformat()} "
                    f"({len(test_bars)} days)"
                ),
                "direction_accuracy": round(metrics_by_symbol[symbol]["direction_accuracy"], 6),
                "direction_macro_f1": round(metrics_by_symbol[symbol]["direction_macro_f1"], 6),
                "return": float(summary["total_return"]),
                "max_drawdown": float(summary["max_drawdown"]),
                "trades": int(summary["trades"]),
                "win_rate": float(summary["win_rate"]),
                "latest_signal": str(summary["latest_signal"]),
                "exit_reasons": dict(summary["exit_reasons"]),
            }
        )

    aggregate = {
        "mean_direction_accuracy": round(sum(float(row["direction_accuracy"]) for row in rows) / len(rows), 6),
        "mean_direction_macro_f1": round(sum(float(row["direction_macro_f1"]) for row in rows) / len(rows), 6),
        "mean_return": round(sum(float(row["return"]) for row in rows) / len(rows), 6),
        "mean_max_drawdown": round(sum(float(row["max_drawdown"]) for row in rows) / len(rows), 6),
        "mean_trades": round(sum(float(row["trades"]) for row in rows) / len(rows), 2),
    }
    result = {
        "version": version,
        "policy": {
            "minimum_strength": "medium",
            "require_relative_confirmation": False,
            "require_momentum_confirmation": False,
            "stop_loss_pct": 0.05,
            "trailing_stop_pct": 0.01,
            "no_same_day_reentry": True,
        },
        "rows": rows,
        "aggregate": aggregate,
    }

    json_path = output_dir / f"{stem}.json"
    summary_path = output_dir / f"{stem}_summary.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary_path.write_text(_build_summary_markdown(result), encoding="utf-8")
    return result


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


def _next_version(output_dir: Path) -> str:
    for index in range(5, 20):
        version = f"V3-{index}"
        stem = f"{version.lower().replace('-', '_')}_top5_trailing_1pct_no_same_day_reentry.json"
        if not (output_dir / stem).exists():
            return version
    raise RuntimeError("Unable to allocate a new V3 variant version.")


def _build_summary_markdown(result: dict[str, object]) -> str:
    lines = [
        f"# {result['version']} on Top 5 Stocks with 1% Trailing Stop and No Same-Day Re-entry",
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
        "| Symbol | Direction accuracy | Macro F1 | Total return | Max drawdown | Trades |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| `{row['symbol']}` | `{row['direction_accuracy']:.2%}` | `{row['direction_macro_f1']:.4f}` | "
            f"`{row['return']:.2%}` | `{row['max_drawdown']:.2%}` | `{row['trades']}` |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- mean direction accuracy: `{result['aggregate']['mean_direction_accuracy']:.2%}`",
            f"- mean macro F1: `{result['aggregate']['mean_direction_macro_f1']:.4f}`",
            f"- mean total return: `{result['aggregate']['mean_return']:.2%}`",
            f"- mean max drawdown: `{result['aggregate']['mean_max_drawdown']:.2%}`",
            f"- mean trades: `{result['aggregate']['mean_trades']}`",
        ]
    )
    return "\n".join(lines) + "\n"
