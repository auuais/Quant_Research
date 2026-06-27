"""Close-to-close ("market-on-close") execution model.

Decisions and fills happen only at the bar close; the trailing/fixed stop is evaluated on
close-to-close prices. Because no intrabar high/low is used, there is no look-ahead and no
stop-gap fill -- the gross trace is realistic and lines up with the net (cost-aware) replay.
This is the Alpaca-executable counterpart to the idealized intraday-stop trace in
``deepseek_directional_model.build_directional_trade_trace``.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from math import sqrt

from algoding.research.deepseek_directional_model import (
    DirectionalPrediction,
    DirectionalTradeDefinition,
    _should_be_long,
)
from algoding.research.historical_backtest import BacktestBar, _infer_periods_per_year, _max_drawdown


# A trailing/stop pct this large is never reached -> disables that leg of the stop.
NO_TRAIL = 10.0


def relabel_stop_events(trace: dict) -> dict:
    """Map a directional trace's generic ``risk_exit`` to ``trailing_stop`` so the cost-aware replay
    applies stop slippage + the intrabar low floor. Used only for the *stop-market* execution model
    (the close-execution model emits ``close_exit`` and needs no relabeling)."""
    reason_map = {"risk_exit": "trailing_stop", "prediction_exit": "signal_exit"}
    events = [dict(event, reason=reason_map.get(str(event.get("reason")), event.get("reason"))) for event in trace["events"]]
    return {"summary": trace["summary"], "events": events, "trace_points": trace.get("trace_points", [])}


def _conviction_fraction(prediction) -> float:
    """Position size as a fraction of equity, by signal conviction (works for both the
    DeepSeek strength buckets and the calibrated-classifier strength)."""
    return {"low": 0.25, "medium": 0.5, "high": 1.0}.get(prediction.strength, 0.5)


def build_close_execution_trace(
    *,
    bars: list,
    predictions_by_day: dict,
    definition: DirectionalTradeDefinition,
    stop_loss_pct: float,
    trailing_stop_pct: float,
    conviction_sizing: bool = False,
) -> dict:
    equity_curve = [1.0]
    cash, units = 1.0, 0.0
    in_position = False
    entry_close = high_water = trade_start_equity = None
    trade_returns: list = []
    exit_reasons: Counter = Counter()
    events: list = []

    for bar in bars:
        trading_day = datetime.fromisoformat(bar.timestamp).date().isoformat()
        close = bar.close
        prediction = predictions_by_day.get(
            trading_day,
            DirectionalPrediction(
                symbol="", trading_day=trading_day, direction="neutral", strength="low",
                relative_to_spy="inline", event_type="other", session="mixed", raw_completion="",
            ),
        )
        wants_long = _should_be_long(prediction=prediction, definition=definition, momentum_ok=True)
        exited = False
        if in_position and entry_close is not None:
            high_water = max(high_water, close)
            effective_stop = max(entry_close * (1 - stop_loss_pct), high_water * (1 - trailing_stop_pct))
            if close <= effective_stop:
                cash += units * close
                units = 0.0
                if trade_start_equity:
                    trade_returns.append((cash / trade_start_equity) - 1)
                exit_reasons["risk_exit"] += 1
                events.append({"timestamp": bar.timestamp, "event": "sell", "price": round(close, 6),
                               "reason": "close_exit", "fraction": 1.0})
                in_position, entry_close, high_water, trade_start_equity, exited = False, None, None, None, True
            elif not wants_long:
                cash += units * close
                units = 0.0
                if trade_start_equity:
                    trade_returns.append((cash / trade_start_equity) - 1)
                exit_reasons["prediction_exit"] += 1
                events.append({"timestamp": bar.timestamp, "event": "sell", "price": round(close, 6),
                               "reason": "close_exit", "fraction": 1.0})
                in_position, entry_close, high_water, trade_start_equity, exited = False, None, None, None, True
        if not in_position and not exited and wants_long:
            in_position = True
            fraction = _conviction_fraction(prediction) if conviction_sizing else 1.0
            trade_start_equity, entry_close, high_water = cash, close, close
            invested = cash * fraction
            units = invested / close if close > 0 else 0.0
            cash = cash - invested
            events.append({"timestamp": bar.timestamp, "event": "buy", "price": round(close, 6),
                           "reason": "directional_entry", "fraction": 1.0, "buy_fraction": round(fraction, 6)})
        equity_curve.append(cash + units * close)

    positive = [value for value in trade_returns if value > 0]
    negative = [value for value in trade_returns if value < 0]
    total_return = equity_curve[-1] - 1.0
    win_rate = len(positive) / len(trade_returns) if trade_returns else 0.0
    periods_per_year = _infer_periods_per_year(bars)
    annualized_return = (
        ((1 + total_return) ** (periods_per_year / max(1, len(bars) - 1))) - 1 if total_return > -1 else -1.0
    )
    periodic = [(equity_curve[i] / equity_curve[i - 1]) - 1 for i in range(1, len(equity_curve))]
    annualized_volatility = sqrt(sum(v * v for v in periodic) / max(1, len(periodic))) * sqrt(periods_per_year) if periodic else 0.0
    profit_factor = (
        sum(positive) / abs(sum(negative)) if negative and abs(sum(negative)) > 0
        else float(len(positive)) if positive else 0.0
    )
    summary = {
        "strategy_name": definition.name,
        "strategy_family": "close_execution_directional",
        "run_type": "close_execution",
        "latest_signal": "long" if in_position else "flat",
        "total_return": round(total_return, 6),
        "annualized_return": round(annualized_return, 6),
        "annualized_volatility": round(annualized_volatility, 6),
        "max_drawdown": round(_max_drawdown(equity_curve), 6),
        "trades": len(trade_returns),
        "win_rate": round(win_rate, 6),
        "profit_factor": round(profit_factor, 6),
        "bars_tested": len(bars),
        "risk": {
            "stop_loss_pct": stop_loss_pct,
            "trailing_stop_pct": trailing_stop_pct,
            "execution": "market_on_close",
            "minimum_strength": definition.minimum_strength,
        },
        "exit_reasons": dict(exit_reasons),
    }
    return {"summary": summary, "events": events, "trace_points": []}
