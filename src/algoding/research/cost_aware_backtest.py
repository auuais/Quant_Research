from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import sqrt

from algoding.research.historical_backtest import BacktestBar, _infer_periods_per_year, _max_drawdown


@dataclass(frozen=True)
class ExecutionAssumptions:
    starting_capital: float = 100_000.0
    commission_per_order: float = 0.0
    buy_fee_bps: float = 0.0
    sell_fee_bps: float = 0.0
    quoted_spread_bps: float = 1.0
    market_impact_bps: float = 1.0
    stop_extra_slippage_bps: float = 1.0
    max_bar_participation_rate: float = 0.005
    max_bar_shares: float | None = None
    sec_fee_per_million_sell: float = 0.0
    finra_taf_per_share_sell: float = 0.000195
    finra_taf_cap_per_trade: float = 9.79


def run_execution_aware_replay(
    bars: list[BacktestBar],
    trace: dict[str, object],
    assumptions: ExecutionAssumptions,
) -> dict[str, object]:
    events_by_ts: dict[str, list[dict[str, object]]] = defaultdict(list)
    for event in trace["events"]:
        events_by_ts[str(event["timestamp"])].append(dict(event))

    cash = assumptions.starting_capital
    qty = 0.0
    equity_curve: list[float] = [cash]
    trade_returns: list[float] = []
    trade_pnls: list[float] = []
    exit_reasons: Counter[str] = Counter()
    partial_fill_events = 0
    total_fees = 0.0
    total_slippage_cost = 0.0
    current_trade_total_cost = 0.0
    current_trade_remaining_cost_basis = 0.0
    current_trade_realized_pnl = 0.0

    for bar in bars:
        for event in events_by_ts.get(bar.timestamp, []):
            reason = str(event.get("reason", ""))
            if event["event"] == "buy" and qty <= 0:
                desired_cash = cash
                base_slippage_bps = (assumptions.quoted_spread_bps / 2.0) + assumptions.market_impact_bps
                reference_price = float(event["price"])
                fill_price = reference_price * (1 + (base_slippage_bps / 10_000.0))

                participation_notional = _bar_dollar_volume(bar) * assumptions.max_bar_participation_rate
                if assumptions.max_bar_shares is not None and assumptions.max_bar_shares > 0:
                    participation_notional = assumptions.max_bar_shares * fill_price
                max_spend = desired_cash - assumptions.commission_per_order
                if participation_notional > 0:
                    max_spend = min(max_spend, participation_notional)
                if max_spend <= 0:
                    continue
                if max_spend + assumptions.commission_per_order < desired_cash - 1e-9:
                    partial_fill_events += 1

                fill_qty = max_spend / fill_price
                total_slippage_cost += max(0.0, fill_price - reference_price) * fill_qty
                gross_cost = fill_qty * fill_price
                variable_fee = gross_cost * (assumptions.buy_fee_bps / 10_000.0)
                fees = assumptions.commission_per_order + variable_fee
                spend = gross_cost + fees
                cash -= spend
                qty += fill_qty
                current_trade_total_cost += spend
                current_trade_remaining_cost_basis += spend
                total_fees += fees

            elif event["event"] == "sell" and qty > 0:
                requested_fraction = float(event.get("fraction", 1.0))
                requested_fraction = min(max(requested_fraction, 0.0), 1.0)
                base_slippage_bps = (assumptions.quoted_spread_bps / 2.0) + assumptions.market_impact_bps
                if reason in {"stop_loss", "trailing_stop"}:
                    base_slippage_bps += assumptions.stop_extra_slippage_bps

                reference_price = float(event["price"])
                if reason in {"stop_loss", "trailing_stop"} and bar.open is not None and bar.open < reference_price:
                    reference_price = float(bar.open)

                fill_price = reference_price * (1 - (base_slippage_bps / 10_000.0))
                if reason in {"stop_loss", "trailing_stop"}:
                    fill_price = max(float(bar.low), fill_price)

                qty_before = qty
                requested_qty = qty_before * requested_fraction
                max_qty = requested_qty
                if assumptions.max_bar_shares is not None and assumptions.max_bar_shares > 0:
                    max_qty = min(max_qty, assumptions.max_bar_shares)
                elif bar.volume is not None and bar.volume > 0:
                    max_qty = min(max_qty, float(bar.volume) * assumptions.max_bar_participation_rate)
                if max_qty < requested_qty - 1e-12:
                    partial_fill_events += 1

                total_slippage_cost += max(0.0, float(event["price"]) - fill_price) * max_qty
                gross_proceeds = max_qty * fill_price
                variable_fee = gross_proceeds * (assumptions.sell_fee_bps / 10_000.0)
                sec_fee = (gross_proceeds / 1_000_000.0) * assumptions.sec_fee_per_million_sell
                taf_fee = min(max_qty * assumptions.finra_taf_per_share_sell, assumptions.finra_taf_cap_per_trade)
                fees = assumptions.commission_per_order + variable_fee + sec_fee + taf_fee
                proceeds = gross_proceeds - fees
                cash += proceeds
                qty -= max_qty
                total_fees += fees
                allocated_cost_basis = (
                    current_trade_remaining_cost_basis * (max_qty / qty_before)
                    if qty_before > 0
                    else 0.0
                )
                current_trade_remaining_cost_basis -= allocated_cost_basis
                current_trade_realized_pnl += proceeds - allocated_cost_basis
                exit_reasons[reason or "signal_exit"] += 1

                if qty <= 1e-12:
                    qty = 0.0
                    trade_pnl = current_trade_realized_pnl
                    trade_pnls.append(trade_pnl)
                    trade_returns.append(trade_pnl / current_trade_total_cost if current_trade_total_cost > 0 else 0.0)
                    current_trade_total_cost = 0.0
                    current_trade_remaining_cost_basis = 0.0
                    current_trade_realized_pnl = 0.0

        equity_curve.append(cash + (qty * bar.close))

    total_return = (equity_curve[-1] / assumptions.starting_capital) - 1 if assumptions.starting_capital > 0 else 0.0
    max_drawdown = _max_drawdown(equity_curve)
    positive = [value for value in trade_pnls if value > 0]
    negative = [value for value in trade_pnls if value < 0]
    win_rate = len(positive) / len(trade_pnls) if trade_pnls else 0.0
    periods_per_year = _infer_periods_per_year(bars)
    annualized_return = (
        ((1 + total_return) ** (periods_per_year / max(1, len(bars) - 1))) - 1 if total_return > -1 else -1.0
    )
    periodic_returns = [(equity_curve[i] / equity_curve[i - 1]) - 1 for i in range(1, len(equity_curve))]
    variance = sum(value * value for value in periodic_returns) / max(1, len(periodic_returns))
    annualized_volatility = sqrt(variance) * sqrt(periods_per_year) if periodic_returns else 0.0
    profit_factor = (
        sum(positive) / abs(sum(negative))
        if negative and abs(sum(negative)) > 0
        else float(len(positive)) if positive else 0.0
    )
    score = total_return + (max_drawdown * 0.5) + (win_rate * 0.1) + (min(profit_factor, 3.0) * 0.02)

    summary = dict(trace["summary"])
    source_run_type = str(summary.get("run_type", "historical"))
    summary.update(
        {
            "source_run_type": source_run_type,
            "run_type": f"{source_run_type}_execution_aware",
            "execution_assumptions": {
                "starting_capital": assumptions.starting_capital,
                "commission_per_order": assumptions.commission_per_order,
                "buy_fee_bps": assumptions.buy_fee_bps,
                "sell_fee_bps": assumptions.sell_fee_bps,
                "quoted_spread_bps": assumptions.quoted_spread_bps,
                "market_impact_bps": assumptions.market_impact_bps,
                "stop_extra_slippage_bps": assumptions.stop_extra_slippage_bps,
                "max_bar_participation_rate": assumptions.max_bar_participation_rate,
                "max_bar_shares": assumptions.max_bar_shares,
                "sec_fee_per_million_sell": assumptions.sec_fee_per_million_sell,
                "finra_taf_per_share_sell": assumptions.finra_taf_per_share_sell,
                "finra_taf_cap_per_trade": assumptions.finra_taf_cap_per_trade,
            },
            "total_return": round(total_return, 6),
            "annualized_return": round(annualized_return, 6),
            "annualized_volatility": round(annualized_volatility, 6),
            "max_drawdown": round(max_drawdown, 6),
            "trades": len(trade_pnls),
            "win_rate": round(win_rate, 6),
            "profit_factor": round(profit_factor, 6),
            "score": round(score, 6),
            "partial_fill_events": partial_fill_events,
            "total_fees_paid": round(total_fees, 6),
            "total_slippage_cost": round(total_slippage_cost, 6),
            "final_equity": round(equity_curve[-1], 6),
            "exit_reasons": dict(exit_reasons),
        }
    )
    return {
        "summary": summary,
        "equity_curve": equity_curve,
        "trade_returns": trade_returns,
    }


def _bar_dollar_volume(bar: BacktestBar) -> float:
    if bar.volume is None or bar.close <= 0:
        return 0.0
    return float(bar.volume) * float(bar.close)
