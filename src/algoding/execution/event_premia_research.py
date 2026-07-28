"""EVENT-PREMIA-V1: the D2 calendar/announcement premia battery.

Pre-registered in reports/research/frontier_hypotheses.md: 6 families / 14 variants, and an effect
counts only if its bootstrap CI excludes zero in the early AND late subperiod independently and the
mean effect is at least 2x round-trip cost. Failed hypotheses are dropped permanently -- no threshold
re-tuning after seeing results.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from algoding.data.events import (
    EventCalendarClient,
    opex_dates,
    opex_week_flags,
    quarter_end_dates,
    turn_of_month_flags,
)
from algoding.data.volatility import VolatilityDataClient
from algoding.execution.long_short_research import UNIVERSE_100
from algoding.research.event_study import EffectTest, car_profile, deflated_sharpe_hurdle, test_effect


TRADING_DAYS = 252.0
INDEX_SYMBOLS = ("SPY", "QQQ")
PRIMARY_COST_BPS = 1.0
STRESS_COST_BPS = 3.0
DECLARED_VARIANTS = 14


@dataclass(frozen=True)
class SleeveResult:
    name: str
    metrics: dict[str, float]
    time_in_market: float
    round_trips: int


class EventPremiaResearchLab:
    def __init__(self, settings=None) -> None:
        self._settings = settings
        self._events = EventCalendarClient()
        self._volatility = VolatilityDataClient()
        self._panel_cache: dict[str, pd.DataFrame] = {}
        self._last_pead_events: pd.DataFrame | None = None

    def run(
        self,
        *,
        output_root: str = "reports/research/event_premia_v1",
        start: str = "2005-01-01",
        panel_symbols: int = 100,
        include_panel: bool = True,
    ) -> dict[str, object]:
        output_dir = Path(output_root)
        output_dir.mkdir(parents=True, exist_ok=True)

        ohlc = self._events.load_ohlc(list(INDEX_SYMBOLS), start=start)
        if not ohlc:
            raise RuntimeError("could not load index OHLC data")
        fomc = self._events.load_fomc_calendar(start_year=int(start[:4]))
        panel = (
            self._events.load_ohlc(list(UNIVERSE_100)[:panel_symbols], start=start) if include_panel else {}
        )
        self._panel_cache = panel
        vix_frame = self._volatility.build_panel(start=start).frame

        tests: list[EffectTest] = []
        diagnostics: dict[str, object] = {}

        tests += self._e1_pre_fomc(ohlc, fomc.announcements, diagnostics)
        tests += self._e2_overnight_intraday(ohlc, panel, diagnostics)
        tests += self._e3_turn_of_month(ohlc, diagnostics)
        tests += self._e4_opex_quarter_end(ohlc, diagnostics)
        tests += self._e5_gap_pead(panel, diagnostics)
        tests += self._e6_vix_spike(ohlc, vix_frame, diagnostics)

        survivors = [test for test in tests if test.survives]
        sleeves = self._survivor_sleeves(survivors, ohlc, vix_frame, diagnostics)
        observations = int(len(ohlc["SPY"]))
        multiple_testing = deflated_sharpe_hurdle(trials=DECLARED_VARIANTS, observations=observations)

        result: dict[str, object] = {
            "version": "EVENT-PREMIA-V1",
            "direction": "D2",
            "thesis": (
                "documented calendar/announcement premia persist and are harvestable with daily "
                "close-decision orders on index ETFs"
            ),
            "pre_registration": {
                "registry": "reports/research/frontier_hypotheses.md#d2",
                "declared_variants": DECLARED_VARIANTS,
                "realized_variants": len(tests),
                "confirmation_rule": (
                    "bootstrap 95% CI excludes zero in the early 70% AND late 30% independently, same sign, "
                    "and mean effect >= 2x round-trip cost"
                ),
                "gate": "standalone micro-sleeve net Sharpe >= 0.8 with <= 10% DD; overlay +0.03 portfolio Sharpe",
            },
            "config": {
                "start": start,
                "index_symbols": list(INDEX_SYMBOLS),
                "panel_symbols_loaded": len(panel),
                "primary_cost_bps_per_side": PRIMARY_COST_BPS,
                "stress_cost_bps_per_side": STRESS_COST_BPS,
                "observations": observations,
                "history_range": f"{ohlc['SPY'].index.min().date()} -> {ohlc['SPY'].index.max().date()}",
            },
            "fomc_calendar": fomc.as_dict(),
            "tests": [test.as_dict() for test in tests],
            "survivors": [test.name for test in survivors],
            "dropped": [
                {"name": test.name, "reason": test.as_dict()["drop_reason"]}
                for test in tests
                if not test.survives
            ],
            "sleeves": {name: sleeve for name, sleeve in sleeves.items()},
            "multiple_testing": multiple_testing,
            "diagnostics": diagnostics,
            "decision": _decide(tests, sleeves),
        }
        (output_dir / "event_premia_v1_run.json").write_text(
            json.dumps(_jsonable(result), indent=2), encoding="utf-8"
        )
        (output_dir / "event_premia_v1_summary.md").write_text(_summary_md(result), encoding="utf-8")
        _write_charts(output_dir, result, ohlc, fomc.announcements)
        return result

    def _e1_pre_fomc(
        self,
        ohlc: dict[str, pd.DataFrame],
        announcements: pd.DatetimeIndex,
        diagnostics: dict[str, object],
    ) -> list[EffectTest]:
        """E1: buy the close before an FOMC announcement, sell the announcement-day close."""
        tests: list[EffectTest] = []
        for symbol in INDEX_SYMBOLS:
            frame = ohlc.get(symbol)
            if frame is None:
                continue
            close = frame["close"]
            returns = close.pct_change()
            event_returns = {}
            for stamp in announcements:
                if stamp not in returns.index:
                    following = returns.index[returns.index >= stamp]
                    if len(following) == 0:
                        continue
                    stamp = following[0]
                value = returns.get(stamp)
                if value is not None and np.isfinite(value):
                    event_returns[stamp] = float(value)
            series = pd.Series(event_returns).sort_index()
            tests.append(
                test_effect(
                    f"E1_pre_fomc_{symbol}",
                    f"{symbol} return from the close before an FOMC announcement to the announcement close",
                    series,
                    round_trip_cost=2 * PRIMARY_COST_BPS / 10_000.0,
                    events_per_year=8.0,
                    extras={"events": len(series)},
                )
            )
            diagnostics[f"E1_car_{symbol}"] = car_profile(returns, announcements, pre=5, post=5)
        return tests

    def _e2_overnight_intraday(
        self,
        ohlc: dict[str, pd.DataFrame],
        panel: dict[str, pd.DataFrame],
        diagnostics: dict[str, object],
    ) -> list[EffectTest]:
        """E2: does the close-to-open leg carry the return that the open-to-close leg does not?"""
        tests: list[EffectTest] = []
        summary: dict[str, object] = {}
        targets: list[tuple[str, pd.Series, pd.Series]] = []
        for symbol in INDEX_SYMBOLS:
            frame = ohlc.get(symbol)
            if frame is None:
                continue
            overnight = frame["open"] / frame["close"].shift(1) - 1.0
            intraday = frame["close"] / frame["open"] - 1.0
            targets.append((symbol, overnight, intraday))
        if panel:
            overnight_panel = pd.concat(
                [(frame["open"] / frame["close"].shift(1) - 1.0).rename(symbol) for symbol, frame in panel.items()],
                axis=1,
            ).mean(axis=1)
            intraday_panel = pd.concat(
                [(frame["close"] / frame["open"] - 1.0).rename(symbol) for symbol, frame in panel.items()],
                axis=1,
            ).mean(axis=1)
            targets.append(("panel", overnight_panel, intraday_panel))

        for label, overnight, intraday in targets:
            overnight = overnight.dropna()
            intraday = intraday.dropna()
            tests.append(
                test_effect(
                    f"E2_overnight_{label}",
                    f"{label} close-to-open (overnight) daily return",
                    overnight,
                    round_trip_cost=2 * PRIMARY_COST_BPS / 10_000.0,
                    events_per_year=TRADING_DAYS,
                    extras={"leg": "overnight"},
                )
            )
            total_overnight = float((1.0 + overnight).prod() - 1.0)
            total_intraday = float((1.0 + intraday).prod() - 1.0)
            summary[label] = {
                "overnight_mean_bps": round(float(overnight.mean()) * 10_000, 3),
                "intraday_mean_bps": round(float(intraday.mean()) * 10_000, 3),
                "overnight_total_return": round(total_overnight, 4),
                "intraday_total_return": round(total_intraday, 4),
                "overnight_sharpe": round(float(overnight.mean() / overnight.std() * math.sqrt(TRADING_DAYS)), 3),
                "intraday_sharpe": round(float(intraday.mean() / intraday.std() * math.sqrt(TRADING_DAYS)), 3),
                "overnight_share_of_total": (
                    round(total_overnight / (total_overnight + total_intraday), 4)
                    if (total_overnight + total_intraday) != 0
                    else None
                ),
            }
        diagnostics["E2_overnight_vs_intraday"] = summary
        return tests

    def _e3_turn_of_month(
        self, ohlc: dict[str, pd.DataFrame], diagnostics: dict[str, object]
    ) -> list[EffectTest]:
        """E3: the last few and first few trading days of a month."""
        frame = ohlc["SPY"]
        returns = frame["close"].pct_change().dropna()
        tests: list[EffectTest] = []
        specs = (("base_last4_first3", 4, 3), ("wide_last5_first4", 5, 4))
        for label, last_n, first_n in specs:
            flags = turn_of_month_flags(returns.index, last_n=last_n, first_n=first_n)
            window_returns = _window_event_returns(returns, flags)
            tests.append(
                test_effect(
                    f"E3_turn_of_month_{label}",
                    f"SPY cumulative return over each turn-of-month window (last {last_n} + first {first_n})",
                    window_returns,
                    round_trip_cost=2 * PRIMARY_COST_BPS / 10_000.0,
                    events_per_year=12.0,
                    extras={"windows": len(window_returns), "time_in_market": round(float(flags.mean()), 4)},
                )
            )
            diagnostics[f"E3_{label}"] = {
                "mean_daily_in_window_bps": round(float(returns[flags].mean()) * 10_000, 3),
                "mean_daily_outside_window_bps": round(float(returns[~flags].mean()) * 10_000, 3),
            }
        return tests

    def _e4_opex_quarter_end(
        self, ohlc: dict[str, pd.DataFrame], diagnostics: dict[str, object]
    ) -> list[EffectTest]:
        """E4: OPEX week, OPEX day, and quarter-end effects."""
        frame = ohlc["SPY"]
        returns = frame["close"].pct_change().dropna()
        tests: list[EffectTest] = []

        week_flags = opex_week_flags(returns.index)
        tests.append(
            test_effect(
                "E4_opex_week",
                "SPY cumulative return over each monthly option-expiration week",
                _window_event_returns(returns, week_flags),
                round_trip_cost=2 * PRIMARY_COST_BPS / 10_000.0,
                events_per_year=12.0,
                extras={"time_in_market": round(float(week_flags.mean()), 4)},
            )
        )
        expirations = opex_dates(returns.index)
        tests.append(
            test_effect(
                "E4_opex_day",
                "SPY single-day return on monthly option expiration",
                returns.reindex(expirations).dropna(),
                round_trip_cost=2 * PRIMARY_COST_BPS / 10_000.0,
                events_per_year=12.0,
            )
        )
        quarter_ends = quarter_end_dates(returns.index)
        tests.append(
            test_effect(
                "E4_quarter_end_day",
                "SPY single-day return on the last trading day of each quarter",
                returns.reindex(quarter_ends).dropna(),
                round_trip_cost=2 * PRIMARY_COST_BPS / 10_000.0,
                events_per_year=4.0,
            )
        )
        diagnostics["E4_counts"] = {
            "opex_days": len(expirations),
            "quarter_ends": len(quarter_ends),
            "opex_week_day_fraction": round(float(week_flags.mean()), 4),
        }
        return tests

    def _e5_gap_pead(
        self, panel: dict[str, pd.DataFrame], diagnostics: dict[str, object]
    ) -> list[EffectTest]:
        """E5: post-announcement drift after a top-decile positive earnings gap."""
        if not panel:
            diagnostics["E5"] = {"status": "NOT_TESTED", "reason": "no panel loaded"}
            return []
        announcements = self._events_earnings_dates(list(panel), diagnostics)
        if announcements.empty:
            diagnostics["E5"] = {
                "status": "NOT_TESTED",
                "reason": "earnings dates unavailable; hypothesis needs a real announcement calendar",
            }
            return []

        records: list[dict[str, object]] = []
        for symbol, frame in panel.items():
            symbol_events = announcements[announcements["symbol"] == symbol]
            if symbol_events.empty:
                continue
            close = frame["close"]
            open_ = frame["open"]
            gap = (open_ / close.shift(1) - 1.0).dropna()
            forward_5 = close.shift(-5) / close - 1.0
            forward_20 = close.shift(-20) / close - 1.0
            for reaction_day in symbol_events["reaction_day"]:
                if reaction_day not in gap.index:
                    continue
                records.append(
                    {
                        "symbol": symbol,
                        "date": reaction_day,
                        "gap": float(gap.loc[reaction_day]),
                        "forward_5d": float(forward_5.loc[reaction_day])
                        if reaction_day in forward_5.index and np.isfinite(forward_5.loc[reaction_day])
                        else np.nan,
                        "forward_20d": float(forward_20.loc[reaction_day])
                        if reaction_day in forward_20.index and np.isfinite(forward_20.loc[reaction_day])
                        else np.nan,
                    }
                )
        events = pd.DataFrame(records)
        if events.empty:
            diagnostics["E5"] = {"status": "NOT_TESTED", "reason": "no matched announcement/price rows"}
            return []

        # Threshold must come from prior events only: a full-sample quantile would let the future
        # decide which of today's gaps counts as top-decile.
        events = events.sort_values("date").reset_index(drop=True)
        expanding_threshold = (
            events["gap"].shift(1).expanding(min_periods=200).quantile(0.90)
        )
        events["threshold"] = expanding_threshold
        top_decile = events[
            events["threshold"].notna() & (events["gap"] >= events["threshold"])
        ].copy()
        diagnostics["E5"] = {
            "status": "TESTED",
            "matched_events": int(len(events)),
            "top_decile_events": int(len(top_decile)),
            "threshold_rule": (
                "expanding 90th percentile of prior gaps only (min 200 prior events); a full-sample "
                "quantile would be look-ahead"
            ),
            "final_threshold": round(float(expanding_threshold.dropna().iloc[-1]), 5)
            if expanding_threshold.notna().any()
            else None,
            "mean_gap_in_top_decile": round(float(top_decile["gap"].mean()), 5) if not top_decile.empty else None,
            "decile_monotonicity": _decile_monotonicity(events),
            "universe_caveat": (
                "the panel is UNIVERSE_100, a currently-listed mega-cap set, so this inherits survivorship "
                "bias and must be re-run on the point-in-time panel before any deployment decision"
            ),
        }
        self._last_pead_events = top_decile
        tests: list[EffectTest] = []
        for horizon, column in (("5d", "forward_5d"), ("20d", "forward_20d")):
            series = (
                top_decile.dropna(subset=[column])
                .set_index("date")[column]
                .sort_index()
            )
            tests.append(
                test_effect(
                    f"E5_gap_pead_{horizon}",
                    f"forward {horizon} return after a top-decile positive earnings gap",
                    series,
                    round_trip_cost=2 * STRESS_COST_BPS / 10_000.0,
                    extras={"events": len(series), "horizon": horizon},
                )
            )
        return tests

    def _events_earnings_dates(
        self, symbols: list[str], diagnostics: dict[str, object]
    ) -> pd.DataFrame:
        """Earnings announcement dates mapped to the first trading reaction day."""
        import yfinance as yf

        cache = Path("cache/events/earnings_dates.parquet")
        if cache.exists():
            try:
                return pd.read_parquet(cache)
            except Exception:
                pass
        records: list[dict[str, object]] = []
        failures = 0
        for symbol in symbols:
            try:
                frame = yf.Ticker(symbol).get_earnings_dates(limit=100)
            except Exception:
                failures += 1
                continue
            if frame is None or not len(frame):
                failures += 1
                continue
            index = frame.index
            for stamp in index:
                naive = stamp.tz_localize(None) if stamp.tz is not None else stamp
                # Announcements at or after 16:00 ET are digested by the next session.
                reaction = naive.normalize() + (pd.Timedelta(days=1) if naive.hour >= 16 else pd.Timedelta(0))
                records.append({"symbol": symbol, "announced": naive, "reaction_day": reaction})
        out = pd.DataFrame(records)
        diagnostics["E5_earnings_source"] = {
            "symbols_requested": len(symbols),
            "symbols_with_dates": int(out["symbol"].nunique()) if not out.empty else 0,
            "failures": failures,
            "rows": int(len(out)),
            "range": (
                f"{out['announced'].min().date()} -> {out['announced'].max().date()}" if not out.empty else None
            ),
            "reaction_day_rule": "announcements at/after 16:00 local timestamp are assigned to the next day",
        }
        if not out.empty:
            cache.parent.mkdir(parents=True, exist_ok=True)
            try:
                out.to_parquet(cache)
            except Exception:
                pass
        return out

    def _e6_vix_spike(
        self,
        ohlc: dict[str, pd.DataFrame],
        vix_frame: pd.DataFrame,
        diagnostics: dict[str, object],
    ) -> list[EffectTest]:
        """E6: SPY forward returns after the VIX crosses a high rolling percentile."""
        if "vix" not in vix_frame:
            diagnostics["E6"] = {"status": "NOT_TESTED", "reason": "VIX unavailable"}
            return []
        close = ohlc["SPY"]["close"]
        forward_5 = (close.shift(-5) / close - 1.0).dropna()
        vix = vix_frame["vix"].reindex(close.index).ffill(limit=3)
        percentile = vix.rolling(504, min_periods=120).rank(pct=True)
        tests: list[EffectTest] = []
        counts: dict[str, object] = {}
        for label, level in (("p90", 0.90), ("p95", 0.95)):
            crossed = (percentile >= level) & (percentile.shift(1) < level)
            stamps = percentile.index[crossed.fillna(False)]
            series = forward_5.reindex(stamps).dropna()
            counts[label] = {"crossings": int(len(stamps)), "usable": int(len(series))}
            tests.append(
                test_effect(
                    f"E6_vix_spike_{label}",
                    f"SPY forward 5-day return after the VIX crosses its {label} rolling percentile",
                    series,
                    round_trip_cost=2 * PRIMARY_COST_BPS / 10_000.0,
                    extras={"threshold": level},
                )
            )
        diagnostics["E6"] = {"status": "TESTED", **counts}
        return tests

    def _survivor_sleeves(
        self,
        survivors: list[EffectTest],
        ohlc: dict[str, pd.DataFrame],
        vix_frame: pd.DataFrame,
        diagnostics: dict[str, object],
    ) -> dict[str, object]:
        """Turn each surviving hypothesis into a tradeable rule and cost it honestly."""
        sleeves: dict[str, object] = {}
        close = ohlc["SPY"]["close"]
        daily = close.pct_change().dropna()
        benchmark = daily
        for test in survivors:
            if test.name.startswith("E2_overnight"):
                symbol = test.name.rsplit("_", 1)[-1]
                frame = ohlc.get(symbol if symbol in ohlc else "SPY")
                if frame is None:
                    continue
                overnight = (frame["open"] / frame["close"].shift(1) - 1.0).dropna()
                for cost_label, cost_bps in (("primary", PRIMARY_COST_BPS), ("stress", STRESS_COST_BPS)):
                    net = overnight - 2 * cost_bps / 10_000.0
                    sleeves[f"{test.name}|{cost_label}"] = {
                        "rule": "buy at the close every day, sell at the next open, cash overnight otherwise",
                        "gross_metrics": _metrics(overnight, benchmark),
                        "net_metrics": _metrics(net, benchmark),
                        "round_trips_per_year": round(TRADING_DAYS, 1),
                        "cost_drag_annual": round(2 * cost_bps / 10_000.0 * TRADING_DAYS, 4),
                        "note": (
                            "one round trip every session; the cost drag is charged on the full notional "
                            "each day, which is what makes this leg hard to monetise at retail"
                        ),
                    }
            elif test.name.startswith("E3_turn_of_month"):
                label = test.name.replace("E3_turn_of_month_", "")
                last_n, first_n = (4, 3) if label == "base_last4_first3" else (5, 4)
                flags = turn_of_month_flags(daily.index, last_n=last_n, first_n=first_n)
                sleeves[test.name] = _flag_sleeve(daily, flags, PRIMARY_COST_BPS, benchmark)
            elif test.name.startswith("E1_pre_fomc"):
                sleeves[test.name] = {
                    "rule": "hold the index only from the pre-announcement close to the announcement close",
                    "note": "sleeve backtest only built for survivors that pass split-sample confirmation",
                }
            elif test.name.startswith("E5_gap_pead"):
                horizon = 5 if test.name.endswith("5d") else 20
                events = getattr(self, "_last_pead_events", None)
                if events is None or events.empty:
                    continue
                for cost_label, cost_bps in (("primary", PRIMARY_COST_BPS), ("stress", STRESS_COST_BPS)):
                    sleeves[f"{test.name}|{cost_label}"] = _pead_sleeve(
                        events=events,
                        panel=self._panel_cache,
                        horizon=horizon,
                        cost_bps=cost_bps,
                        benchmark=benchmark,
                    )
            elif test.name.startswith("E6_vix_spike"):
                sleeves[test.name] = {
                    "rule": "buy SPY on a VIX percentile crossing, hold 5 trading days",
                    "note": "evaluated as a re-entry timing overlay rather than a standalone sleeve",
                }
        diagnostics["sleeve_count"] = len(sleeves)
        return sleeves


def _window_event_returns(returns: pd.Series, flags: pd.Series) -> pd.Series:
    """Cumulative return of each contiguous run of flagged days, stamped at the run's start."""
    out: dict[pd.Timestamp, float] = {}
    active = False
    start: pd.Timestamp | None = None
    compounded = 1.0
    for stamp, flag in flags.items():
        if flag and not active:
            active, start, compounded = True, stamp, 1.0
        if flag:
            compounded *= 1.0 + float(returns.get(stamp, 0.0))
        if not flag and active:
            out[start] = compounded - 1.0
            active = False
    if active and start is not None:
        out[start] = compounded - 1.0
    return pd.Series(out).sort_index()


def _pead_sleeve(
    *,
    events: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    horizon: int,
    cost_bps: float,
    benchmark: pd.Series | None = None,
) -> dict[str, object]:
    """Equal-weighted book of post-gap names held `horizon` days, costed on realised turnover.

    Entries are stamped at the reaction-day close (the gap is known at that day's open, so nothing
    from the future is used) and each name is held for `horizon` trading sessions.
    """
    closes = pd.DataFrame({symbol: frame["close"] for symbol, frame in panel.items()}).sort_index()
    returns = closes.pct_change()
    calendar = closes.index
    position_of = {stamp: index for index, stamp in enumerate(calendar)}

    holdings = pd.DataFrame(0.0, index=calendar, columns=closes.columns)
    for row in events.itertuples(index=False):
        symbol = getattr(row, "symbol")
        entry = getattr(row, "date")
        if symbol not in holdings.columns or entry not in position_of:
            continue
        start = position_of[entry] + 1
        end = min(start + horizon, len(calendar))
        if start >= len(calendar):
            continue
        holdings.iloc[start:end, holdings.columns.get_loc(symbol)] = 1.0

    active = holdings.sum(axis=1)
    weights = holdings.div(active.where(active > 0), axis=0).fillna(0.0)
    gross = (weights * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    net = gross - turnover * (cost_bps / 10_000.0)
    invested = active > 0
    return {
        "rule": (
            f"buy every name whose earnings gap exceeded the expanding 90th percentile, at the reaction-day "
            f"close, equal-weighted, hold {horizon} sessions"
        ),
        "gross_metrics": _metrics(gross, benchmark),
        "net_metrics": _metrics(net, benchmark),
        "time_in_market": round(float(invested.mean()), 4),
        "mean_positions_when_invested": round(float(active[invested].mean()), 2) if invested.any() else 0.0,
        "mean_daily_turnover": round(float(turnover.mean()), 4),
        "entries": int(len(events)),
        "cost_bps_per_side": cost_bps,
    }


def _flag_sleeve(
    daily: pd.Series, flags: pd.Series, cost_bps: float, benchmark: pd.Series | None = None
) -> dict[str, object]:
    """Hold the asset only on flagged days; charge a round trip on every entry/exit."""
    aligned = flags.reindex(daily.index).astype(bool)
    # Decide at T close for T+1: shift the flag so no same-day information is used.
    position = aligned.shift(1, fill_value=False).astype(float)
    changes = position.diff().abs().fillna(position.abs())
    net = daily * position - changes * (cost_bps / 10_000.0)
    return {
        "rule": "hold only on flagged days, cash otherwise, position set from the prior close",
        "gross_metrics": _metrics(daily * position, benchmark),
        "net_metrics": _metrics(net, benchmark),
        "time_in_market": round(float(position.mean()), 4),
        "round_trips": int(changes.sum() / 2),
    }


def _metrics(returns: pd.Series, benchmark: pd.Series | None = None) -> dict[str, float]:
    series = returns.dropna()
    if series.empty:
        return {key: 0.0 for key in ("total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "days")}
    equity = (1.0 + series).cumprod()
    total = float(equity.iloc[-1]) - 1.0
    volatility = float(series.std(ddof=0))
    out = {
        "total_return": round(total, 6),
        "cagr": round((1.0 + total) ** (TRADING_DAYS / len(series)) - 1.0 if total > -1 else -1.0, 6),
        "ann_vol": round(volatility * math.sqrt(TRADING_DAYS), 6),
        "sharpe": round(float(series.mean()) / volatility * math.sqrt(TRADING_DAYS), 4) if volatility > 0 else 0.0,
        "max_drawdown": round(float((equity / equity.cummax() - 1.0).min()), 6),
        "days": int(len(series)),
    }
    if benchmark is not None:
        joined = pd.concat({"strategy": series, "benchmark": benchmark}, axis=1).dropna()
        if len(joined) > 30 and joined["benchmark"].var() > 0:
            beta = float(joined["strategy"].cov(joined["benchmark"]) / joined["benchmark"].var())
            out["beta_vs_spy"] = round(beta, 4)
            out["annualized_alpha_vs_spy"] = round(
                float(joined["strategy"].mean() - beta * joined["benchmark"].mean()) * TRADING_DAYS, 6
            )
    return out


def _decile_monotonicity(events: pd.DataFrame) -> dict[str, object]:
    frame = events.dropna(subset=["gap", "forward_5d"]).copy()
    if len(frame) < 100:
        return {"testable": False}
    frame["decile"] = pd.qcut(frame["gap"], 10, labels=False, duplicates="drop")
    means = frame.groupby("decile")["forward_5d"].mean()
    increasing = bool(means.is_monotonic_increasing)
    spread = float(means.iloc[-1] - means.iloc[0]) if len(means) > 1 else 0.0
    return {
        "testable": True,
        "decile_mean_forward_5d": {str(int(k)): round(float(v), 6) for k, v in means.items()},
        "monotonically_increasing": increasing,
        "top_minus_bottom": round(spread, 6),
    }


def _decide(tests: list[EffectTest], sleeves: dict[str, object]) -> dict[str, object]:
    survivors = [test for test in tests if test.survives]
    promoted: list[str] = []
    watchlist: list[str] = []
    for name, sleeve in sleeves.items():
        metrics = sleeve.get("net_metrics") if isinstance(sleeve, dict) else None
        if not metrics:
            watchlist.append(name)
            continue
        if float(metrics["sharpe"]) >= 0.8 and abs(float(metrics["max_drawdown"])) <= 0.10:
            promoted.append(name)
        else:
            watchlist.append(name)
    return {
        "tested": len(tests),
        "survived_confirmation": [test.name for test in survivors],
        "dropped_permanently": [test.name for test in tests if not test.survives],
        "sleeves_meeting_standalone_gate": promoted,
        "sleeves_watchlist": watchlist,
        "status": "PROMOTED" if promoted else "WATCHLIST" if survivors else "KILLED",
        "note": (
            "surviving an effect test is necessary but not sufficient: the sleeve must also clear the "
            "standalone Sharpe/drawdown gate net of costs"
        ),
    }


def _summary_md(result: dict[str, object]) -> str:
    config = result["config"]
    decision = result["decision"]
    lines = [
        "# EVENT-PREMIA-V1 (Direction D2)",
        "",
        f"- thesis: `{result['thesis']}`",
        f"- history: `{config['history_range']}` (`{config['observations']}` SPY sessions)",
        f"- declared variants: `{result['pre_registration']['declared_variants']}`, "
        f"realized: `{result['pre_registration']['realized_variants']}`",
        f"- confirmation rule: `{result['pre_registration']['confirmation_rule']}`",
        f"- decision: **`{decision['status']}`**",
        "",
        "## Hypothesis Results",
        "",
        "| Hypothesis | Events | Mean effect | Early CI excl. 0 | Late CI excl. 0 | > 2x cost | Verdict |",
        "|---|---:|---:|:---:|:---:|:---:|---|",
    ]
    for test in result["tests"]:
        mean_bps = test["full"]["mean"] * 10_000
        lines.append(
            f"| `{test['name']}` | `{test['full']['n']}` | `{mean_bps:.1f}` bps | "
            f"{'yes' if test['early']['excludes_zero'] else 'no'} | "
            f"{'yes' if test['late']['excludes_zero'] else 'no'} | "
            f"{'yes' if test['exceeds_cost_hurdle'] else 'no'} | "
            f"**{test['verdict']}** |"
        )

    overnight = result["diagnostics"].get("E2_overnight_vs_intraday", {})
    if overnight:
        lines += [
            "",
            "## E2: Overnight vs Intraday Decomposition",
            "",
            "| Series | Overnight mean | Intraday mean | Overnight total | Intraday total | Overnight Sharpe | Intraday Sharpe |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for label, row in overnight.items():
            lines.append(
                f"| `{label}` | `{row['overnight_mean_bps']:.2f}` bps | `{row['intraday_mean_bps']:.2f}` bps | "
                f"`{row['overnight_total_return']:.1%}` | `{row['intraday_total_return']:.1%}` | "
                f"`{row['overnight_sharpe']:.2f}` | `{row['intraday_sharpe']:.2f}` |"
            )

    lines += ["", "## Dropped Hypotheses", "", "| Hypothesis | Reason |", "|---|---|"]
    for item in result["dropped"]:
        lines.append(f"| `{item['name']}` | {item['reason']} |")

    sleeves = result["sleeves"]
    if sleeves:
        lines += [
            "",
            "## Survivor Sleeves (net of costs)",
            "",
            "| Sleeve | Gross Sharpe | Net Sharpe | Net CAGR | Worst DD | Time in market |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, sleeve in sleeves.items():
            if not isinstance(sleeve, dict) or "net_metrics" not in sleeve:
                continue
            gross = sleeve["gross_metrics"]
            net = sleeve["net_metrics"]
            lines.append(
                f"| `{name}` | `{gross['sharpe']:.2f}` | `{net['sharpe']:.2f}` | `{net['cagr']:.2%}` | "
                f"`{net['max_drawdown']:.1%}` | "
                f"`{sleeve.get('time_in_market', 1.0):.0%}` |"
            )

    fomc = result["fomc_calendar"]
    multiple_testing = result["multiple_testing"]
    lines += [
        "",
        "## Data and Multiple Testing",
        "",
        f"- FOMC announcements scraped from federalreserve.gov: `{fomc['announcements']}` "
        f"(`{fomc['first']}` -> `{fomc['last']}`)",
        f"- years whose statement count differs from 8 scheduled meetings: `{fomc['years_with_unexpected_counts']}` "
        f"({fomc['unexpected_count_note']})",
        f"- with `{multiple_testing['trials']}` pre-registered variants over `{multiple_testing['observations']}` "
        f"sessions, best-of-N selection alone is expected to produce an annualized Sharpe of about "
        f"`{multiple_testing['expected_max_sharpe_under_null']}` under a zero-edge null",
        "",
        "## Decision",
        "",
        f"- survived confirmation: `{decision['survived_confirmation']}`",
        f"- dropped permanently: `{decision['dropped_permanently']}`",
        f"- sleeves meeting the standalone gate: `{decision['sleeves_meeting_standalone_gate']}`",
        f"- watchlist: `{decision['sleeves_watchlist']}`",
        "",
        decision["note"],
        "",
    ]
    return "\n".join(lines)


def _write_charts(
    output_dir: Path,
    result: dict[str, object],
    ohlc: dict[str, pd.DataFrame],
    announcements: pd.DatetimeIndex,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    overnight = result["diagnostics"].get("E2_overnight_vs_intraday", {})
    if overnight and "SPY" in ohlc:
        frame = ohlc["SPY"]
        overnight_series = (frame["open"] / frame["close"].shift(1) - 1.0).dropna()
        intraday_series = (frame["close"] / frame["open"] - 1.0).dropna()
        figure, axis = plt.subplots(figsize=(12, 5.5))
        axis.plot((1 + overnight_series).cumprod().index, (1 + overnight_series).cumprod().to_numpy(),
                  label="overnight (close->open)", color="#4C72B0")
        axis.plot((1 + intraday_series).cumprod().index, (1 + intraday_series).cumprod().to_numpy(),
                  label="intraday (open->close)", color="#C44E52")
        buy_hold = (1 + frame["close"].pct_change().dropna()).cumprod()
        axis.plot(buy_hold.index, buy_hold.to_numpy(), label="buy and hold", color="#55A868", alpha=0.8)
        axis.set_yscale("log")
        axis.set_title("EVENT-PREMIA-V1 E2: SPY return split into overnight and intraday legs (gross)")
        axis.legend()
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_dir / "event_premia_overnight_vs_intraday.png", dpi=130)
        plt.close(figure)

    car = result["diagnostics"].get("E1_car_SPY")
    if car:
        offsets = [int(key) for key in car["cumulative_return_by_offset"]]
        values = [car["cumulative_return_by_offset"][str(key)] * 100 for key in offsets]
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.plot(offsets, values, marker="o", color="#4C72B0")
        axis.axvline(0, color="black", linestyle="--", linewidth=0.9, label="announcement day")
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_xlabel("Trading days from FOMC announcement")
        axis.set_ylabel("Cumulative mean return (%)")
        axis.set_title(f"E1: SPY cumulative return around {car['events_used']} FOMC announcements")
        axis.legend()
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_dir / "event_premia_fomc_car.png", dpi=130)
        plt.close(figure)
    del announcements


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
