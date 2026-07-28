# Frontier Board V1

Status of the [new-directions plan](../../NEW_RESEARCH_DIRECTIONS_PLAN.md) sleeves. Separate from
[research_winner_board_v1.md](research_winner_board_v1.md) on purpose: that board ranks challengers for the
**equity factor book** (relative, +0.05 Sharpe rule), while these are candidate **standalone return streams**
judged on absolute gates and correlation to the core book.

Every entry is pre-registered in [frontier_hypotheses.md](frontier_hypotheses.md) before it runs.

## Current status

| Direction | Sleeve | Status | Best net Sharpe | Worst DD | Verdict |
|---|---|---|---:|---:|---|
| D1 | Volatility risk premium / VIX term structure | `WATCHLIST` | 0.51 (full history) | −61.7% | Premium real, risk profile fails the gate; SPY beats it on Sharpe |
| D2 | Event & seasonality premia battery | `WATCHLIST` | 0.69 (gap-PEAD 5d) | −31.1% | 5 of 14 effects survived; gap-PEAD gives 10.5% alpha at beta 0.52 |
| D3 | Automated alpha factory | `WATCHLIST` (paused) | 1.90 (mined ensemble, test) | −6.8% | Loses to incumbent by 0.65 Sharpe; search proved chaotic w.r.t. trivial data noise |
| D4 | Futures carry + trend (CTA) | `NOT STARTED` | — | — | Wave 2 |
| D5 | Statistical arbitrage / residual reversal | `NOT STARTED` | — | — | Wave 2 |
| D6 | Disclosure change & insider NLP | `NOT STARTED` | — | — | Wave 2 |
| D7 | Options (defined risk) | `BLOCKED` | — | — | Needs STRATEGY.md exclusion amendment |
| D8 | Crypto delta-neutral carry | `NOT STARTED` | — | — | Wave 3 |
| D9 | TSFM / cross-sectional ML benchmark | `BLOCKED` | — | — | Needs point-in-time panel (PMP P2-1) |

No frontier sleeve is promoted. `resid_mom_60d` remains the only research winner.

## D1 — Volatility risk premium (run 2026-07-28)

Full detail: [vol_carry_v1/vol_carry_v1_summary.md](vol_carry_v1/vol_carry_v1_summary.md).

The premium is real, and gating is what makes it survivable rather than ruinous:

| Strategy | CAGR | Ann vol | Sharpe | Worst DD |
|---|---:|---:|---:|---:|
| `slope_02_vrp` (best gated) | 12.56% | 38.9% | 0.51 | −61.7% |
| `baseline_always_short_vol` | −0.28% | 53.4% | 0.36 | −95.2% |
| `baseline_spy` | 13.88% | 17.1% | 0.85 | −33.7% |

Static short-vol harvesting returned **−3.5% in total over 12.7 years** with a −95.2% drawdown. The
term-structure gate turns that into ~12.5% CAGR and cuts crisis losses hard (Volmageddon −90.4% → −39.2%,
COVID −55.3% → −17.6%). But SPY beat every variant on Sharpe with a third of the volatility, so the sleeve
earns no capital allocation on these numbers.

**Reusable finding for all future volatility work:** VIX cash indices settle at 16:15 ET while the ETPs trade
the 16:00 close. Deciding on the same close embeds up to 15 minutes of look-ahead, and Feb-2018's damage
happened inside that window. At lag 1 the long-vol variants showed **+32.0%** through Volmageddon; at lag 2
they show **−32.7%**. About three-quarters of the apparent crisis protection was a settlement artifact.
Anything trading the vol complex here must use `HEADLINE_SIGNAL_LAG = 2` or intraday data.

Two further methodology corrections came out of this run and apply repo-wide:

1. **Synthetic leverage rescaling is unsafe in crises.** A −1x series rebuilt from VXX loses 48.7% over
   Volmageddon where the real −1x product lost 90.4%. Use real instrument closes.
2. **Tail-anchored validation windows hide crises.** The inherited `_validation_windows` helper takes only the
   last N windows; on a 12.7-year sample that covered the final 630 days and excluded every pre-2024 crisis,
   flattering always-short-vol from a true 0.36 Sharpe to 0.75. Frontier runs tile windows across the whole
   sample.

## D2 — Event & seasonality premia battery (run 2026-07-28)

Full detail: [event_premia_v1/event_premia_v1_summary.md](event_premia_v1/event_premia_v1_summary.md).

5 of 14 pre-registered variants survived split-sample confirmation. No sleeve cleared the standalone gate,
so nothing is promoted, but one result is worth real follow-up:

| Sleeve | Net Sharpe | Net CAGR | Worst DD | Beta | Ann. alpha |
|---|---:|---:|---:|---:|---:|
| `E5_gap_pead_5d` (1 bps/side) | 0.69 | 14.9% | −31.1% | 0.52 | **+10.5%** |
| `E5_gap_pead_20d` (1 bps/side) | 0.77 | 18.8% | −51.1% | 0.98 | +9.0% |
| `E2_overnight_QQQ` (1 bps/side) | 0.53 | 6.0% | −30.8% | 0.38 | +2.0% |
| `E2_overnight_QQQ` (3 bps/side) | −0.28 | −4.2% | −67.0% | 0.38 | −8.1% |
| `E3_turn_of_month_base` | 0.47 | 4.5% | −33.6% | 0.31 | +1.2% |

**Gap-PEAD (5-day) is the best frontier finding so far**: ~10.5% annualized alpha at beta 0.52, and nearly
cost-insensitive because the holding period amortizes the round trip. It fails the gate only on drawdown.
It runs on the survivorship-biased `UNIVERSE_100`, so re-running it on the point-in-time panel (PMP P2-1) is
the single highest-value frontier follow-up.

**The pre-FOMC drift did not confirm** in either index ETF (CI includes zero in a subperiod) and is dropped
permanently, contradicting the literature that motivated the direction.

**The overnight effect is real but not tradeable at retail cost.** SPY's overnight leg compounded to +435%
against +68% intraday, yet a daily round trip turns QQQ's gross 0.93 Sharpe into net −0.28 at 3 bps/side. Its
value is an execution insight — prefer closing auctions for entries — not a sleeve.

## D3 — Automated alpha factory (run 2026-07-28)

Full detail: [alpha_factory_v1/alpha_factory_v1_summary.md](alpha_factory_v1/alpha_factory_v1_summary.md).

4,000 candidate evaluations over a 2005-2026 / 100-symbol panel. 482 factors cleared the train thresholds and
**one** cleared validation. Its IC held out of sample (validation +0.0249 → test +0.0190, a 23.6% decay, below
the 50% kill threshold), so the pre-registered kill rule did not fire. The book is what fails:

| Book (test period, top-20 long only, net 1 bps/side) | CAGR | Sharpe | Worst DD | Beta |
|---|---:|---:|---:|---:|
| `baseline_resid_mom_60d` (incumbent) | 62.9% | **2.545** | −7.2% | 1.11 |
| `baseline_ret_20d` | 48.3% | 2.306 | −7.6% | 1.06 |
| `ensemble_rank_average` (mined) | 26.5% | 1.898 | −6.8% | 0.92 |

Not promoted — it loses to the incumbent by 0.65 Sharpe. A useful reminder that IC surviving out of sample is
necessary but not sufficient: this factor's IC held and its book still lost badly.

**The main finding is about the method, not the factor.** A cache bug (see below) meant early runs each
re-downloaded the panel, and Yahoo returns marginally different adjusted prices per fetch. Three runs of the
*same* nominal configuration:

| Run | Panel | Train-passers | Accepted | Status |
|---|---|---:|---:|---|
| 1 (pre-fix) | fresh download | 316 | 0 | would have read `KILLED` |
| 2 (pre-fix) | fresh download | 212 | 1 | `WATCHLIST` |
| 3 (post-fix, authoritative) | cached, stable | 482 | 1 | `WATCHLIST` |

Panel totals differed only in the third decimal (44779147.757 vs .759), yet the verdict moved. **Evolutionary
factor search over this panel is chaotic with respect to negligible data perturbations**, so no single run's
factor list is a result. `D3-b` must report stability across seeds and data vintages.

Second caveat: both incumbents were *negative* over the validation year (`resid_mom_60d` −0.0088, `ret_20d`
−0.0384) before the incumbent recovered to +0.0326 on test — a regime-specific window, so a single contiguous
validation year cannot adjudicate factor quality. `D3-b` needs purged multi-fold walk-forward validation, and
should wait for the point-in-time panel, since mining 21 years of today's mega-caps is a survivorship-bias
machine (the 62.9% incumbent CAGR above is inflated for exactly that reason).

The durable deliverable is the infrastructure: a factor DSL with a hand-written parser (no `eval`),
evolutionary search, an IC harness with correct warmup handling, and the single-test protocol. It re-runs
unchanged on a new panel.

## Reproducibility bug found and fixed (2026-07-28)

Worth its own entry because it invalidates naive re-runs of any panel research in this repo before this date.
The OHLC cache key joined all 100 tickers into the filename (~600 chars), exceeding the Windows path limit;
`to_parquet` raised and a bare `except: pass` discarded the error, so the cache never persisted and every run
re-downloaded from Yahoo. Fixed with a hashed key, and cache failures are now recorded rather than swallowed.

Verified with a controlled 300-candidate search run twice in separate processes:

| | Process A | Process B |
|---|---|---|
| Before the fix | 279 scored, 8 train-passers | 282 scored, 6 train-passers |
| After the fix | 283 scored, 8 train-passers, best IC 0.022514 | **identical** |

D2 was unaffected — re-run on the stable panel it reproduces exactly (same 5 survivors, same 9 drops, identical
sleeve metrics). Event studies over thousands of events are insensitive to third-decimal price noise; an
evolutionary search is not. That contrast is the transferable lesson: **the more adaptive the method, the more
it needs a frozen dataset.**

## How to reproduce

```bash
python -m algoding.cli vol-carry-research
```

```bash
python -m algoding.cli event-premia-research
```

```bash
python -m algoding.cli alpha-factory-run
```

## What Wave 1 established

No frontier sleeve is promoted, which is the expected base rate for nine speculative directions. The
transferable output is four corrections and one lead:

1. **VIX-complex work must lag its signal** (16:15 vs 16:00 settlement) — worth +32% vs −32% on a single
   crisis slice, i.e. the difference between a fake crisis hedge and a real one.
2. **Validation windows must tile the whole sample**, or crises fall outside every window.
3. **Never rescale leverage synthetically** through a crisis; hold the real instrument's returns.
4. **A single contiguous validation year cannot adjudicate factor quality** — check whether your incumbents
   also fail that window before concluding anything about challengers.
5. **The lead:** gap-PEAD, ~10.5% annualized alpha at beta 0.52 and cost-insensitive, pending a
   point-in-time-panel re-run.

The point-in-time panel (PMP P2-1) is now the binding dependency for the two most promising threads
(gap-PEAD magnitude, `D3-b` alpha mining), which raises its priority in the exploit plan.
