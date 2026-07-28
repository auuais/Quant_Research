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
| D2 | Event & seasonality premia battery | `PENDING` | — | — | — |
| D3 | Automated alpha factory | `PENDING` | — | — | — |
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

## How to reproduce

```bash
python -m algoding.cli vol-carry-research
```
