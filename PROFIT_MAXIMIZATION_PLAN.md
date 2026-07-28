# Profit Maximization Plan

Last updated: 2026-07-02
Status: `PROPOSED`
Owner docs: [STRATEGY.md](STRATEGY.md), [TRACKER.md](TRACKER.md), [RESEARCH_OUTCOMES.md](RESEARCH_OUTCOMES.md), [research_winner_board_v1.md](reports/research/research_winner_board_v1.md)

## 0. Purpose

This plan converts three years of research output into a concrete path to maximum *realized net profit*, subject to the project's survival-first constraints. It is deliberately opinionated: it says what to deploy, what to stop spending effort on, what to validate next, and in what order.

The single most important conclusion from the research program so far:

> **The edge found to date is simple price/factor structure (residual momentum and 20-day price momentum) executed cheaply at daily horizon. LLM/news/embedding branches have produced beta and timing overlays, not alpha. Profit maximization now depends on (a) deploying the validated winner, (b) hardening it against the known validity threats, and (c) acquiring new data classes — not on more model complexity over the same price+news data.**

## 1. Where profit actually is today (evidence inventory)

### 1.1 Validated winners (cost-aware, walk-forward, multi-window)

| Rank | Strategy | Evidence | Mean ann | Sharpe | Worst DD | Notes |
|---|---|---|---:|---:|---:|---|
| 1 | `resid_mom_60d` long-only top-20, 5-day rebalance | FACTOR-VALIDATION-V1 | 52.92% | 2.69 | -17.59% | Official winner. Survives 3 bps/side stress (52.22%, 2.66). Beta ~0.85. |
| 2 | `price_momentum_long_only_sign` (ret_20d) | PRICE-MOMENTUM-VALIDATION-V1 | 53.62% | 2.74 | -14.25% | Near-tie; higher turnover (0.77 vs 0.46). Missed promotion by +0.04 Sharpe vs required +0.05. |
| 3 | Commodity top-20 price momentum | CROSS-ASSET-MOMENTUM-V1 | 23.50% CAGR | 0.92 | -20.09% | Best standalone cross-asset sleeve. |
| — | `drawdown_governor` overlay | PORTFOLIO-CONTROLS-V1 | 51.86% | 2.61 | -12.70% | Best near-return-preserving risk overlay; use for drawdown-constrained deployment. |

### 1.2 Confirmed dead ends (stop allocating research time)

These are *negative results* that should now be treated as settled unless new data arrives:

1. **News/LLM sentiment as standalone alpha.** V3-5 DeepSeek is a structural permabull (94.3% bullish / 0.0% bearish over 20,559 bundles); every apparent edge was long exposure in a bull market. It cannot support a short book or market-neutral construction.
2. **Qwen-embedding directional classifiers.** Balanced accuracy ~33% (random for 3 classes); flips between degenerate "always bullish" / "always neutral" modes under small config changes.
3. **Hybrid price+news cross-sectional ML (HYBRID-V1).** OOS balanced accuracy ~0.35, decile spreads ~0 and non-monotonic; trained models *underperform random* market-neutral; adding news made the hybrid worse.
4. **Market-neutral long/short from news signals.** Rank-forced books are genuinely beta-neutral but ~flat net (Sharpe 0.29 best case) and crushed by SPY (1.75) and equal-weight (2.55).
5. **Crypto DL and extended-commodity daily DL.** Effectively flat after fees.
6. **Learned regime switching (REGIME-SWITCH-V1)** and **static construction controls as return improvers (PORTFOLIO-CONTROLS-V1).** Nothing beats the raw factor book on the promotion criteria.

Corollary: news/LLM work continues **only** as (a) a timing overlay watchlist (`factor_plus_ensemble_balanced_50`, `factor_plus_embed_ensemble_veto_boost`) and (b) a candidate for retraining on cross-sectional labels — both gated behind the Phase 2/3 items below.

### 1.3 Known threats to the headline numbers (must be neutralized before scaling)

The 52-54% annualized figures are **not** deployable expectations. Each threat below can materially deflate them:

| Threat | Description | Neutralization |
|---|---|---|
| Survivorship bias | 100-name universe is currently-listed Alpaca names, ranked partly by hindsight liquidity/strength | Point-in-time universe rebuild (Phase 2, item P2-1) |
| Single regime | Validation window 2022-09 → 2026-05 is dominated by one bull market; beta ~0.85 means most of the return is market | Long-history rerun + bear-slice stress (P2-2) |
| Cost realism | 1 bps/side is optimistic for a retail account at size; IEX feed under-reports volume ×30 and interacts with the participation cap | Use `volume_scale` fix + SEC §31 fee + MOC execution assumptions everywhere (P2-3) |
| Stop executability | Tight (1%) trailing stops are not executable as modeled; Alpaca does not guarantee stop prices | Standardize on close-to-close (MOC) evaluation; no intrabar stop assumptions without minute bars |
| Backtest/live gap | No shadow or paper evidence yet for the factor book (only the discredited V3-5 sleeve is on paper) | Shadow → paper pipeline (Phase 1) |

## 2. Plan overview

```
Phase 1 (weeks 1-4):    Deploy the winner to shadow, then paper. Retire the V3-5 paper sleeve.
Phase 2 (weeks 2-8):    Kill the validity threats (survivorship, regime, costs). Re-baseline.
Phase 3 (weeks 4-12):   Alpha expansion: momentum blend, new data classes, satellite sleeves.
Phase 4 (weeks 8-16):   Multi-sleeve portfolio construction and capital allocation.
Phase 5 (month 4+):     Tiny live deployment, then scale on evidence.
Ongoing:                Governance, cadence, and hard risk rules.
```

Phases overlap intentionally: Phase 1 (deployment plumbing) and Phase 2 (validity hardening) run in parallel because neither blocks the other, and both block real money.

## 3. Phase 1 — Deploy the validated winner to paper (weeks 1-4)

Goal: get `resid_mom_60d` long-only top-20 producing daily live-signal evidence. Every week it is not in shadow/paper is a week of unpurchasable validation data lost.

- **P1-1. Productionize the factor signal.** Extract the exact `resid_mom_60d` ranking + top-20 + 5-day-rebalance logic from `src/algoding/execution/factor_validation_research.py` into a deterministic, versioned signal in `src/algoding/signals/` (same feature-at-T-close, trade-at-T-close, earn-T+1 timing as the backtest). Freeze parameters: 60-day residual-momentum lookback, book size 20, rebalance every 5 trading days.
- **P1-2. Wire into shadow execution.** Run it through the existing shadow path (`shadow-run`) with null-order sink for ≥2 weeks. Log signal timestamps, intended orders, and data-staleness events.
- **P1-3. Promote to broker paper.** Use the existing promotion machinery (`promote-candidate`, `promoted-run`) to run the factor book as the primary paper candidate on the Alpaca paper account, MOC-style orders near the close.
- **P1-4. Retire/downgrade the V3-5 paper sleeve.** The research basis for the TSLA/AVGO/NVDA V3-5 sleeve is discredited (permabull artifact + non-execution-aware returns). Keep it running only if its weekly report is explicitly labeled "invalidated baseline, monitoring only"; otherwise stop it to remove noise from attribution.
- **P1-5. Deploy the drawdown-constrained variant in a parallel internal ledger.** Run `drawdown_governor` as a second internal book (not broker paper) so live-signal evidence accumulates for the risk overlay too. Decision rule at go-live: raw book if max-DD tolerance ≥ 20%, governor book if ≤ 13%.
- **P1-6. Daily reconciliation + alerting.** Broker/ledger reconciliation daily, heartbeat alerts, kill switch on stale data — per the STRATEGY.md control list. No new code paths should ship without these.

Exit criteria: 2+ weeks clean shadow, then 8+ weeks paper with fill prices within ~5 bps of modeled MOC assumptions, and paper book tracking the internal backtest-projected book within noise.

## 4. Phase 2 — Neutralize validity threats and re-baseline (weeks 2-8)

Goal: know the *deployable* expectation for the winner, not the optimistic one. Every item produces a re-baselined winner-board entry.

- **P2-1. Survivorship-aware universe rerun.** Rebuild the 100-name universe point-in-time (e.g., top-N by dollar volume as of each rebalance date from a long-history provider; the Yahoo fallback already used for the 15y DL runs can seed this, or add a survivorship-aware source). Rerun FACTOR-VALIDATION-V1 and PRICE-MOMENTUM-VALIDATION-V1 on it. **This is the highest-priority research item in the whole plan** — if the factor edge halves here, everything downstream re-prices.
- **P2-2. Long-history / bear-regime stress.** Extend the factor backtest to 10-15 years of daily data (Yahoo fallback path exists). Report per-regime slices: 2015-16 chop, 2018 Q4, 2020 crash, 2022 bear, 2023-26 bull. A beta-0.85 long book will show large drawdowns in 2022 — quantify them so position sizing at go-live is honest.
- **P2-3. One canonical cost model.** Standardize every backtest and the winner board on the corrected execution model: `volume_scale` ×30 for IEX volume, SEC §31 + FINRA TAF fees, MOC (close-decide/close-fill) execution, 1 bps primary / 3 bps stress. Add a regression test so no research script silently reports gross returns as net again (the V3-5 failure mode).
- **P2-4. Capacity check.** At what book size does the strategy's participation in the close begin to move fills? For a retail account this is likely a non-issue below ~$1-5M, but document it once so scaling decisions later are mechanical.
- **P2-5. Re-baseline the winner board.** Publish `research_winner_board_v2.md` with survivorship-aware, regime-sliced, canonical-cost numbers. All promotion decisions from then on reference V2.

Expected outcome (set expectations now): mean annualized on the corrected basis likely lands well below the current 52.92% — the STRATEGY.md realistic bands (net 8-15% at scale, Sharpe 0.7-1.2 after real-world drag) remain the planning base case; anything better is upside.

## 5. Phase 3 — Alpha expansion (weeks 4-12)

Goal: raise the ceiling. Ordered by expected profit-per-effort.

- **P3-1. Momentum blend (cheapest upside).** Test `resid_mom_60d + ret_20d` blended rank (50/50 and rank-average variants). The two winners are near-tied with different drawdown profiles and moderately different turnover; a blend plausibly diversifies timing. This is already the winner board's stated next step. Promotion hurdle: +0.05 Sharpe over the V2 baseline at equal or better drawdown.
- **P3-2. New data classes (the real lever).** HYBRID-V1 proved the feature matrix's fundamental/positioning columns are all NaN placeholders — that is where cross-sectional equity alpha lives. Acquisition order by cost/benefit:
  1. **Fundamentals + valuation** (free: SEC EDGAR/XBRL; or yfinance snapshots) → quality/value ranks to blend with momentum.
  2. **Short interest** (FINRA bi-monthly, free) → crowding filter and (later) short-book candidate screen.
  3. **Earnings dates + estimate revisions** (earnings calendar is cheap; revisions may need a paid source) → avoid holding into events, or exploit post-earnings drift.
  4. **Options implied volatility** (delayed chains are enough for daily horizon) → IV rank as a risk/timing feature.
  Each becomes a candidate rank blended into the factor book, evaluated with the exact FACTOR-VALIDATION harness (walk-forward, embargo, cost-aware). No new model families until at least one new data class is in.
- **P3-3. Commodity momentum satellite.** Take the CROSS-ASSET-MOMENTUM-V1 commodity top-20 sleeve (0.92 Sharpe, low correlation to the equity factor book) and run the volatility-targeting/drawdown-governor pass the winner board already prescribes. Target: a deployable 10-20%-of-capital satellite that diversifies the equity book's beta.
- **P3-4. Crypto as capped satellite only.** Cap at ≤5% of capital, only inside the combined top-5 cross-asset book, only after stress-cost behavior improves. Do not spend model-development effort here; it rides for free on the portable momentum logic.
- **P3-5. (Conditional) Cross-sectional LLM retrain.** The one legitimate remaining LLM thread: retrain the directional model with market-relative labels (outperform vs universe median) so the label prior is ~50/50 — the `long_short_research`/`long_short_backtest` harness is already built to evaluate it. Gate: run only if idle GPU capacity exists and P3-1/P3-2 are underway; kill permanently if the retrain still produces <45% balanced accuracy or a one-sided signal (note: the first cross-sectional retrain already came back 60.6/39.4/0.0 bull/neutral/bear — the prior is against this working).

## 6. Phase 4 — Portfolio construction across sleeves (weeks 8-16)

Goal: maximize portfolio-level compounding, not single-sleeve return.

- **P4-1. Sleeve architecture.** Core: equity factor book (60-80% of risk). Satellite 1: commodity momentum (10-20%). Satellite 2: crypto momentum (0-5%). Cash/T-bill proxy: remainder and regime buffer.
- **P4-2. Risk budgeting.** Allocate by inverse drawdown-adjusted risk, not equal notional. Rebalance sleeve weights monthly; never intra-week (turnover is a tax).
- **P4-3. Portfolio-level drawdown governor.** PORTFOLIO-CONTROLS-V1 showed sleeve-level de-risking costs return; apply the governor at the *portfolio* level instead: cut gross by 50% when portfolio DD from high-water exceeds 12%, restore on new high-water. Backtest this rule before adopting.
- **P4-4. Attribution.** Internal ledgers per sleeve (the XLE-era multi-book machinery already supports this). Weekly per-sleeve P&L, turnover, cost, and slippage-vs-model report.

## 7. Phase 5 — Live deployment and scaling (month 4+)

Gate to first live dollar (all must hold):

1. Winner board V2 (survivorship-aware, regime-sliced) still shows the factor book beating equal-weight and SPY on Sharpe after stress costs.
2. ≥8 weeks paper with slippage within model tolerance and zero unexplained execution bugs.
3. Reconciliation, kill switch, and max-daily-loss controls tested (including a deliberate failure drill).

Then:

- **P5-1. Tiny live.** Start at risk capital only; 25-50 bps of equity risk per position per STRATEGY.md; the top-20 book naturally caps single-name exposure at ~5%.
- **P5-2. Scale on evidence, not enthusiasm.** Double allocation only after each 3-month block where live tracking error vs paper stays within tolerance and portfolio DD stays inside the governor band. Halve it after any block that violates either.
- **P5-3. Broker path.** Stay on Alpaca for the paper→tiny-live transition; open the IBKR migration project only when live size or instrument scope (FX, futures) demands it.

## 8. What we explicitly will NOT do

Profit maximization is as much about not spending effort where there is no edge:

1. No new LLM/news standalone-alpha branches on the current data (settled negative result; see §1.2).
2. No intraday/tight-stop strategies without minute-bar data and stop-limit modeling (not executable as modeled).
3. No promotion of any strategy from gross returns — canonical cost model or it does not count.
4. No day-trading pivot; no options-selling income systems; no leveraged FX/CFDs (STRATEGY.md exclusions stand).
5. No single-window results on the winner board — multi-window walk-forward only.
6. No real money before the Phase 5 gate, no matter how good paper looks.

## 9. Governance and cadence

- **Promotion rule (unchanged, now global):** a challenger replaces an incumbent only with ≥ +0.05 mean Sharpe at equal-or-better worst drawdown and equal-or-better profitable-window consistency, on the V2 (survivorship-aware) basis, net of stress costs.
- **Weekly:** paper/shadow performance report; slippage vs model; sleeve attribution; tracker update.
- **Monthly:** winner-board refresh with new data; kill/keep review of every active research branch (default is kill for anything two months without a promotion-relevant result).
- **Quarterly:** full regime review; capital re-allocation across sleeves; scaling decision per P5-2.
- All experiments continue to log to `strategy_runs` / `logs/strategy_runs.jsonl` with the checklist in [TRACKER.md](TRACKER.md) §11 before any external publication.

## 10. Expected outcomes (honest bands)

| Horizon | Base case (plan) | Upside case | Failure signal |
|---|---|---|---|
| Months 1-3 | Factor book live on paper; V2 winner board published | Survivorship rerun confirms >30% ann gross edge | Survivorship rerun cuts edge below equal-weight benchmark |
| Months 4-6 | Paper tracks model; blend and 1-2 new data classes evaluated | Blend or fundamentals rank promoted (+0.05 Sharpe) | Paper slippage persistently exceeds stress-cost assumptions |
| Months 7-12 | Tiny live; net return in the 4-10% band with DD < 12% | Multi-sleeve portfolio net 10-15% | Live tracking error breaks tolerance two blocks running |
| Years 2-3 | Net 8-15%, DD 10-15%, ≥2 uncorrelated sleeves live | Sustained Sharpe > 1.2 net at meaningful size | Regime shift the governor fails to contain |

The backtested 52% annualized is the *ceiling shown by an optimistic lens*, not the plan. The plan's profit-maximizing move is to find out — cheaply and quickly, via P2-1/P2-2 — how much of it survives honest measurement, deploy that remainder with tight execution, and widen the edge with data the market's crowded price-only models don't share.

## 11. Immediate next actions (this week)

1. Start P1-1: extract `resid_mom_60d` into a versioned production signal.
2. Start P2-1: pick and wire the survivorship-aware universe source (highest research priority).
3. Standardize the canonical cost model flags (P2-3) so all reruns land on comparable numbers.
4. Decide the V3-5 paper sleeve's fate (P1-4) and update [TRACKER.md](TRACKER.md).
5. Queue the momentum-blend experiment (P3-1) on the existing FACTOR-VALIDATION harness.
