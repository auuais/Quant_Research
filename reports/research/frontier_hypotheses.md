# Frontier Hypotheses Pre-Registration Registry

Purpose: per [NEW_RESEARCH_DIRECTIONS_PLAN.md](../../NEW_RESEARCH_DIRECTIONS_PLAN.md) §7.1, every frontier hypothesis is recorded here **before** compute is spent, with its variant count and its promotion/kill gate. Results are appended after the run; the pre-registration block itself is never edited after the run (corrections are appended as dated notes).

Reading order: a hypothesis with `status: PRE-REGISTERED` has no results yet — the "Result" section stays empty until the backtest has actually executed. `RUN` means results are in the linked report. Variant counts are the multiple-testing denominators used for deflated-Sharpe adjustment.

---

## D1 — Volatility risk premium / VIX term-structure sleeve

- **Registered:** 2026-07-28
- **Status:** `RUN` (2026-07-28) → [vol_carry_v1/](vol_carry_v1/)
- **Hypothesis (H-D1):** The VIX futures term-structure slope, combined with a realized-vol forecast, times exposure to short-volatility ETPs well enough to beat both always-short-vol and cash on risk-adjusted net return, including through volatility crises.
- **Null:** Signal-gated rotation is no better net of costs than static always-short-vol (i.e. the premium is real but untimeable), or the crises make any short-vol allocation unacceptable at any gating.

### Data and instruments (pre-declared)

| Item | Decision |
|---|---|
| Signal source | CBOE official index CSVs (`VIX`, `VIX3M`, `VIX9D`, `VVIX`), Yahoo fallback |
| Term-structure proxy | Cash-index slope `VIX3M/VIX − 1`. CBOE per-day VIX **futures** settlement files are Cloudflare-blocked (verified HTTP 403 on 2026-07-28), so the front-future basis is not available for free; the cash slope is the documented standard proxy. Declared as a limitation, not worked around. |
| Tradeable leg | Short-term VIX futures index daily returns reconstructed from ETPs: prefer `VXX` (+1x, 2018-01-25+), else `SVXY` (−1x through 2018-02-27, −0.5x after), cross-checked against `UVXY` (+2x/+1.5x). Leverage regimes are verified empirically by rolling beta vs VXX before use, not assumed. |
| History | 2011-10-04 (first ETP date) → present. Contains 2015-08, 2016-06 Brexit, 2018-02 Volmageddon (real −83% SVXY day), 2020-03 COVID, 2022 bear, 2024-08-05 unwind. |
| Cash leg | `BIL` |
| Costs | Primary 3 bps/side, stress 10 bps/side (plan §3 VIX-ETP profile) |
| Position cap | Short-vol leg ≤ 5% of portfolio capital at deployment |

### Variants (pre-declared: 12 total)

Contango entry threshold on the slope `{0.0, 0.02, 0.05}` × VRP gate `{off, on}` × backwardation long-vol leg `{off, on}` = 12. No other tuning dimension is permitted; anything beyond this grid requires a new registration entry.

### Baselines (must be beaten)

`always_short_vol` (static −1x), `cash_only` (BIL), `spy_buy_and_hold`.

### Gate

Promote if: net Sharpe ≥ 1.0 across windows; ≥ 70% profitable windows; standalone worst DD ≤ 25%; and beats always-short-vol on drawdown in both the 2018-02 and 2020-03 slices.
**Kill** if: no gated variant beats always-short-vol net of primary costs, or no variant survives the crisis slices.

### Result — `WATCHLIST`, not promoted (run 2026-07-28)

Evaluation 2013-11-06 → 2026-07-27, 3,197 days, 12 variants as declared. Best variant `slope_02_vrp`.

| Strategy | CAGR | Ann vol | Sharpe | Worst DD (full history) |
|---|---:|---:|---:|---:|
| `slope_02_vrp` (best gated) | 12.56% | 38.9% | 0.51 | −61.7% |
| `slope_02` | 13.37% | 40.5% | 0.52 | −61.2% |
| `baseline_spy` | 13.88% | 17.1% | 0.85 | −33.7% |
| `baseline_always_short_vol` | −0.28% | 53.4% | 0.36 | **−95.2%** |

Gate: **3 of 7 checks failed** — mean Sharpe ≥ 1.0 (best window-mean 0.68), ≥ 70% profitable windows (58%), and full-history worst DD ≤ 25% (−61.7%). Passed: beats always-short-vol on drawdown in both 2018-02 and 2020-03, clears the 12-trial multiple-testing hurdle (0.467), and beats always-short-vol on full-history Sharpe. The pre-registered **kill** rule did *not* fire, so this is watchlist rather than dead.

What is true: the volatility risk premium is real and the term-structure gate is what makes it survivable. Always-short-vol returned −3.5% in total over 12.7 years with a −95.2% drawdown — the premium exists but static harvesting is ruinous. Gating lifts that to ~12.5% CAGR and cuts Volmageddon from −90.4% to −39.2%, COVID from −55.3% to −17.6%, Aug-2015 from −54.2% to −24.9%. What is also true: SPY beat every variant on Sharpe (0.85) with a third of the volatility, so this sleeve earns no capital on these numbers.

**Three methodology amendments, all made before strategy results were examined:**

1. **Real instruments, not synthetic leverage.** A −1x series rebuilt from VXX loses 48.7% over Volmageddon while the real −1x product (SVXY) lost 90.4%. VIX futures moved between the 16:00 equity close and the 16:15 futures settlement, so ETPs striking NAV at different times are not interchangeable in exactly the crises that decide this sleeve. Backtests use the real closes of the product actually holdable.
2. **Windows tiled across the whole sample.** The inherited harness takes the last `window_count` windows, which for a 12.7-year sample covered only the final 630 days and excluded every crisis before 2024 — making always-short-vol look survivable (window-mean Sharpe 0.75 vs its true full-history 0.36). All 26 non-overlapping windows are now used.
3. **Signal lag 2, not 1.** VIX cash indices settle at 16:15 ET while the ETP trades the 16:00 close, so the usual same-close convention embeds up to 15 minutes of look-ahead — landing precisely on Feb-2018.

**The look-ahead test is the most important result in this entry.** At lag 1 the long-vol variants returned **+32.0%** through Volmageddon and looked like a crisis hedge; at lag 2 the same variants return **−32.7%**, a sign flip. Best-variant Sharpe fell from 0.72 to 0.48 and Volmageddon protection from −10.6% to −39.2%. Roughly three-quarters of the apparent crisis protection was a settlement-timing artifact. Any future volatility work in this repo must use lag 2 or intraday data.

Secondary diagnostic worth carrying forward: the HAR-RV forecast has out-of-sample correlation 0.34 with realized variance but **does not** beat an expanding-mean baseline on RMSE, so the VRP gate is weaker than the term-structure gate — visible in the results, where the VRP variants are near-ties with their slope-only counterparts.

**Next action if revisited:** the front-future basis (not the cash slope) is the signal the literature actually uses, and it needs paid curve data. Do not re-tune this grid — the failure is a risk-profile failure, not a threshold failure.

---

## D2 — Event and seasonality premia battery

- **Registered:** 2026-07-28
- **Status:** `PRE-REGISTERED`
- **Hypothesis (H-D2):** A set of documented calendar/announcement premia persist and are harvestable with daily MOC orders on index ETFs.
- **Null:** Effects are in-sample artifacts, have decayed post-publication, or are smaller than round-trip cost.

### Sub-hypotheses (pre-declared: 6 families / 14 variants)

| ID | Hypothesis | Variants |
|---|---|---|
| E1 | Pre-FOMC drift: long from T−1 close to announcement-day close is positive | 2 (SPY, QQQ) |
| E2 | Overnight (close→open) returns dominate intraday (open→close) on index ETFs and the stock panel | 3 (SPY, QQQ, panel) |
| E3 | Turn-of-month: last 4 + first 3 trading days outperform | 2 (base window, ±1 day) |
| E4 | OPEX week / OPEX day / quarter-end effects are non-zero | 3 |
| E5 | Announcement-gap PEAD: top-decile positive gap names drift up | 2 (5d, 20d holds) |
| E6 | VIX-spike mean reversion: SPY forward returns after VIX percentile spikes | 2 (p90, p95) |

### Confirmation rule (stricter than usual, because this is a battery)

A hypothesis survives only if **both**: (a) the bootstrap 95% CI of the mean event return excludes zero in the first 70% **and** the last 30% of history independently, and (b) mean effect ≥ 2× round-trip cost. Threshold re-tuning after seeing results is forbidden — a failed hypothesis is dropped permanently. Deflated Sharpe is reported against the 14-variant denominator.

### Gate

Standalone micro-sleeve: net Sharpe ≥ 0.8 with ≤ 10% DD. Overlay: +0.03 portfolio Sharpe.
**Kill:** any hypothesis failing two-subperiod confirmation is dropped permanently.

### Result

_Empty — not yet run._

---

## D3 — Automated alpha factory (formulaic factor search)

- **Registered:** 2026-07-28
- **Status:** `PRE-REGISTERED`
- **Hypothesis (H-D3):** Automated search over a formulaic price/volume factor DSL finds factor ensembles with higher validation RankIC than the hand-built winners (`resid_mom_60d`, `ret_20d`), and the ensemble survives to a once-run test period.
- **Null:** Search finds only overfitted expressions whose IC decays to zero out of sample — the expected outcome if the 100-name / ~3.5-year panel is too small for mining.

### Protocol (pre-declared; this protocol is the point of the entry)

| Split | Range | Use |
|---|---|---|
| Train | panel start → 2024-06-30 | factor generation + in-sample IC |
| Validation | 2024-07-01 → 2025-06-30 | acceptance decisions (RankIC ≥ 0.02, ICIR ≥ 0.25) |
| Test | 2025-07-01 → panel end | **run exactly once, on the final ensemble only** |

Embargo: 5 trading days between splits. Dedupe: reject any candidate with |Spearman| > 0.7 vs an already-accepted factor. Search budget declared up front: ≤ 4000 candidate evaluations; the realized count is reported and used as the deflated-Sharpe denominator. Baselines: `resid_mom_60d`, `ret_20d`, equal-weight.

### Gate

Standard challenger rule: +0.05 mean Sharpe vs incumbent at equal-or-better DD, on the once-run test period and the multi-window harness.
**Kill:** if validation→test RankIC decays > 50% across the accepted pool, freeze the factory until the point-in-time panel (PMP P2-1) exists. Do not iterate against the test period.

### Result

_Empty — not yet run._

---

## Registry conventions

1. New entries are appended, never inserted, so registration order is auditable.
2. `status` transitions are one-way: `PRE-REGISTERED` → `RUN`. A re-run under changed conditions (new panel, new data source) gets a **new** entry (`D3-b`, etc.), not an edit of the original.
3. Variant counts are binding. Exceeding a declared grid invalidates the entry's deflated-Sharpe claim and requires a new registration.
4. Result sections stay empty until the corresponding backtest has actually executed. No projected, expected, or illustrative numbers are ever written here.
5. Kill decisions are recorded even when inconvenient — negative results are the main reason this registry exists.
