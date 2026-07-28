# New Research Directions Plan (Frontier Plan V1)

Last updated: 2026-07-03
Status: `PROPOSED`
Companion docs: [PROFIT_MAXIMIZATION_PLAN.md](PROFIT_MAXIMIZATION_PLAN.md) (exploit/harden the incumbent), [STRATEGY.md](STRATEGY.md), [research_winner_board_v1.md](reports/research/research_winner_board_v1.md), [RESEARCH_OUTCOMES.md](RESEARCH_OUTCOMES.md)

## 0. Purpose and scope boundary

[PROFIT_MAXIMIZATION_PLAN.md](PROFIT_MAXIMIZATION_PLAN.md) is the **exploit** plan: deploy `resid_mom_60d`, neutralize validity threats, and squeeze incremental data classes into the existing equity factor book.

This document is the **explore** plan: net-new alpha families on instruments, risk premia, data classes, and model techniques this repo has *never* tested. Nothing here modifies the incumbent pipeline. Every direction is a self-contained research sleeve with its own harness, its own promotion gate, and a kill rule. The goal is maximum *portfolio* return, which at this point is bounded less by tuning the equity momentum book and more by adding return streams that do not share its risk (beta ~0.85, single premium, single asset class).

Design rules for everything below:

1. **Different premium or different information, not a different model on the same data.** Three years of results show model complexity over the same price+headline data does not create alpha here (HYBRID-V1, Qwen classifiers, DL branches, REGIME-SWITCH-V1 all confirmed this).
2. **Agent-implementable.** Each direction has a numbered playbook with file paths, data sources, experiment specs, report artifacts, and decision rules that an agent can execute without further context.
3. **Same governance.** Canonical cost model, walk-forward multi-window evaluation, pre-registered hypotheses, net-of-stress-cost promotion only.

## 1. Coverage audit: what has been explored vs not

### 1.1 Universe coverage

| Universe | Status | Evidence |
|---|---|---|
| US large-cap stocks (~100 currently-listed Alpaca names) | Heavily tested | FACTOR-VALIDATION-V1, HYBRID-V1, all sentiment branches |
| Top-5 US mega caps | Heavily tested | DeepSeek V0-V4 directional runs |
| Liquid US ETF basket (SPY/QQQ/IWM/TLT/GLD/sectors) | Tested (rule-based) | Historical research, basket daily cycle |
| Commodity ETFs (24) | Tested | CROSS-ASSET-MOMENTUM-V1, DL/commodity runs |
| Crypto spot (11 Alpaca pairs) | Tested (directional only) | Crypto momentum, crypto DL, DeepSeek crypto |
| FX (daily ECB mid rates via Frankfurter) | Tested (overlays) | DL FX, DeepSeek FX overlay |
| Non-US stocks via ADRs | Lightly tested | DeepSeek non-US overlay |
| **Index options / equity options (any)** | **Never** | — |
| **VIX complex / volatility ETPs (VXX, SVIX, SVXY, UVXY)** | **Never** | — |
| **Futures of any kind (index, rates, FX, commodity)** | **Never** (commodities only via ETFs) | — |
| **Rates/credit beyond TLT-in-a-basket** | **Never** | — |
| **Crypto derivatives (perps, CME futures, funding)** | **Never** | — |
| **Broader equity cross-section (500-1000 names)** | **Never** (capped at 100) | — |

### 1.2 Technique coverage

| Technique family | Status |
|---|---|
| Rule-based momentum / mean reversion / trailing stops | Exhausted on single names and baskets |
| Classical ML (logit, HGB, extra trees, SGD) on price+news features | Tested, negative (HYBRID-V1, SENTIMENT-V2) |
| CNN/DL on single-asset price windows | Tested, flat after fees (DL branches) |
| LLM prompt sentiment (DeepSeek, Qwen3) + LoRA fine-tune | Tested, structural permabull, dead end as alpha |
| Embedding classifiers (Qwen 2560-d headline embeddings) | Tested, ~random accuracy (V3-Q3E, SENTIMENT-V3) |
| Cross-sectional factor construction + walk-forward validation harness | Built and validated (the one clear success) |
| Static portfolio controls, learned regime switch (HGB) | Tested, not promoted |
| Conviction sizing, triple-barrier labels, calibration | Tested (Q3E); sizing helps, labels hurt |
| **Carry signals (any asset class)** | **Never** |
| **Volatility risk premium / term-structure signals** | **Never** |
| **Statistical arbitrage: PCA/ETF residuals, OU, cointegration, Kalman** | **Never** |
| **Short-term reversal as a factor** | **Never** |
| **Event studies (FOMC, earnings, expiration, turn-of-month, overnight)** | **Never** |
| **Formulaic alpha search (genetic/evolutionary/LLM-generated factors)** | **Never** |
| **Time-series foundation models (Chronos, TimesFM, Moirai, Kronos)** | **Never** |
| **Cross-sectional neural rankers (listwise loss), GKX-style ML benchmark** | **Never** |
| **Reinforcement learning (portfolio or alpha mining)** | **Never** |
| **Vol forecasting models (HAR-RV, GARCH) as signal inputs** | **Never** |
| **Filings/insider-disclosure NLP (10-K/10-Q diffs, Form 4)** | **Never** (only third-party headlines) |

### 1.3 Data-class coverage

Available today: daily/hourly OHLCV (Alpaca IEX + Yahoo fallback), Alpaca news headlines, ECB FX mids. Never acquired: fundamentals, filings text, earnings dates/estimates, insider transactions, short interest, options chains/IV, futures curves, funding rates, macro-event calendars, ETF flows.

### 1.4 The structural lesson

The winner board's repeated result — price/factor structure works, headline-derived signals do not — points the frontier at four things: **(a)** different risk premia (volatility, carry), **(b)** issuer-sourced rather than media-sourced text (filings, insider actions), **(c)** cross-sectional structure not yet extracted (residual reversal, stat-arb), and **(d)** automated search over factor space instead of hand-crafting one factor at a time.

## 2. What top-tier researchers run that this repo does not (SOTA survey, 2024-2026)

1. **Multi-asset carry + trend on futures.** Carry positively predicts returns in every major asset class ([Koijen, Moskowitz, Pedersen, Vrugt, "Carry", JFE 2018](https://www.aqr.com/Insights/Research/Journal-Article/Carry)); combining carry with time-series momentum is the standard CTA stack, with the 2022 caveat that trend must be allowed to override carry ([Research Affiliates 2026](https://www.advisorperspectives.com/commentaries/2026/02/26/trend-follow-carry-lessons-bonds-gold-2022?topic=insurance-annuities), [ReSolve](https://investresolve.com/enhancing-portfolio-returns-with-futures-carry-strategies/), [Return Stacked](https://www.returnstacked.com/carry-the-yield-ride-the-trend-a-strategic-partnership/)).
2. **Volatility risk premium and VIX term-structure carry.** Index implied vol persistently exceeds realized; VIX futures sit in contango roughly 80% of the time, and the term-structure slope is one of the most robust signals for vol-linked instrument returns ([Macrosynergy](https://macrosynergy.com/research/vix-term-structure-as-a-trading-signal/), [QuantSeeker](https://www.quantseeker.com/p/timing-volatility-with-the-vix-term), [Johnson, "Equity Risk Premia and the VIX Term Structure"](https://bauer.uh.edu/departments/finance/documents/seminars/Johnson_020712.pdf), [Volatility Box on contango/roll yield](https://volatilitybox.com/research/vix-contango-backwardation/)).
3. **ML on the equity cross-section, done properly.** Gu-Kelly-Xiu boosted-tree/NN panels remain the benchmark; heavily overparameterized models beat simple ones out of sample ([Kelly, Malamud, Zhou, "The Virtue of Complexity in Return Prediction", JF 2024](https://onlinelibrary.wiley.com/doi/10.1111/jofi.13298), ["The Virtue of Complexity Everywhere"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4166368)).
4. **Automated alpha mining with LLMs/RL.** 2023-2026 produced an entire literature on machine-generated formulaic factors: AlphaGen (RL), AlphaForge, [AlphaAgent (KDD 2025)](https://dl.acm.org/doi/10.1145/3711896.3736838) ([code](https://github.com/RndmVariableQ/AlphaAgent)), [Chain-of-Alpha](https://arxiv.org/abs/2508.06312), [LLM-powered MCTS factor mining](https://arxiv.org/html/2505.11122v2), [AlphaEval evaluation framework](https://arxiv.org/html/2508.13174v1), and Microsoft's open-source [RD-Agent](https://github.com/microsoft/RD-Agent) quant loop.
5. **Time-series foundation models.** TimesFM-2.5, Chronos-2, Moirai-2, Kronos. Honest reading: zero-shot daily-return forecasting is weak (Chronos-large OOS R² ≈ -1.4%, directional accuracy ≈ 51%; TimesFM-500M worse — [Pretrained TSFMs for Financial Return Forecasting](https://arxiv.org/abs/2606.27100), [Re(Visiting) TSFMs in Finance](https://arxiv.org/pdf/2511.18578)), so the SOTA use is as *feature generators* and fine-tuned components, not oracles ([Kinlay on Kronos](https://jonathankinlay.com/2026/02/time-series-foundation-models-for-financial-markets-kronos-and-the-rise-of-pre-trained-market-models/)).
6. **Statistical arbitrage on factor residuals.** Classical PCA/ETF-residual OU books earn Sharpe ~0.9-0.95 ([2025 pairs-trading survey](https://ideas.repec.org/p/war/wpaper/2025-22.html), [WNE WP484](https://www.wne.uw.edu.pl/application/files/8917/5759/3293/WNE_WP484.pdf)); deep-learning extensions on residual paths substantially improve gross performance ([Deep Learning Statistical Arbitrage, Management Science](https://doi.org/10.1287/mnsc.2022.03132), [Attention Factors for Statistical Arbitrage](https://arxiv.org/html/2510.11616v1)).
7. **Announcement and clock premia.** The pre-FOMC announcement drift persists through 2024 tests ([QuantSeeker](https://www.quantseeker.com/p/trading-the-fed-the-pre-fomc-drift), [Applied Economics 2024](https://www.tandfonline.com/doi/full/10.1080/00036846.2024.2322573), original [Lucca-Moench](https://www.bostonfed.org/-/media/Documents/conference/PDF/Lucca_preFOMCDrift.pdf) — a large share of 1994-2011 equity excess returns accrued in the 24h pre-FOMC window); overnight-vs-intraday return splits and similar macro-release drifts are documented across ETFs.
8. **Issuer-disclosure NLP.** Textual *changes* in periodic filings predict returns ("Lazy Prices", Cohen-Malloy-Nguyen, JF 2020); LLMs tracking evolving disclosure signals in earnings calls generate portable alpha signals ([From Text to Alpha, 2025](https://arxiv.org/html/2510.03195v5)); opportunistic and clustered insider buys carry ~5%+ annual alpha (Cohen-Malloy-Pomorski, JF 2012; [recent survey of insider-signal studies](https://verityplatform.com/wp-content/uploads/2026/04/VerityData-Insider-Academic-Studies.pdf)).
9. **Options cross-section ML.** Delta-hedged option returns are predictable with nonlinear ML, surviving transaction costs in recent studies ([Option Return Predictability, RFS](https://academic.oup.com/rfs/article-abstract/35/3/1394/6294944), [JFM 2025 China evidence](https://onlinelibrary.wiley.com/doi/10.1002/fut.22604), [commodity options JFM 2025](https://onlinelibrary.wiley.com/doi/10.1002/fut.22614), [Alpha Architect summary](https://alphaarchitect.com/using-machine-learning-to-predict-options-returns/)).
10. **Crypto delta-neutral carry.** Perp funding-rate arbitrage is the documented persistent crypto premium — market-neutral 15-35% annualized in recent studies ([CEX/DEX funding-arb risk/return study](https://www.sciencedirect.com/science/article/pii/S2096720925000818), [two-tiered funding market structure](https://www.mdpi.com/2227-7390/14/2/346)) — but the edge is decaying: carry Sharpe 6.45 over 2020-2025 falls to 4.06 from 2024 and turns negative in 2025 ([Crypto as an Investable Asset Class](https://arxiv.org/pdf/2510.14435)).

## 3. Feasibility frame

| Constraint | Implication |
|---|---|
| Broker: Alpaca today | Equities/ETPs (incl. VIX ETPs), crypto spot, **US equity/ETF options up to Level 3 (multi-leg)**, historical options data since Feb 2024 ([docs](https://docs.alpaca.markets/us/docs/historical-option-data)). No futures, no index (SPX) options. |
| Broker: IBKR planned | Unlocks futures (micro contracts), index options, non-US. Required for D4 deployment; research does not wait for it. |
| GPU: RTX 3080 10GB | Local LLM ≤8B 4-bit (existing DeepSeek/Qwen stack), Chronos-Bolt/TimesFM-200M fine. No large fine-tunes. |
| Data budget | Cap **$500 total** on new data until a frontier sleeve is promoted. Free-first ordering below. |
| STRATEGY.md exclusions | "Options-selling income systems", leveraged FX/CFDs, HFT remain excluded. **Proposed amendment (needs owner sign-off): defined-risk option structures (max loss = spread width, ≤2% equity per structure) and micro futures are permitted as research sleeves.** Naked short options remain banned permanently. |
| Known data quirks | IEX volume under-reporting (×30 `volume_scale` fix) must be applied in any capacity/participation logic; Frankfurter FX are mids with no tradeable spread. |

Per-instrument canonical cost profiles (extend the Phase-2 cost model with these; every backtest below reports primary and stress):

| Instrument | Primary | Stress |
|---|---|---|
| US stocks/ETFs | 1 bps/side + SEC/TAF | 3 bps/side |
| VIX ETPs (VXX/SVIX/SVXY) | 3 bps/side | 10 bps/side |
| Futures (micro, IBKR) | $0.85/side/contract + 1 tick | + 2 ticks |
| Options (SPY liquid weeklies) | 1% of premium + $0.65/contract | 3% of premium |
| Crypto spot | 30 bps/side | 60 bps/side |
| Crypto perp (research only) | 5 bps taker/leg | 10 bps/leg |

## 4. The directions

Priority = expected sleeve value × probability of survival × (1 / cost-to-first-result). Tier 1 starts immediately and needs no new spend.

---

### D1. Volatility risk premium / VIX term-structure sleeve — Tier 1

**Hypothesis.** The index volatility risk premium (implied > realized ~80% of the time) and VIX futures contango are persistent, structural premia. A signal-gated rotation across short-vol ETP / cash / long-vol ETP, keyed on term-structure slope and a realized-vol forecast, earns a return stream with low calm-regime correlation to the equity momentum book.

**Why new here.** Volatility as an asset class has never been touched; no term-structure or implied-vol data exists in the repo. This is the highest-conviction unexplored premium: decades of academic and practitioner evidence, tradeable today on Alpaca via ETPs (no options needed), free data.

**Data (all free).** `^VIX`, `^VIX9D`, `^VIX3M`, `^VIX6M` daily from Yahoo; VIX futures settlement CSVs from CBOE's historical data portal (per-contract files); ETPs from existing Yahoo/Alpaca paths: `VXX` (2009+, split-adjusted), `SVXY` (2011+, note −0.5x leverage change Feb 2018), `SVIX` (Mar 2022+), `UVXY`, `BIL` (cash proxy); SPY for realized vol.

**Signals.**
1. Term slope: `VIX3M/VIX − 1` and front-future basis `(F1 − VIX)/VIX` (contango/backwardation).
2. VRP: `VIX² − HAR-RV forecast of 21d realized variance` (implement HAR-RV; first vol-forecasting model in the repo).
3. Guards: VVIX percentile, VIX level percentile, 5-day VIX trend.

**Book.** Daily close decision, MOC execution, three states: short-vol (SVIX or SVXY), cash (BIL), long-vol (VXX, only on confirmed backwardation + rising VIX trend). Position cap: 5% of portfolio capital initially — sized so a Feb-2018-style overnight event (XIV lost ~96%) costs ≤3% of portfolio.

**Agent playbook.**
1. Create `src/algoding/data/volatility.py`: Yahoo loaders for the VIX family + ETPs; CBOE futures-curve CSV downloader/parser; constant-maturity F1/F2 series construction; parquet cache under `cache/volatility/`.
2. Create `src/algoding/research/vol_signals.py`: HAR-RV realized-vol forecaster, term-structure features, VRP series, guard features. Unit-check timing: all features known at T close for a T-close trade.
3. Create `src/algoding/execution/vol_carry_research.py` (CLI: `vol-carry-research`): backtest 2011→present (use short-VXX synthetic with borrow-cost assumption pre-2022, real SVIX after; report both), state machine above, grid over slope thresholds (small grid, ≤12 variants, pre-registered), VIX-ETP cost profile, five 126-day windows + 252-day recent + full-history.
4. Mandatory crisis-slice report: Aug 2015, Feb 2018, Mar 2020, 2022 H1, Aug 5 2024. Include worst-overnight-gap table and a "what if short-vol leg −60% overnight" scenario row.
5. Write `reports/research/vol_carry_v1/` (run JSON + summary md) and add a winner-board entry.

**Promotion gate (absolute, new-sleeve).** Net Sharpe ≥ 1.0 across windows; ≥70% profitable windows; sleeve worst DD ≤ 25% standalone (≤1.5% at portfolio level given 5% cap); must beat both buy-and-hold SVIX and cash in the 2018+2020 slices on DD. **Kill.** Signal-gated variant fails to beat always-short-vol net of costs, or no gating scheme survives the crisis slices.

---

### D2. Event & seasonality premia battery — Tier 1

**Hypothesis.** A set of documented calendar/announcement premia — each small, capital-efficient, and uncorrelated with cross-sectional momentum — still exists and can be harvested with daily MOC orders on index ETFs (and checked on the 100-name panel and crypto).

**Why new here.** Zero event studies exist in the repo; the FOMC calendar, earnings dates, and expiration calendar have never been loaded. These are index-level *timing* premia — a dimension (when) orthogonal to everything tested (what).

**Battery (each pre-registered before any backtest is run).**
- E1 Pre-FOMC drift: long SPY from T−1 close to announcement-day close, ~8 events/yr (persists through 2024 per [QuantSeeker](https://www.quantseeker.com/p/trading-the-fed-the-pre-fomc-drift)).
- E2 Overnight vs intraday split: SPY/QQQ close→open vs open→close legs; repeat on the 100-name panel (retail-heavy names) and BTC/ETH hourly (US-hours effect).
- E3 Turn-of-month: long equities last 4 + first 3 trading days.
- E4 Monthly OPEX week and quarter-end rebalance effects (exploratory tier: stricter gate).
- E5 Earnings-anchored: (a) skip-earnings-day variant of the factor book (risk check, not alpha), (b) PEAD-lite — hold names whose announcement-day return is a top-decile positive gap for 5-20 days (announcement-day return as the surprise proxy; no estimates data needed).
- E6 VIX-spike mean-reversion re-entry timing (feeds D1 and the equity book's drawdown governor).

**Agent playbook.**
1. Create `src/algoding/data/events.py`: static FOMC meeting calendar 2011→present (hardcoded list from the Fed website, updated yearly); earnings dates via yfinance with EDGAR 8-K fallback, cached to parquet; computed OPEX (3rd Friday) and month-turn calendars.
2. Create `src/algoding/research/event_study.py`: generic CAR harness — align returns on event windows, bootstrap CIs, subperiod split (first 70% vs last 30% of history), effect-size vs cost ratio. This harness is shared with D6.
3. Create `src/algoding/execution/event_premia_research.py` (CLI: `event-premia-research`): run the battery E1-E6, then convert survivors (CI excludes zero in *both* subperiods AND mean effect ≥ 2× round-trip cost) into rule backtests through the canonical cost engine.
4. Write `reports/research/event_premia_v1/` with one section per hypothesis, including the pre-registration block and the number of variants tested (multiple-testing denominator).
5. Survivors become either (a) micro-sleeves (e.g., pre-FOMC overlay on cash) or (b) overlays on existing books (E5a, E6).

**Promotion gate.** Standalone micro-sleeve: net Sharpe ≥ 0.8 with ≤10% DD and low time-in-market; overlay: +0.03 portfolio Sharpe in combination test. **Kill.** Any hypothesis failing the two-subperiod confirmation is dropped permanently — no threshold re-tuning.

---

### D3. Automated alpha factory (LLM + evolutionary formulaic search) — Tier 1

**Hypothesis.** The space of formulaic price/volume factors over the existing panel contains rankings better than the two hand-built winners, and a generate→evaluate→prune loop (evolutionary search seeded and critiqued by the local LLM) can find an ensemble of uncorrelated factors that beats `resid_mom_60d` under the standard promotion rule.

**Why new here.** Every factor in the repo was hand-crafted (roughly a dozen tried in total). Top-tier 2023-2026 work — [AlphaGen](https://arxiv.org/abs/2508.06312) (RL), [AlphaAgent](https://dl.acm.org/doi/10.1145/3711896.3736838), [Chain-of-Alpha](https://arxiv.org/abs/2508.06312), [LLM-MCTS mining](https://arxiv.org/html/2505.11122v2), [RD-Agent](https://github.com/microsoft/RD-Agent) — industrialize this search. Crucially, this reuses the local LLM stack in the one role where its permabull bias is irrelevant: it writes *factor code*, not market opinions.

**Agent playbook.**
1. Create `src/algoding/research/alpha_dsl.py`: a small factor DSL — operators `ts_rank, ts_mean, ts_std, ts_corr, ts_cov, delta, delay, decay_linear, ts_min/max, cs_rank, cs_zscore, winsorize, sign, log, abs` over columns `open, high, low, close, volume, vwap, returns` — with a safe vectorized evaluator on the panel DataFrame, expression parser, max depth 8, NaN discipline, and a deterministic hash per canonical expression.
2. Create `src/algoding/research/alpha_search.py`: (a) seed pool from Alpha101-style classics; (b) mutation/crossover operators; (c) LLM proposal mode — local Qwen/DeepSeek receives the DSL grammar + current best factors + a market hypothesis prompt and emits candidate expressions (AlphaAgent pattern: hypothesis → factor → critique); (d) dedupe: reject candidates with |Spearman corr| > 0.7 to any accepted factor.
3. Evaluation pipeline inside `src/algoding/execution/alpha_factory_research.py` (CLI: `alpha-factory-run`): per-candidate daily RankIC and ICIR on **train** (2022-09→2024-06) and **validation** (2024-06→2025-06) with 5-day embargo; acceptance requires val RankIC ≥ 0.02, ICIR ≥ 0.25, turnover below cap. The **test** period (2025-06→present) is run exactly once, on the final ensemble only.
4. Ensemble: top-K (K≈5-10) accepted factors → rank-average book and LightGBM-stacked book, both through the exact FACTOR-VALIDATION harness (top-20, 5-day rebalance, canonical costs) against `resid_mom_60d`.
5. Report `reports/research/alpha_factory_v1/`: every accepted factor's expression + hypothesis text, the full count of candidates evaluated, deflated-Sharpe adjustment for that count, and the single test-period result.

**Promotion gate.** Standard challenger rule: +0.05 mean Sharpe vs incumbent at equal-or-better DD, on the once-run test period and the multi-window harness. **Kill.** If validation→test RankIC decays >50% across the accepted pool, the 100-name/3.5-year panel is too small for mining — freeze the factory until the PMP P2-1 point-in-time panel (survivorship-fixed, longer) exists, then re-run. Do not iterate against the test period.

---

### D4. Futures carry + time-series momentum (CTA sleeve) — Tier 2

**Hypothesis.** A vol-scaled combination of 3/6/12-month time-series momentum and curve carry across ~20 liquid futures (rates, FX, metals, energy, grains, equity index) replicates the classic CTA premium (lit Sharpe ~0.7-1.1) with near-zero correlation to the equity book — the single best portfolio-level diversifier available.

**Why new here.** No futures, no carry signal, no rates/FX shorting capability exists in the repo; commodity exposure so far is long-only ETFs. Carry requires curve data (front vs next contract), which no current source provides.

**Data plan (two stages).** Stage 1 (free): Yahoo continuous front-month series (`ES=F, NQ=F, ZN=F, GC=F, CL=F, 6E=F, ZC=F`, etc.) for the TSMOM prototype — accepting their roll imperfections for a go/no-go read. Stage 2 (paid, gated on Stage 1 showing trend Sharpe ≥ 0.5): daily settlements for the full curve from [Databento CME](https://databento.com/datasets/OPRA.PILLAR) usage-based (budget ≤ $150) or Norgate, enabling carry + proper back-adjusted rolls.

**Agent playbook.**
1. Create `src/algoding/data/futures.py`: Yahoo continuous loader; Databento adapter (definitions + daily statistics schemas); roll calendar + back-adjustment logic; contract-spec table (tick size/value, multiplier, margin) for ~25 contracts and their micro equivalents.
2. Create `src/algoding/research/futures_carry.py`: TSMOM signals (sign of 3/6/12m vol-scaled returns), carry signal (annualized front-vs-next slope, sign-adjusted per asset class convention), combo z-score, inverse-vol position sizing to a 10% portfolio vol target, weekly rebalance.
3. Create `src/algoding/execution/futures_cta_research.py` (CLI: `futures-cta-research`): Stage-1 trend-only backtest 2010→present on continuous series; Stage-2 rerun with carry + real rolls; costs per the futures profile; **integer micro-contract simulation at $25k/$100k book sizes** (rounding kills small CTA books — report both).
4. Report `reports/research/futures_cta_v1/` with per-asset-class attribution, the 2022 slice (trend-vs-carry conflict year), and correlation vs the equity factor book.
5. Deployment prerequisite (separate task, not research): IBKR paper account + adapter skeleton in `src/algoding/execution/` mirroring the Alpaca adapter interface.

**Promotion gate.** Net Sharpe ≥ 0.8 over 10+ years multi-window; worst DD ≤ 15% at 10% vol target; correlation to equity book ≤ 0.3; integer-contract drag at $25k ≤ 25% of gross Sharpe. **Kill.** Stage 1 trend Sharpe < 0.5 → do not buy Stage-2 data; archive.

---

### D5. Statistical arbitrage / residual reversal — Tier 2

**Hypothesis.** Daily residuals vs PCA/sector-ETF factors mean-revert (OU), and a dollar-neutral s-score book over a *wider* universe (~500 liquid US names) earns a genuinely market-neutral Sharpe ~0.9+ (classical), with a documented DL upgrade path on residual paths ([Management Science](https://doi.org/10.1287/mnsc.2022.03132), [attention factors](https://arxiv.org/html/2510.11616v1)) — the complementary premium to the 5-day momentum book (reversal vs continuation).

**Why new here.** Mean reversion was only ever tested as single-ETF rules; nothing cross-sectional, no residualization, no OU/cointegration/Kalman anywhere. Also the first strategy family that is *actually* beta-neutral — every "market-neutral" attempt so far came from news signals and failed.

**Honest risk up front.** This family lives or dies on costs: daily turnover at 1-3 bps/side retail is the main killer. The playbook therefore requires a cost-sensitivity table (0.5/1/2/3/5 bps) as a first-class deliverable, plus a slower weekly variant as fallback.

**Agent playbook.**
1. Extend the universe: pull daily bars for the ~500 most liquid US names (Alpaca, `volume_scale` fix applied) into the existing panel format; document the survivorship caveat and coordinate with PMP P2-1's point-in-time build (this sleeve must be re-run on the PIT panel before any deployment decision).
2. Create `src/algoding/research/stat_arb.py`: rolling 60-day PCA (top 10-15 factors) and sector-ETF regression variants → residual series; per-name OU fit (κ, σ_eq, half-life); s-score = (spread − mean)/σ_eq; entry |s|>1.25, exit |s|<0.5 (Avellaneda-Lee defaults, pre-registered, no tuning grid beyond ±25%).
3. Create `src/algoding/execution/stat_arb_research.py` (CLI: `stat-arb-research`): dollar-neutral book, half-life filter (≤ 10 days), name cap 2%, borrow cost on shorts, daily and weekly rebalance variants, cost-sensitivity table, standard windows.
4. Report `reports/research/stat_arb_v1/` incl. beta, factor exposures, and net Sharpe by cost tier.
5. Conditional v2 (only if v1 net Sharpe ≥ 0.6 at 2 bps): CNN/GRU classifier on 60-day residual windows predicting next-period residual sign (DL-stat-arb architecture), same harness, GPU-sized.

**Promotion gate.** Net Sharpe ≥ 1.2, |beta| ≤ 0.1, worst DD ≤ 10% (the point of this sleeve is quality, not size). **Kill.** Net edge ≤ 0 at 2 bps/side in both daily and weekly variants.

---

### D6. Corporate disclosure change & insider-event NLP — Tier 2

**Hypothesis.** Signals sourced from *issuers themselves* — semantic changes in 10-K/10-Q risk sections ("Lazy Prices": firms that change their filings underperform), 8-K events, and opportunistic/clustered insider Form-4 buys (~5%+ annual alpha in JF-published work; [survey](https://verityplatform.com/wp-content/uploads/2026/04/VerityData-Insider-Academic-Studies.pdf)) — carry information that third-party headlines (the failed branch) do not.

**Why new here.** All prior NLP was media headlines scored for tone — a crowded, hype-contaminated source that produced the permabull dead end. This direction changes the *information source* (regulatory disclosures, insider actions; free from [SEC EDGAR](https://www.sec.gov/)), the *signal construct* (change vs the firm's own prior filing — self-controlled, no market-tone prior), and reuses the Qwen embedding cache machinery exactly where the evidence says embeddings work ([From Text to Alpha](https://arxiv.org/html/2510.03195v5)).

**Agent playbook.**
1. Create `src/algoding/data/edgar.py`: EDGAR full-text index + filing fetcher (10-K/10-Q/8-K) for the 100-name panel (then 500), respecting SEC rate limits; Form 4 parser (transaction code P open-market buys, insider role, ownership %); parquet caches under `cache/edgar/`.
2. Create `src/algoding/research/filing_features.py`: section extraction (Item 1A Risk Factors, Item 7 MD&A); chunk → embed via the existing Qwen embedding path → per-section cosine distance vs the same firm's previous filing → cross-sectional *change score*. Second feature: local LLM classifies changed spans as risk-increasing/decreasing (bounded task, not market prediction). Third: Form-4 opportunistic cluster-buy score (≥2 distinct non-routine insiders within 10 days; routine = same-calendar-month-as-last-year filter per Cohen-Malloy-Pomorski).
3. Run each feature through the D2 event-study harness (filing date = event) and, as factor ranks, through the FACTOR-VALIDATION harness.
4. Report `reports/research/disclosure_alpha_v1/`: CAR curves by change-score decile, insider-cluster CARs, overlay results on the factor book.

**Promotion gate.** Overlay: standard +0.05 Sharpe rule. Standalone event sleeve (insider clusters): net Sharpe ≥ 0.8 multi-window. **Kill.** No monotonic decile CAR spread over 3 years of filings for the change score; insider CARs indistinguishable from zero after costs.

---

### D7. Options strategies — Tier 3 (gated on exclusion amendment + D1 results)

Three sub-branches, strictly ordered:

1. **D7a — Defined-risk expression of the D1 signal.** When D1 says short-vol, compare SPY put credit spreads / iron condors (defined max loss = width, ≤2% equity per structure) against the ETP position on risk-adjusted return. Data: Alpaca historical options (Feb 2024+) is enough to model *execution* (spreads, fills); the *signal* comes from D1's long history. Deliverable: `src/algoding/data/options.py` (chains + greeks via Black-Scholes internal calc), `src/algoding/research/options_backtest.py` (event-window option P&L simulator crossing the spread), `execution/options_vol_research.py`, report `reports/research/options_vol_v1/`.
2. **D7b — Earnings-vol events.** Implied move (front straddle) vs historical realized move distribution on liquid names pre-earnings; long-only vol initially (buy underpriced straddles), defined-risk short structures only post-amendment. Links D2-E5 calendar data.
3. **D7c — Cross-sectional delta-hedged option-return ML.** The [RFS/JFM literature](https://academic.oup.com/rfs/article-abstract/35/3/1394/6294944) shows real predictability, but this needs years of chain history (Theta Data / ORATS / [Databento OPRA](https://databento.com/datasets/OPRA.PILLAR), ~$50-100/mo) and careful spread-crossing costs. **Research-only until a vol sleeve is live and the data budget is re-opened.**

**Gate to start D7 at all:** owner approves the §3 exclusion amendment, and D1 has produced its v1 report. **Kill:** option execution costs (half-spread) consume >60% of the modeled edge in D7a → stay in ETPs.

---

### D8. Crypto delta-neutral carry (funding + basis) — Tier 3

**Hypothesis.** Perp funding and futures basis are the structural crypto premia (unlike the failed directional momentum). Even with the documented decay (Sharpe 6.45 → negative in 2025 per [this study](https://arxiv.org/pdf/2510.14435)), a conditional harvester (deploy only when trailing funding/basis exceeds a threshold) may clear T-bills with near-zero beta.

**Reality check.** Offshore perps are not accessible to US retail; the deployable expression is CME (micro) BTC futures basis vs spot via IBKR, or nothing. Research is nearly free, so it earns a small slot; deployment is explicitly conditional.

**Agent playbook.**
1. Create `src/algoding/data/funding.py`: public REST loaders for Binance/Bybit/OKX historical funding rates (BTC, ETH, SOL + top-10 by OI) and CME BTC futures settlements; parquet cache.
2. Create `src/algoding/execution/crypto_carry_research.py` (CLI: `crypto-carry-research`): simulate spot-long + perp-short with all-in costs (4 taker legs round trip, funding 3×/day, 2× liquidation buffer capital drag); entry when trailing 7d funding annualizes > 10%, exit < 0; separate CME quarterly basis-capture backtest vs T-bill.
3. Report `reports/research/crypto_carry_v1/` with year-by-year decomposition (the decay question is the headline result) and a jurisdiction/deployment note.

**Promotion gate.** Net-of-fees annualized ≥ T-bill + 5% with beta ≈ 0 over the last 18 months (not the juicy 2020-21 sample). **Kill.** Fails that bar → archive as "known premium, decayed/inaccessible," revisit only if funding regimes change.

---

### D9. TSFMs & modern neural rankers benchmark — Tier 3 (gated on PMP P2-1 PIT panel)

**Hypothesis.** Not that a foundation model predicts returns zero-shot (the literature says it does not: negative OOS R², ~51% directional — [arXiv 2606.27100](https://arxiv.org/abs/2606.27100)), but that (a) TSFM-derived *features* (vol forecasts, quantile spreads, trend probabilities from Chronos-Bolt / TimesFM-200M) add information to a cross-sectional ranker, and (b) a properly built GKX-style ML benchmark (LightGBM on ~50 engineered features; virtue-of-complexity random-features ridge) beats the single hand-built factor — establishing the repo's first serious cross-sectional ML baseline.

**Why new here and why gated.** Prior DL was per-asset CNNs; there has never been a cross-sectional learning-to-rank model or a GKX benchmark. It is gated on the point-in-time panel because ML on a survivorship-biased 100-name panel would manufacture fake alpha — the exact failure mode this program has spent months rooting out.

**Agent playbook.**
1. After PMP P2-1 delivers the PIT panel: create `src/algoding/research/cross_sectional_ml.py` — feature library (~50: momentum family, reversal, vol, dollar-volume, gap stats, plus D1's VIX features and any D3-accepted factors), LightGBM ranker with purged walk-forward, random-features ridge (VoC) benchmark.
2. Create `src/algoding/research/tsfm_features.py`: Chronos-Bolt-small / TimesFM-200M inference wrappers producing 21d vol forecast, P90−P10 forecast spread, and P(up) per name per day (batched, cached; 10GB GPU is sufficient).
3. Create `src/algoding/execution/cs_ml_research.py` (CLI: `cs-ml-research`): ablation grid — LightGBM alone, +TSFM features, ridge benchmark, small listwise transformer — all evaluated only through the FACTOR-VALIDATION harness as challenger ranks.
4. Report `reports/research/cs_ml_v1/` with per-ablation IC and book results.

**Promotion gate.** Standard +0.05 rule vs the then-current incumbent on the PIT basis. **Kill.** If LightGBM-with-everything cannot beat the hand-built factor on validation IC, record it and stop — do not escalate model size.

---

## 5. Shared infrastructure (build once, first)

| ID | Deliverable | Used by |
|---|---|---|
| I1 | `src/algoding/data/volatility.py`, `events.py`, `edgar.py`, `futures.py`, `funding.py`, `options.py` — each: fetch → validate → parquet cache → panel-join helpers, same interface style as `data/alpaca.py` | D1-D9 |
| I2 | `src/algoding/research/event_study.py` — generic CAR/bootstrap harness with subperiod confirmation | D2, D6, D7b |
| I3 | Sleeve report contract: every frontier backtest emits the standard run JSON + summary md into `reports/research/<name>_v1/` and a winner-board row; instrument-specific cost profiles added to the canonical cost model (§3 table) | all |
| I4 | `reports/research/frontier_hypotheses.md` — pre-registration registry: hypothesis, variant count, and gate are written **before** each run; deflated Sharpe reported wherever a battery/search was run (D2, D3 especially) | all |

## 6. Sequencing

```
Wave 1 (weeks 1-3, no new spend):
  I1 (volatility, events) + I2 + I4
  D1 vol-carry v1  |  D2 event battery v1  |  D3 DSL + search MVP     ← three parallel agent tracks
Wave 2 (weeks 3-8):
  I1 (edgar) → D6 filings/insider v1
  D4 Stage-1 trend prototype (free data) → data-purchase decision (≤$150)
  D5 classical stat-arb v1 (+ universe widening to ~500 names)
Wave 3 (weeks 8-16, conditional):
  D4 Stage-2 carry (if Stage 1 passes)  |  D5 DL variant (if v1 passes)
  D7a options expression (if amendment approved + D1 v1 done)
  D8 crypto carry study  |  D9 CS-ML benchmark (after PMP P2-1 PIT panel)
Integration: any promoted sleeve enters PROFIT_MAXIMIZATION_PLAN Phase 4 (multi-sleeve
portfolio construction, risk budgeting, portfolio-level drawdown governor).
```

Dependencies to respect: D3 and D9 want the PIT panel for final judgment (interim runs on the current panel are exploratory only); D7a depends on D1; D4 Stage 2 depends on Stage 1; nothing here blocks or is blocked by PMP Phases 1-2.

## 7. Governance

1. **Pre-registration or it didn't happen.** Every hypothesis and its variant count goes into `frontier_hypotheses.md` before compute is spent. Batteries and factories (D2, D3) report deflated Sharpe with the true trial count.
2. **Absolute gates for new sleeves** (uncorrelated return streams): net stress-cost Sharpe ≥ 0.8-1.2 per direction as specified, correlation to the equity factor book ≤ 0.4, DD within the stated cap. **Relative gate for overlays/challengers:** the standing +0.05 Sharpe rule.
3. **Kill cadence.** Monthly review; any direction two months without a promotion-relevant result is killed by default (same rule as PMP §9). Kill criteria above are binding — no threshold migration after seeing results.
4. **Budget.** ≤ $500 total external data until first frontier promotion; per-direction caps as stated (D4 ≤ $150; D7c deferred).
5. **Risk rules inherited** from STRATEGY.md §14; new-instrument specifics: short-vol capped at 5% of capital with overnight-gap sizing; options defined-risk only, ≤2% equity max loss per structure; futures sized by vol target with integer-contract simulation; no naked short options ever.
6. **Paper before live, always:** any promoted sleeve follows the same shadow → paper → tiny-live ladder as the core book.

## 8. Explicitly out of scope (unchanged from prior plans)

HFT/latency-sensitive intraday; leveraged FX/CFDs; naked short options or undefined-risk premium selling; strategies requiring perfect fills; single-window results; gross-return promotion. The §3 amendment (defined-risk options, micro futures) requires explicit owner sign-off before D7 begins.

## 9. References

Carry/trend: [Koijen et al., "Carry" (JFE 2018)](https://www.aqr.com/Insights/Research/Journal-Article/Carry) · [Research Affiliates on trend vs carry 2022](https://www.advisorperspectives.com/commentaries/2026/02/26/trend-follow-carry-lessons-bonds-gold-2022?topic=insurance-annuities) · [ReSolve futures carry](https://investresolve.com/enhancing-portfolio-returns-with-futures-carry-strategies/) · [Return Stacked carry+trend](https://www.returnstacked.com/carry-the-yield-ride-the-trend-a-strategic-partnership/)
Volatility: [Macrosynergy VIX term structure](https://macrosynergy.com/research/vix-term-structure-as-a-trading-signal/) · [QuantSeeker timing volatility](https://www.quantseeker.com/p/timing-volatility-with-the-vix-term) · [Johnson, VIX term structure](https://bauer.uh.edu/departments/finance/documents/seminars/Johnson_020712.pdf) · [VolatilityBox contango](https://volatilitybox.com/research/vix-contango-backwardation/) · [Cboe VIX products](https://www.cboe.com/tradable-products/vix/)
Alpha mining: [AlphaAgent (KDD 2025)](https://dl.acm.org/doi/10.1145/3711896.3736838) · [AlphaAgent code](https://github.com/RndmVariableQ/AlphaAgent) · [Chain-of-Alpha](https://arxiv.org/abs/2508.06312) · [LLM-MCTS factor mining](https://arxiv.org/html/2505.11122v2) · [AlphaEval](https://arxiv.org/html/2508.13174v1) · [Microsoft RD-Agent](https://github.com/microsoft/RD-Agent)
TSFMs: [Pretrained TSFMs for financial return forecasting](https://arxiv.org/abs/2606.27100) · [Re(Visiting) TSFMs in finance](https://arxiv.org/pdf/2511.18578) · [Chronos multivariate financial forecasting](https://arxiv.org/html/2605.21504) · [Kinlay on Kronos](https://jonathankinlay.com/2026/02/time-series-foundation-models-for-financial-markets-kronos-and-the-rise-of-pre-trained-market-models/)
Cross-sectional ML: [Kelly-Malamud-Zhou, Virtue of Complexity (JF 2024)](https://onlinelibrary.wiley.com/doi/10.1111/jofi.13298) · [Virtue of Complexity Everywhere](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4166368) · [NBER w30217](https://www.nber.org/system/files/working_papers/w30217/w30217.pdf)
Stat-arb: [Deep Learning Statistical Arbitrage (Mgmt Sci)](https://doi.org/10.1287/mnsc.2022.03132) · [Attention Factors for Statistical Arbitrage](https://arxiv.org/html/2510.11616v1) · [2025 ML pairs-trading survey](https://ideas.repec.org/p/war/wpaper/2025-22.html) · [WNE WP484 pairs performance](https://www.wne.uw.edu.pl/application/files/8917/5759/3293/WNE_WP484.pdf)
Events: [Lucca-Moench pre-FOMC drift](https://www.bostonfed.org/-/media/Documents/conference/PDF/Lucca_preFOMCDrift.pdf) · [QuantSeeker pre-FOMC 2025](https://www.quantseeker.com/p/trading-the-fed-the-pre-fomc-drift) · [Applied Economics 2024 pre-FOMC](https://www.tandfonline.com/doi/full/10.1080/00036846.2024.2322573)
Disclosures/insiders: [From Text to Alpha (2025)](https://arxiv.org/html/2510.03195v5) · [Insider-signal academic survey](https://verityplatform.com/wp-content/uploads/2026/04/VerityData-Insider-Academic-Studies.pdf) · Cohen-Malloy-Nguyen "Lazy Prices" (JF 2020) · Cohen-Malloy-Pomorski "Decoding Inside Information" (JF 2012)
Options: [Option Return Predictability (RFS)](https://academic.oup.com/rfs/article-abstract/35/3/1394/6294944) · [JFM 2025 ML option returns (China)](https://onlinelibrary.wiley.com/doi/10.1002/fut.22604) · [JFM 2025 commodity options](https://onlinelibrary.wiley.com/doi/10.1002/fut.22614) · [Alpha Architect summary](https://alphaarchitect.com/using-machine-learning-to-predict-options-returns/) · [Alpaca historical options data](https://docs.alpaca.markets/us/docs/historical-option-data) · [Databento OPRA](https://databento.com/datasets/OPRA.PILLAR)
Crypto carry: [Funding-rate arbitrage risk/return (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/pii/S2096720925000818) · [Two-tiered funding markets](https://www.mdpi.com/2227-7390/14/2/346) · [Crypto as an investable asset class](https://arxiv.org/pdf/2510.14435) · [Funding-rate design](https://arxiv.org/html/2506.08573v1)
