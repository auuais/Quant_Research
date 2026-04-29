# Research Tracker

Last updated: 2026-04-08

## 1. Program objective

Build a research-first algorithmic trading platform that:

- evaluates multiple strategies in parallel
- keeps clean internal attribution by strategy
- promotes only the strongest candidates to broker paper trading
- does not move to real-money trading until research is concluded
- can later publish methodology and results in a defensible way

## 2. Current operating policy

- Broker paper trading is only for the current promoted candidate.
- Parallel strategy evaluation happens in internal ledgers.
- Every research track must log:
  - hypothesis
  - dataset
  - strategy definition
  - metrics
  - drawdown
  - decision

## 3. Research phases

### Phase 1: Current

- U.S. ETF basket
- XLE parallel strategy books
- Rule-based daily/weekly strategies

Status: `ACTIVE`

### Phase 2: Next

- Gold
- Large-cap stocks

Status: `ACTIVE INTERNAL RESEARCH`

### Phase 3: Later

- Major FX pairs

Status: `ACTIVE INTERNAL RESEARCH`

### Phase 4: Research overlays

- ML/DL on the same basket
- LLM sentiment overlay

Status: `ACTIVE EXPANDED STOCK RESEARCH`

## 4. Current promoted candidate

- Strategy family: ETF momentum basket
- Broker state: paper trading
- Current live broker candidate status: `ACTIVE PAPER`

## 5. Internal strategy books

### XLE research set

- `xle_momentum_daily`
- `xle_mean_reversion_daily`
- `xle_breakout_daily`
- `xle_momentum_aggressive`

Purpose:

- compare multiple XLE variants without mixing broker attribution

### Gold research set

- `gld_momentum_daily`
- `gld_mean_reversion_daily`
- `gld_breakout_daily`
- `gld_momentum_aggressive`

Purpose:

- test liquid gold exposure through `GLD` in internal ledgers before broker promotion

### Large-cap stock research set

- `aapl_*`
- `msft_*`
- `nvda_*`
- `amzn_*`
- `meta_*`

Purpose:

- compare the same rule families across a concentrated high-liquidity stock basket

### FX research set

- `eurusd_*`
- `usdjpy_*`
- `gbpusd_*`

Purpose:

- evaluate major FX pairs in research-only ledgers without changing the current broker stack

## 6. Performance log registry

These are the current performance and tracking stores:

- Broker paper orders:
  - `logs/paper_orders.jsonl`
  - PostgreSQL table: `paper_orders`
- Broker positions:
  - `logs/paper_positions.jsonl`
  - PostgreSQL table: `paper_positions`
- Account snapshots:
  - `logs/account_snapshots.jsonl`
  - PostgreSQL table: `account_snapshots`
- Strategy comparison runs:
  - `logs/strategy_runs.jsonl`
  - PostgreSQL table: `strategy_runs`
- Virtual strategy books:
  - `logs/virtual_strategy_books.jsonl`
  - PostgreSQL table: `virtual_strategy_books`
- Virtual strategy trades:
  - `logs/virtual_strategy_trades.jsonl`
  - PostgreSQL table: `virtual_strategy_trades`
- Virtual strategy snapshots:
  - `logs/virtual_strategy_snapshots.jsonl`
  - PostgreSQL table: `virtual_strategy_snapshots`

## 7. Research questions to answer before real trading

- Which rule-based basket model is the strongest after robust testing?
- Which internal strategy books survive across different market regimes?
- Does ML/DL add value over the classical baselines after costs?
- Does LLM sentiment improve timing or ranking enough to matter?
- Which markets are the next best expansion targets after ETFs?
- What risk controls remain non-negotiable before live capital?

## 8. Promotion criteria

To move a strategy from internal research to broker paper candidate:

- positive out-of-sample performance
- acceptable drawdown
- understandable failure modes
- clean execution assumptions
- stable behavior across different windows

To move from broker paper candidate to real-money candidate:

- enough paper duration
- broker sync and reconciliation working
- realistic slippage assumptions
- research review completed
- written approval in this tracker

## 9. Current findings

### ETF basket

- Momentum basket is the current promoted paper candidate.
- Current broker paper position exists.

### XLE research

- `xle_momentum_daily` is currently the strongest internal XLE variant.
- More aggressive variants are still research-only.

### Historical brute-force research

- Cross-market historical backtests are active for:
  - ETFs
  - commodity proxies
  - large-cap stocks
  - major FX pairs
- Current standard windows:
  - `3m`
  - `6m`
  - `9m`
  - `1y`
  - `2y`
  - `3y`
- Current risk controls applied in historical tests:
  - fixed stop loss `5%`
  - trailing stop `7%`
- Current top aggregate contenders across all six windows are:
  - `USO / uso_momentum_daily`
  - `SLV / slv_momentum_daily`
  - `SLV / slv_momentum_aggressive`
  - `NVDA / nvda_mean_reversion_daily`
  - `AAPL / aapl_momentum_aggressive`
  - `USDJPY / usdjpy_breakout_daily`
  - `SMH / smh_mean_reversion_daily`
  - `XLE / xle_momentum_daily`
  - `GLD / gld_momentum_aggressive`
- Current interpretation:
  - commodity proxies are the strongest rule-based research class so far
  - `USO / uso_momentum_daily` is the current strongest all-around contender on the latest April 1 full run
  - `SLV` momentum remains one of the strongest and most informative research families
  - `XLE` and `GLD` remain valid promoted-paper families because they also screen well historically
  - some stock momentum variants show very high long-window upside but weaker multi-window consistency

### Hourly execution-aware research

- A newer execution-aware branch is now active for short-horizon commodity and ETF contenders.
- Current assumptions include:
  - hourly bars
  - regular session only
  - spread, market impact, stop slippage, and regulatory fee modeling
  - optional no-same-day-reentry rule
- Current trailing-stop sweep result for `SLV / slv_momentum_daily`:
  - best tested hourly stop: `1.5%`
- Current no-same-day-reentry result:
  - helps `SLV / slv_momentum_daily`
  - helps `SLV / slv_momentum_aggressive`
  - hurts `USO / uso_momentum_daily`
  - mildly improves but does not rescue `XLE / xle_momentum_daily`
- Current best hourly execution-aware contender:
  - `SLV / slv_momentum_aggressive`
  - `1.5%` trailing stop
  - `no same-day reentry`
- Current interpretation:
  - `SLV` is the strongest commodity family after adding execution drag
  - `USO` remains viable, but not with the no-same-day-reentry rule
  - `XLE` is not currently a strong hourly contender under realistic assumptions

### Initial ML/DL baseline research

- ML/DL historical research is now active.
- CUDA-backed training is available locally and verified on:
  - `NVIDIA GeForce RTX 3070 Ti`
  - `torch 2.8.0+cu126`
- First execution-aware ML/DL pass completed for:
  - market: `commodities`
  - symbols: `GLD`, `SLV`, `USO`
  - windows: `1y`, `2y`, `3y`
  - timeframe: `hour`
  - regular-hours only
  - trailing stop: `1.5%`
  - no same-day reentry
- Models tested:
  - `logistic_regression`
  - `hist_gradient_boosting`
  - `mlp_classifier`
  - `torch_mlp` on CUDA
- Current result:
  - `SLV` remains the best ML/DL family
  - `slv_logistic_regression_ml` currently leads on composite score
  - `slv_mlp_classifier_ml` currently leads on average total return
  - `slv_torch_mlp_ml` is profitable across `3/3` windows but does not beat the simpler baselines yet
- Current interpretation:
  - the CUDA path works
  - the first GPU model is viable
  - model complexity is not yet producing better research results than the simpler baselines
- Extension pass completed for:
### DeepSeek LLM overlay research

- The daily LLM news overlay has now been rerun with a stronger local model:
  - `U:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B`
- Scope of the rerun:
  - stocks only
  - expanded stock universe:
    - `AAPL`, `MSFT`, `NVDA`, `AMZN`, `META`, `GOOGL`, `TSLA`, `AVGO`
    - `AMD`, `NFLX`, `JPM`, `XOM`, `ORCL`, `CRM`, `WMT`, `COST`
  - windows: `1y`, `2y`, `3y`
  - execution-aware replay
  - stop loss `5%`
  - trailing stop `1.5%`
  - no same-day reentry `false`
- Fixes applied after the earlier local-model pass:
  - causal-model likelihood scoring
  - after-hours and weekend news mapped to the next trading day
  - decayed carry-forward for sparse-news days
  - per-symbol sentiment threshold calibration
- Current trust signal:
  - average bullish label ratio: `39.5%`
  - average bearish label ratio: `51.4%`
  - average neutral label ratio: `9.1%`
  - this is materially better than the earlier positive-skew run
- Current best overlay candidates:
  - `AVGO / avgo_momentum_llm_entry_filter`
  - `NVDA / nvda_momentum_llm_exit_filter`
  - `TSLA / tsla_momentum_llm_exit_filter`
  - `GOOGL / googl_momentum_llm_exit_filter`
  - `CRM / crm_momentum_llm_entry_filter`
  - `JPM / jpm_momentum_llm_exit_filter`
- Overlay-vs-baseline summary across `48` stock-window pairs:
  - entry filter better in `19`
  - exit filter better in `25`
  - combo better in `17`
- Current interpretation:
  - the LLM overlay is now credible enough to continue
  - `exit_filter` is the strongest general overlay type
  - the overlay should remain symbol-selective rather than basket-wide
  - the next LLM step should be hourly follow-up on the strongest daily-stock candidates

Reference artifacts:

- `reports/research/llm_news_sentiment_deepseek_stocks_run.json`
- `reports/research/llm_news_sentiment_deepseek_stocks_report.json`
- `reports/research/llm_news_deepseek_stocks_summary.md`
  - markets: `etfs`, `stocks`
  - symbols:
    - ETFs: `SPY`, `QQQ`, `IWM`, `TLT`, `XLF`, `XLE`, `XLV`, `SMH`
    - Stocks: `AAPL`, `MSFT`, `NVDA`, `AMZN`, `META`
  - same execution-aware assumptions as the commodity ML branch
- Current extension result:
  - strongest stock-side ML family: `NVDA`
  - strongest ETF-side ML family: `IWM` / `hist_gradient_boosting`
  - ETF-side ML is currently weaker than the stock-side and commodity ML branches
  - the CUDA model remains competitive in some names but does not lead the branch overall

### Dedicated DL sequence research

- A separate DL branch is now active.
- Current DL models:
  - `torch_mlp`
  - `torch_lstm`
  - `torch_cnn`
- Current DL scope:
  - markets: `commodities`, `stocks`
  - windows: `1y`, `2y`, `3y`
  - timeframe: `hour`
  - regular-hours only
  - fixed stop loss: `5%`
  - trailing stop: `1.5%`
  - no same-day reentry
- Current DL result:
  - strongest overall DL contender: `SLV / torch_lstm`
  - strongest stock-side DL family: `NVDA`
  - strongest gold-side DL family: `GLD / torch_cnn`
  - strongest oil-side DL family: `USO / torch_lstm`
- Current interpretation:
  - the DL branch is now credible and reproducible
  - `SLV` remains the strongest learned-model family overall
  - DL helps in some symbols, especially `SLV`
  - broadening the DL branch further should wait until direct head-to-head comparisons are complete

### DL seven-window no-rule follow-up

- A new DL rerun is complete with:
  - windows: `1m`, `3m`, `6m`, `9m`, `1y`, `2y`, `3y`
  - timeframe: `hour`
  - regular-hours only
  - trailing stop: `1.5%`
  - no same-day reentry: disabled
- Current result:
  - strongest overall DL contender is now `SLV / torch_cnn`
  - strongest secondary DL contender remains `SLV / torch_lstm`
  - `USO` improved materially in the no-rule branch
- Current interpretation:
  - no same-day reentry is not universally beneficial
  - the DL branch responds differently to that rule than the tuned rule-based `SLV` momentum branch
  - `SLV` remains the clearest DL benchmark family
  - the best overall cross-branch result in the current PDF comparison is `SLV / torch_cnn` with no-same-day reentry disabled

### Torch CNN universe expansion

- `torch_cnn_dl` is now expanded to:
  - ETF universe: `SPY`, `QQQ`, `IWM`, `TLT`, `XLF`, `XLE`, `XLV`, `SMH`
  - FX universe: `EURUSD`, `USDJPY`, `GBPUSD`
- Current ETF-side result:
  - strongest new contender: `SMH / torch_cnn`
  - secondary new contender: `XLE / torch_cnn`
- Current FX-side result:
  - strongest new contender: `USDJPY / torch_cnn`
- Current interpretation:
  - commodities still dominate the DL branch
  - `SMH` is the only newly expanded ETF name that looks worth carrying forward
  - FX remains research-only and weaker under the current DL feature set

### Daily torch_cnn expansion to extended commodities and crypto

- Daily execution-aware `torch_cnn_dl` is now expanded to:
  - extended commodities: `DBC`, `PDBC`, `DBA`, `UNG`
  - crypto: `BTC/USD`, `ETH/USD`, `SOL/USD`
- Current result:
  - best extended-commodity candidate: `PDBC / torch_cnn`
  - best crypto candidate by a small margin: `ETH/USD / torch_cnn`
- Current interpretation:
  - the extended commodity proxies are too flat to matter
  - the crypto branch has no usable edge once realistic crypto fees are applied
  - these new daily branches are not promotion candidates

### Expanded crypto DL basket

- The crypto DL research set is now widened to:
  - `BTC/USD`, `ETH/USD`, `SOL/USD`, `DOGE/USD`, `LTC/USD`, `BCH/USD`, `AVAX/USD`, `LINK/USD`, `UNI/USD`, `AAVE/USD`, `XRP/USD`
- Additional DL models tested:
  - `torch_gru`
  - `torch_transformer`
  - `torch_transformer_gru`
- Current result:
  - strongest single crypto strategy: `BTC/USD / torch_transformer_gru`
  - best average crypto model family: `torch_transformer`
- Current interpretation:
  - the stronger crypto DL stack slightly improves the research branch
  - after realistic crypto fees, the entire crypto basket still remains economically weak
  - crypto stays research-only and out of the promotion path

### Crypto rule-based trend / momentum research

- Crypto is now also tested with the same rule-based strategy families used elsewhere:
  - momentum daily
  - momentum aggressive
  - breakout daily
  - mean reversion daily
- Current setup:
  - daily bars
  - execution-aware replay
  - fixed stop loss: `5%`
  - trailing stop: `1.5%`
  - no same-day reentry: disabled
  - crypto cost model aligned with the DL crypto branch
- Current result:
  - highest-return crypto trend candidate: `SOL/USD / breakout_daily`
  - second highest-return crypto trend candidate: `SOL/USD / momentum_aggressive`
  - most consistent small-edge trend candidates: `LINK/USD / breakout_daily` and `ETH/USD / breakout_daily`
  - best momentum-only consistent candidate: `DOGE/USD / momentum_daily`
- Current interpretation:
  - rule-based crypto trend is materially stronger than the current crypto DL branch
  - the strongest crypto trend winners are unstable across windows and remain regime-dependent
  - the more stable crypto trend winners are close to flat after realistic fees
  - crypto remains research-only and is still outside the promotion path

### Fifteen-year daily DL contender benchmark

- A long-horizon DL benchmark is now complete for:
  - `SLV`
  - `GLD`
  - `USO`
  - `SMH`
  - `XLE`
  - `NVDA`
- Models tested:
  - `torch_cnn`
  - `torch_gru`
  - `torch_transformer`
  - `torch_transformer_gru`
- Current result:
  - strongest long-horizon candidate: `SMH / torch_transformer_gru`
  - strongest long-horizon commodity-like candidate: `GLD / torch_gru`
  - strongest long-horizon stock candidate in this set: `NVDA / torch_cnn`
  - `SLV` is weak on the long daily horizon
- Current interpretation:
  - the best short-horizon DL candidate and best long-horizon DL candidate are not the same
  - `SMH` looks materially better than `SLV` when the horizon is extended to an approximate 15 trading years
  - long-horizon and short-horizon promotion decisions should be tracked separately

### Fixed-split DL benchmark: learn 14 years, apply to last year

- A stricter fixed-split DL benchmark is now complete for:
  - `SLV`
  - `GLD`
  - `USO`
  - `SMH`
  - `XLE`
  - `NVDA`
- Models tested:
  - `torch_cnn`
  - `torch_gru`
  - `torch_transformer`
  - `torch_transformer_gru`
- Setup:
  - train bars: `3528`
  - test bars: `252`
  - timeframe: `day`
  - cost-aware execution replay
  - fixed stop loss: `5%`
  - trailing stop: `1.5%`
  - no same-day reentry: disabled
- Data constraint:
  - a true `14y` train plus `1y` test hourly benchmark was not possible with the available history depth
  - this benchmark therefore uses daily bars with the Yahoo Finance long-history fallback
- Current result:
  - strongest last-year out-of-sample contender: `SLV / torch_transformer`
  - second: `SMH / torch_transformer_gru`
  - strongest overall family by repeated top placements: `GLD`
- Current interpretation:
  - the stricter last-year test does not match the earlier full-window `15y` ranking
  - `SLV` re-emerges as the best last-year DL candidate
  - `GLD` is the most stable long-horizon DL family in the fixed-split view
  - long-horizon DL decisions should prioritize fixed-split out-of-sample tests over full-window walk-forward summaries

### LLM news sentiment overlay research

- A full historical LLM overlay branch is now active for:
  - `SLV`
  - `GLD`
  - `AAPL`
  - `MSFT`
  - `NVDA`
  - `AMZN`
  - `META`
- Engine design:
  - baseline: `20`-day momentum
  - local model: `google/flan-t5-base`
  - historical Alpaca news headlines and summaries
  - news mapped to effective trading days, including after-hours carry
  - short sentiment decay carry for `1-5` day impact modeling
  - execution-aware replay with the standard daily stock/ETF cost model
- Overlay variants:
  - `momentum_llm_entry_filter`
  - `momentum_llm_exit_filter`
  - `momentum_llm_combo`
- Current result:
  - strongest overall overlay winner: `NVDA / momentum_llm_entry_filter`
  - strongest transformation of a weak baseline: `MSFT / momentum_llm_combo`
  - strongest commodity-side overlay: `GLD / momentum_llm_entry_filter`
  - baseline still best on `META`
- Current interpretation:
  - LLM sentiment is useful as a timing overlay, not as a standalone engine
  - the overlay improved baseline returns in multiple stock and gold cases
  - the local model still shows a positive-tone bias, so the branch is promising but not yet production-grade

### DeepSeek LLM overlay expansion

- The daily stock-overlay branch was rerun with:
  - `U:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B`
  - a broadened 16-stock universe
  - per-symbol threshold calibration
  - execution-aware replay
- That branch is now followed by an hourly regular-hours-only rerun on the same 16 stocks.
- Hourly setup:
  - windows: `1y`, `2y`, `3y`
  - timeframe: `hour`
  - regular-hours only
  - fixed stop `5%`
  - trailing stop `1.5%`
  - no same-day reentry: disabled
- Current hourly result:
  - strongest overall rows remain baseline momentum on `AVGO` and `NVDA`
  - `entry_filter` beat baseline in `26/48` symbol-window pairs
  - `combo` beat baseline in `28/48`
  - `exit_filter` beat baseline in `0/48`
  - best overlay per asset beat the baseline on `9/16` assets
- Current hourly interpretation:
  - the LLM overlay still adds value, but mostly through `entry_filter` and `combo`
  - hourly `exit_filter` is currently not a viable default overlay mode
  - the strongest hourly overlay follow-ups are:
    - `GOOGL / combo`
    - `XOM / entry_filter`
    - `AMZN / combo`
    - `AMD / entry_filter`
    - `ORCL / entry_filter`
    - `MSFT / combo`

- DeepSeek daily FX overlay follow-up is now complete on the broader basket:
  - symbols: `EURUSD`, `USDJPY`, `GBPUSD`, `AUDUSD`, `USDCAD`, `USDCHF`, `EURJPY`
  - execution-aware replay
  - proxy-news mapping via currency ETFs and `UUP`
  - summary:
    - `entry_filter` beat baseline in `18/21` symbol-window pairs
    - `combo` beat baseline in `20/21`
    - `exit_filter` beat baseline in `3/21`
  - interpretation:
    - the overlay helps mainly by reducing losses and drawdowns
    - only `EURUSD / combo` is net positive on average across `1y`, `2y`, and `3y`
  - current FX overlay candidates:
    - `EURUSD / combo`
    - `AUDUSD / combo`
    - `USDJPY / entry_filter`
    - `GBPUSD / combo`

- DeepSeek daily commodity metals/energy overlay follow-up is now complete:
  - symbols: `GLD`, `SLV`, `USO`, `UNG`
  - execution-aware replay
  - summary:
    - `entry_filter` beat baseline in `6/12` symbol-window pairs
    - `exit_filter` beat baseline in `5/12`
    - `combo` beat baseline in `5/12`
  - interpretation:
    - `GLD / entry_filter` is the strongest commodity LLM candidate
    - `USO` remains stronger as a baseline momentum branch than as an LLM overlay branch
    - `SLV` improved only modestly and still carries deep drawdown
    - `UNG` is not a serious contender
  - comparison vs large-cap stocks:
    - stocks still have the stronger LLM overlay edge overall
    - commodities are narrower, with `GLD` as the clearest positive overlay case

- DeepSeek daily non-U.S. stock follow-up via Alpaca ADR/global listings is now complete:
  - basket:
    - `ASML`, `TSM`, `NVO`, `SAP`, `BABA`, `PDD`, `HSBC`, `TM`, `RIO`, `SHEL`, `SONY`, `INFY`
  - scope:
    - U.S.-listed ADRs/global stocks through Alpaca
    - not direct foreign exchange listings
  - summary:
    - `entry_filter` beat baseline in `16/36` symbol-window pairs
    - `exit_filter` beat baseline in `18/36`
    - `combo` beat baseline in `17/36`
  - strongest names:
    - `TSM / exit_filter`
    - `BABA / exit_filter`
    - `PDD / exit_filter`
    - `NVO / exit_filter`
  - interpretation:
    - the branch is viable, but weaker than U.S. large-cap stocks
    - the edge is more defensive and lower-volatility
    - `exit_filter` is the strongest mode in this basket

- DeepSeek top-5 U.S. stock exit-rule sweep is now complete:
  - contenders:
    - `AVGO / entry_filter`
    - `NVDA / exit_filter`
    - `TSLA / exit_filter`
    - `GOOGL / exit_filter`
    - `XOM / exit_filter`
  - new rules tested:
    - `profit_lock_only`
    - `profit_lock_partial_25_at_5`
    - `profit_lock_partial_50_at_5`
    - `profit_lock_partial_25_at_8`
    - `profit_lock_partial_50_at_8`
  - interpretation:
    - the aggregate best rule was `profit_lock_partial_50_at_8`, but only by a small margin
    - rule changes are symbol-specific, not uniform
  - current symbol-level decisions:
    - `AVGO`: use `profit_lock_only`
    - `NVDA`: use `profit_lock_partial_50_at_8`
    - `GOOGL`: use `profit_lock_partial_50_at_5`
    - `TSLA`: keep current rule
    - `XOM`: keep current rule
  - follow-up rule-grid sweep completed on the same top 5 names:
    - tested `19,440` execution-aware runs across stop loss, trailing stop, break-even, time stop, cooldown, asymmetric entry gating, sentiment freshness, and weak-cluster blocking
    - aggregate result:
      - mean return improved slightly from `25.79%` to `26.50%`
      - mean max drawdown improved materially from `-12.54%` to `-5.34%`
    - main parameter conclusions:
      - `2%` trailing stop scored best on average
      - `1`-bar cooldown slightly beat `2`
      - `1`-day freshness gave the highest average return; `3`-day freshness was the stronger secondary setting
      - `asymmetric stricter entry` hurt on average
      - `break-even after +3%` had no measurable aggregate effect
      - weak-cluster block only marginally helped overall, but helped `XOM`
    - current best rule by symbol:
      - `AVGO`: `sl_4_trail_1.5_be_off_time_10_cool_1_asym_off_weak_off`
      - `NVDA`: `sl_4_trail_1_be_off_time_10_cool_2_asym_off_weak_off`
      - `GOOGL`: `sl_4_trail_2_be_off_time_10_cool_1_asym_off_weak_off`
      - `XOM`: `sl_4_trail_1.5_be_off_time_10_cool_1_asym_off_weak_on`
      - `TSLA`: `sl_4_trail_1.5_be_off_time_10_cool_2_asym_off_weak_off`
  - shared-policy check completed:
    - best single combined policy across the five names was `sl_4_trail_1.5_be_off_time_10_cool_2_asym_on_weak_off` with freshness `3`
    - this reduced mean drawdown from `-12.54%` to `-4.62%`
    - but it also reduced mean return from `25.79%` to `13.96%`
    - conclusion: keep symbol-specific optimization; do not collapse the branch into one shared daily rule set
  - daily-to-hourly portability check completed:
    - applied the daily top-5 optimized rules to hourly regular-session bars using `1 daily bar = 7 hourly bars`
    - result:
      - mean daily optimized return: `26.50%`
      - mean hourly return with transplanted daily rules: `-1.04%`
      - mean daily optimized drawdown: `-5.34%`
      - mean hourly drawdown with transplanted daily rules: `-10.87%`
    - conclusion: daily rule tuning does not transfer to hourly; keep the hourly branch independently optimized
  - open-source local-model comparison completed for the top 5 U.S. stock branch:
    - downloaded `Qwen3-4B` to `U:\models\Qwen3-4B`
    - downloaded `Qwen3-1.7B` to `U:\models\Qwen3-1.7B`
    - `Qwen3-4B` was not practical on the local `RTX 3070 Ti 8GB` because it required CPU offload
    - completed run used `Qwen3-1.7B`, fully GPU-resident
    - aggregate best-per-symbol comparison:
      - Qwen3 mean return: `34.35%`
      - DeepSeek mean return: `44.44%`
      - Qwen3 mean drawdown: `-7.94%`
      - DeepSeek mean drawdown: `-10.84%`
    - conclusion:
      - DeepSeek remains the stronger primary local LLM for stock-news overlays
      - Qwen3-1.7B is a credible smaller fallback model, but not the new leader

### Recent reproducible research artifacts

- [slv_momentum_daily_hourly_cost_compare.json](C:\SVNProjects\Algoding\reports\research\slv_momentum_daily_hourly_cost_compare.json)
- [slv_momentum_daily_hourly_trailing_sweep.json](C:\SVNProjects\Algoding\reports\research\slv_momentum_daily_hourly_trailing_sweep.json)
- [slv_momentum_daily_hourly_1p5_no_same_day_compare.json](C:\SVNProjects\Algoding\reports\research\slv_momentum_daily_hourly_1p5_no_same_day_compare.json)
- [hourly_no_same_day_reentry_contenders.json](C:\SVNProjects\Algoding\reports\research\hourly_no_same_day_reentry_contenders.json)
- [ml_commodities_hourly_gpu_run.json](C:\SVNProjects\Algoding\reports\research\ml_commodities_hourly_gpu_run.json)
- [ml_etfs_stocks_hourly_gpu_run.json](C:\SVNProjects\Algoding\reports\research\ml_etfs_stocks_hourly_gpu_run.json)
- [dl_commodities_stocks_hourly_gpu_run.json](C:\SVNProjects\Algoding\reports\research\dl_commodities_stocks_hourly_gpu_run.json)
- [dl_commodities_stocks_hourly_gpu_no_reentry_off_7w_run.json](C:\SVNProjects\Algoding\reports\research\dl_commodities_stocks_hourly_gpu_no_reentry_off_7w_run.json)
- [dl_etfs_hourly_torch_cnn_no_reentry_off_7w_run.json](C:\SVNProjects\Algoding\reports\research\dl_etfs_hourly_torch_cnn_no_reentry_off_7w_run.json)
- [dl_fx_daily_torch_cnn_no_reentry_off_supported_run.json](C:\SVNProjects\Algoding\reports\research\dl_fx_daily_torch_cnn_no_reentry_off_supported_run.json)
- [dl_commodities_extended_daily_torch_cnn_no_reentry_off_run.json](C:\SVNProjects\Algoding\reports\research\dl_commodities_extended_daily_torch_cnn_no_reentry_off_run.json)
- [dl_crypto_daily_torch_cnn_no_reentry_off_run.json](C:\SVNProjects\Algoding\reports\research\dl_crypto_daily_torch_cnn_no_reentry_off_run.json)
- [dl_crypto_basket_daily_sota_no_reentry_off_run.json](C:\SVNProjects\Algoding\reports\research\dl_crypto_basket_daily_sota_no_reentry_off_run.json)
- [crypto_rule_based_execution_aware_run.json](C:\SVNProjects\Algoding\reports\research\crypto_rule_based_execution_aware_run.json)
- [dl_contenders_15y_daily_no_reentry_off_run.json](C:\SVNProjects\Algoding\reports\research\dl_contenders_15y_daily_no_reentry_off_run.json)
- [dl_contenders_train14y_test1y_daily_no_reentry_off_run.json](C:\SVNProjects\Algoding\reports\research\dl_contenders_train14y_test1y_daily_no_reentry_off_run.json)
- [llm_news_sentiment_overlay_run.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_overlay_run.json)
- [llm_news_research_summary.md](C:\SVNProjects\Algoding\reports\research\llm_news_research_summary.md)
- [llm_news_sentiment_deepseek_stocks_run.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_stocks_run.json)
- [llm_news_sentiment_deepseek_stocks_report.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_stocks_report.json)
- [llm_news_deepseek_stocks_summary.md](C:\SVNProjects\Algoding\reports\research\llm_news_deepseek_stocks_summary.md)
- [llm_news_sentiment_deepseek_stocks_hourly_run.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_stocks_hourly_run.json)
- [llm_news_deepseek_stocks_hourly_summary.md](C:\SVNProjects\Algoding\reports\research\llm_news_deepseek_stocks_hourly_summary.md)
- [llm_news_sentiment_deepseek_fx_daily_run.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_fx_daily_run.json)
- [llm_news_deepseek_fx_daily_summary.md](C:\SVNProjects\Algoding\reports\research\llm_news_deepseek_fx_daily_summary.md)
- [llm_news_sentiment_deepseek_commodities_daily_run.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_commodities_daily_run.json)
- [llm_news_deepseek_commodities_vs_stocks_summary.md](C:\SVNProjects\Algoding\reports\research\llm_news_deepseek_commodities_vs_stocks_summary.md)
- [llm_news_sentiment_deepseek_non_us_stocks_daily_run.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_non_us_stocks_daily_run.json)
- [llm_news_deepseek_non_us_stocks_summary.md](C:\SVNProjects\Algoding\reports\research\llm_news_deepseek_non_us_stocks_summary.md)
- [llm_deepseek_top5_us_profit_lock_partial_run.json](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_profit_lock_partial_run.json)
- [llm_deepseek_top5_us_profit_lock_partial_summary.md](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_profit_lock_partial_summary.md)
- [llm_deepseek_top5_us_rule_grid_run.json](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_rule_grid_run.json)
- [llm_deepseek_top5_us_rule_grid_summary.md](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_rule_grid_summary.md)
- [llm_deepseek_top5_us_best_combined_vs_previous.md](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_best_combined_vs_previous.md)
- [llm_deepseek_top5_us_hourly_from_daily_rules_run.json](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_hourly_from_daily_rules_run.json)
- [llm_deepseek_top5_us_hourly_vs_daily_summary.md](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_hourly_vs_daily_summary.md)
- [llm_news_sentiment_qwen3_top5_stocks_daily_run.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_qwen3_top5_stocks_daily_run.json)
- [llm_qwen3_top5_vs_deepseek_summary.md](C:\SVNProjects\Algoding\reports\research\llm_qwen3_top5_vs_deepseek_summary.md)
- [deepseek_news_price_meta_plan.md](C:\SVNProjects\Algoding\reports\research\deepseek_news_price_meta_plan.md)
- [deepseek_news_price_meta_run.json](C:\SVNProjects\Algoding\reports\research\deepseek_news_price_meta_full\deepseek_news_price_meta_run.json)
- [deepseek_news_price_meta_summary.md](C:\SVNProjects\Algoding\reports\research\deepseek_news_price_meta_summary.md)
- [deepseek_directional_run.json](C:\SVNProjects\Algoding\reports\research\deepseek_directional_v1\deepseek_directional_run.json)
- [deepseek_directional_summary.md](C:\SVNProjects\Algoding\reports\research\deepseek_directional_v1\deepseek_directional_summary.md)
- [deepseek_top5_strict_exact_vs_meta.json](C:\SVNProjects\Algoding\reports\research\deepseek_top5_strict_exact_vs_meta.json)
- [deepseek_top5_strict_exact_vs_meta_summary.md](C:\SVNProjects\Algoding\reports\research\deepseek_top5_strict_exact_vs_meta_summary.md)
- [deepseek_top5_strict_exact_vs_meta_dark.pdf](C:\SVNProjects\Algoding\output\pdf\deepseek_top5_strict_exact_vs_meta_dark.pdf)
- [deepseek_hourly_overlay_vs_baseline_dark.pdf](C:\SVNProjects\Algoding\output\pdf\deepseek_hourly_overlay_vs_baseline_dark.pdf)
- [dl_research_summary.md](C:\SVNProjects\Algoding\reports\research\dl_research_summary.md)
- [research_strategy_comparison_dark.pdf](C:\SVNProjects\Algoding\output\pdf\research_strategy_comparison_dark.pdf)

## 10. Immediate next actions

1. Keep running basket daily cycle.
2. Keep syncing broker orders and positions.
3. Expand internal ledgers beyond XLE to gold and selected large-cap stocks.
4. Keep the new FX research-only ledgers running in parallel.
5. Keep re-running the historical research leaderboard as new market data arrives.
6. Extend the execution-aware hourly branch to the next stock and FX contenders.
7. Decide whether `SLV / slv_momentum_aggressive` should be promoted to broker paper trading.
8. Compare the best DL candidates directly against the strongest rule-based and ML versions for `SLV`, `NVDA`, and `USO`.
9. Improve the learned-model feature set only after those head-to-head results are logged.
10. Keep the strongest simple ML baseline as the comparison target for future CUDA and LLM work.
11. Compare the LLM overlay branch directly against a non-LLM text baseline such as FinBERT or dictionary sentiment on the same news bundles.
12. Compare the hourly DeepSeek winners directly against non-LLM text baselines such as FinBERT or dictionary sentiment.
13. Decide whether the hourly overlay branch should keep only `entry_filter` and `combo` going forward.
14. Carry the new daily top-5 rule grid into the next hourly DeepSeek stock pass, starting with `AVGO`, `NVDA`, `GOOGL`, `TSLA`, and `XOM`.
15. Improve the fine-tuned DeepSeek news+price meta-model by adding raw prompt sentiment as a fused feature and lowering the probability-entry grid.
16. Run symbol-specific fine-tuned meta-models for `AVGO`, `NVDA`, and `GOOGL` under the same strict split.
17. Treat the exact strict prompt rerun as the stock-news benchmark instead of the mixed-score comparison.
18. Compare the next fine-tuned branch against the exact strict prompt benchmark, not the validation-selected prompt variant.
19. Tighten the directional branch trade mapping with stronger entry strength, relative-to-market confirmation, capped holding periods, and no immediate re-entry.
20. Compare the directional branch against a simple price-only directional classifier before trusting its trading returns.
21. Keep the new commodity `V0 vs V2` comparison as a strict benchmark branch and do not promote the current `V2` commodity results until the trade mapper is tightened.
22. Treat the tightened commodity `V2` branch as the current credible directional commodity benchmark, with `GLD` and `USO` ahead of `SLV` and `UNG`.
23. Treat stock `V3-3` as a loose sensitivity branch only; do not use it as a promotion benchmark.
24. Use stock `V3-4` as the current stricter directional follow-up for `AVGO`, `NVDA`, and `TSLA`.
25. Prioritize the next directional iteration on entry discipline and label quality rather than a larger DeepSeek checkpoint.
26. Treat `V4` as the current clean directional benchmark branch for stocks.
27. Use the `V4` price-only sector model as the baseline to beat before adding more LLM complexity.
28. Do not claim `50%+` directional accuracy has been achieved; it has not.
29. Treat `V4-3` as the current loose-but-credible fused directional follow-up; it is conservative but still below the accuracy target.
30. Treat `V3-5` as another loose sensitivity branch only; `1%` trailing stop plus no-same-day-reentry did not make the stock directional branch credible.
31. Do not use `V3-3` or `V3-5` as directional benchmarks; keep `V3-4` or `V4` for credibility checks.
32. The `V3-5` paper rollout is active as a paper-only validation sleeve on `TSLA`, `AVGO`, and `NVDA`, but it does not upgrade the branch’s research credibility.
33. Keep weekly reporting for the `V3-5` paper sleeve separate from research leaderboards.

34. Treat `V3-5-Hourly` as the current hourly rerun of the loose stock directional branch; it is more believable than daily `V3-5`, but still below the quality bar for a benchmark.
35. Use hourly `AVGO` and `XOM` as the strongest `V3-5-Hourly` cases if the branch is extended further; keep `GOOGL` deprioritized.
36. Do not promote `V3-5-Hourly`; it is a back-test sensitivity result, not a validated trading model.
37. Treat `C_V-0` as the initial crypto directional analogue to `V3-5`; it is cost-aware and crypto-native, but the edge is too small for promotion.
38. Keep `BTC/USD`, `ETH/USD`, and `SOL/USD` as the only validated `C_V-0` evaluation set for now; `DOGE` and `XRP` were not carried forward into the final completed run.

## 11. Publication readiness checklist

Before publishing any result:

- clearly label hypothetical vs paper vs live
- disclose assumptions and limitations
- include drawdown and failure cases
- avoid presenting backtests as guarantees
- maintain reproducible experiment references
