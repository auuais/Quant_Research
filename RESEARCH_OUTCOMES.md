# Historical Research Outcomes

Last updated: 2026-04-08

## Scope

This report compiles the current rule-based historical research results and the initial ML/DL baseline pass.

Markets tested:

- ETFs
- commodity proxies
- large-cap stocks
- major FX pairs

Windows tested:

- `3m`
- `6m`
- `9m`
- `1y`
- `2y`
- `3y`

Risk controls applied in the historical harness:

- fixed stop loss: `5%`
- trailing stop: `7%`

Primary storage:

- PostgreSQL table: `strategy_runs`
- fallback log: `logs/strategy_runs.jsonl`

## Method

The historical harness brute-forces the existing rule-based strategy families across each symbol:

- momentum daily
- momentum aggressive
- mean reversion daily
- breakout daily

Each run records:

- total return
- annualized return
- annualized volatility
- max drawdown
- trades
- win rate
- profit factor
- exposure ratio
- latest signal
- composite score

The leaderboard aggregates the latest run for each `(market, symbol, strategy_name, window)` pair and then scores contenders by:

- average return across windows
- drawdown penalty
- win rate contribution
- profit factor contribution
- consistency across profitable windows

## Top Contenders

### Overall leaderboard

1. `commodities / USO / uso_momentum_daily`
   - profitable windows: `6/6`
   - average total return: `81.24%`
   - average max drawdown: `-13.22%`
   - composite score: `0.958383`
2. `commodities / SLV / slv_momentum_daily`
   - profitable windows: `5/6`
   - average total return: `86.19%`
   - average max drawdown: `-17.40%`
   - composite score: `0.939654`
3. `commodities / SLV / slv_momentum_aggressive`
   - profitable windows: `6/6`
   - average total return: `74.82%`
   - average max drawdown: `-21.18%`
   - composite score: `0.833463`
4. `stocks / NVDA / nvda_mean_reversion_daily`
   - profitable windows: `6/6`
   - average total return: `48.61%`
   - average max drawdown: `-7.41%`
   - composite score: `0.662470`
5. `stocks / AAPL / aapl_momentum_aggressive`
   - profitable windows: `6/6`
   - average total return: `31.98%`
   - average max drawdown: `-5.65%`
   - composite score: `0.498434`
6. `etfs / SMH / smh_mean_reversion_daily`
   - profitable windows: `6/6`
   - average total return: `28.91%`
   - average max drawdown: `-5.72%`
   - composite score: `0.496952`
7. `stocks / NVDA / nvda_momentum_aggressive`
   - profitable windows: `3/6`
   - average total return: `42.21%`
   - average max drawdown: `-11.17%`
   - composite score: `0.471050`
8. `commodities / USO / uso_momentum_aggressive`
   - profitable windows: `6/6`
   - average total return: `38.27%`
   - average max drawdown: `-21.93%`
   - composite score: `0.461701`
9. `commodities / SLV / slv_mean_reversion_daily`
   - profitable windows: `6/6`
   - average total return: `30.99%`
   - average max drawdown: `-12.01%`
   - composite score: `0.461113`
10. `fx / USDJPY / usdjpy_breakout_daily`
   - profitable windows: `6/6`
   - average total return: `20.19%`
   - average max drawdown: `-2.80%`
   - composite score: `0.390785`

## Best By Market

### Commodities

- Best contender: `USO / uso_momentum_daily`
- Best consistency contender: `SLV / slv_momentum_aggressive`
- Additional strong candidate: `SLV / slv_momentum_daily`

### ETFs

- Best contender: `XLE / xle_momentum_daily`
- Secondary contender: `SMH / smh_mean_reversion_daily`

### Stocks

- Best durable contender: `NVDA / nvda_mean_reversion_daily`
- Best momentum contender: `AAPL / aapl_momentum_aggressive`

### FX

- Best contender: `USDJPY / usdjpy_breakout_daily`
- Secondary contender: `USDJPY / usdjpy_momentum_aggressive`

## Interpretation

- The current strongest research class is still `commodity proxies`.
- `USO / uso_momentum_daily` is the current top aggregate contender on the latest April 1 full run.
- `SLV` still dominates the deeper commodity research set with multiple strong variants and remains a high-value strategy family to inspect visually and compare.
- `XLE` and `GLD` remain credible promoted-paper candidates because they perform well both in paper trading workflow and in the historical leaderboard.
- Some stock momentum variants produce very large upside in the `3y` window, but they are less stable across shorter windows and should not be treated as the best all-around contenders yet.
- FX results are positive but smaller in magnitude than the strongest commodity and stock candidates.

## Current Recommendation

Rule-based research should continue in this order:

1. Keep `XLE` and `GLD` in the promoted-paper candidate set.
2. Review `USO / uso_momentum_daily` as the next commodity promotion candidate.
3. Keep `SLV` momentum as a parallel high-priority commodity research family.
4. Keep `AAPL`, `NVDA`, and `USDJPY` as the strongest non-commodity internal contenders.

## Hourly Execution-Aware Follow-up

This section supersedes the earlier optimistic interpretation for short-horizon commodity momentum work.

The recent follow-up used:

- hourly bars
- regular session only
- cost-aware execution replay
- fixed stop loss: `5%`
- tested trailing stops: `1.0%`, `1.5%`, `2.0%`, `3.0%`
- explicit no-same-day-reentry rule as a separate toggle

Execution-aware assumptions:

- commission per order: `0.00`
- quoted spread: `1 bp`
- market impact: `1 bp`
- extra stop slippage: `2 bps`
- partial-fill cap: `4,000` shares per hour
- FINRA TAF: `0.000195` per share, capped at `9.79`
- SEC Section 31 fee: `0.00` for the tested advisory window

Stored experiment references:

- [slv_momentum_daily_hourly_cost_compare.json](C:\SVNProjects\Algoding\reports\research\slv_momentum_daily_hourly_cost_compare.json)
- [slv_momentum_daily_hourly_trailing_sweep.json](C:\SVNProjects\Algoding\reports\research\slv_momentum_daily_hourly_trailing_sweep.json)
- [slv_momentum_daily_hourly_1p5_no_same_day_compare.json](C:\SVNProjects\Algoding\reports\research\slv_momentum_daily_hourly_1p5_no_same_day_compare.json)
- [hourly_no_same_day_reentry_contenders.json](C:\SVNProjects\Algoding\reports\research\hourly_no_same_day_reentry_contenders.json)

### Sweep Result

For `SLV / slv_momentum_daily`, the best tested hourly trailing stop was `1.5%`.

- `1.5%` trailing stop:
  - average total return: `25.16%`
  - average annualized return: `27.12%`
  - average max drawdown: `-20.52%`
  - profitable windows: `6/6`
- `1.0%` trailing stop:
  - average total return: `19.43%`
  - average annualized return: `21.20%`
  - average max drawdown: `-20.09%`
  - profitable windows: `6/6`
- `2.0%` trailing stop:
  - average total return: `8.39%`
  - average annualized return: `1.74%`
  - average max drawdown: `-28.90%`
  - profitable windows: `5/6`
- `3.0%` trailing stop:
  - average total return: `5.58%`
  - average annualized return: `-3.53%`
  - average max drawdown: `-34.90%`
  - profitable windows: `5/6`

### No-Same-Day-Reentry Result

The no-same-day-reentry rule did not generalize uniformly. It helped some strategies and hurt others.

Strong positive effect:

- `SLV / slv_momentum_daily`
  - baseline average total return at `1.5%` trail: `25.16%`
  - with no same-day reentry: `30.51%`
  - drawdown improved from `-20.52%` to `-9.27%`
- `SLV / slv_momentum_aggressive`
  - baseline average total return: `15.19%`
  - with no same-day reentry: `37.98%`
  - drawdown improved from `-21.12%` to `-11.62%`

Negative effect:

- `USO / uso_momentum_daily`
  - baseline average total return: `17.28%`
  - with no same-day reentry: `4.97%`
  - drawdown worsened slightly from `-7.83%` to `-8.35%`

Marginal but still weak:

- `XLE / xle_momentum_daily`
  - baseline average total return: `-7.16%`
  - with no same-day reentry: `-4.48%`
  - still unprofitable across all tested windows

## Updated Interpretation

- The earlier daily-bar leaderboard is still useful for broad screening, but the newer hourly execution-aware research is the more realistic basis for short-horizon contender decisions.
- `SLV` remains the strongest research family after realistic execution drag is added.
- The current best hourly commodity contender is:
  - `SLV / slv_momentum_aggressive`
  - `1.5%` trailing stop
  - `no same-day reentry`
- `USO / uso_momentum_daily` remains viable, but only without the no-same-day-reentry rule.
- `XLE / xle_momentum_daily` is not a strong hourly contender under the current assumptions.

## Updated Recommendation

The current rule-based recommendation is:

1. Keep `SLV` as the top commodity research family.

## DeepSeek LLM Stocks Overlay Rerun

This branch reran the historical LLM news overlay using the local DeepSeek model at:

- `U:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B`

Reference artifacts:

- [llm_news_sentiment_deepseek_stocks_run.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_stocks_run.json)
- [llm_news_sentiment_deepseek_stocks_report.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_stocks_report.json)
- [llm_news_deepseek_stocks_summary.md](C:\SVNProjects\Algoding\reports\research\llm_news_deepseek_stocks_summary.md)

Setup:

- stocks only
- broadened stock universe:
  - `AAPL`, `MSFT`, `NVDA`, `AMZN`, `META`, `GOOGL`, `TSLA`, `AVGO`
  - `AMD`, `NFLX`, `JPM`, `XOM`, `ORCL`, `CRM`, `WMT`, `COST`
- windows: `1y`, `2y`, `3y`
- timeframe: `day`
- execution-aware replay
- fixed stop loss: `5%`
- trailing stop: `1.5%`
- no same-day reentry: `false`

Adjustments relative to the earlier local-model run:

- causal-model likelihood scoring instead of seq2seq-only scoring
- trading-day alignment for after-hours and weekend news
- decayed carry-forward for sparse-news days
- per-symbol threshold calibration from the observed score distribution

The main caveat from the earlier local run was the positive/neutral label bias. The DeepSeek rerun materially improved that. Across the stock universe, the average label mix was:

- bullish: `39.5%`
- bearish: `51.4%`
- neutral: `9.1%`

Top aggregate overlay contenders:

1. `AVGO / avgo_momentum_llm_entry_filter`
   - profitable windows: `3/3`
   - average total return: `62.16%`
   - average max drawdown: `-5.59%`
2. `NVDA / nvda_momentum_llm_exit_filter`
   - profitable windows: `3/3`
   - average total return: `58.71%`
   - average max drawdown: `-15.98%`
3. `TSLA / tsla_momentum_llm_exit_filter`
   - profitable windows: `3/3`
   - average total return: `51.56%`
   - average max drawdown: `-21.70%`
4. `GOOGL / googl_momentum_llm_exit_filter`
   - profitable windows: `3/3`
   - average total return: `33.57%`
   - average max drawdown: `-7.84%`

Overlay versus baseline summary across `48` symbol-window pairs:

- entry filter beat baseline in `19`
- exit filter beat baseline in `25`
- combo beat baseline in `17`

Best symbol-level overlay improvements:

- `AVGO`: `entry_filter`, `+23.22 pts` average total return versus baseline
- `TSLA`: `exit_filter`, `+11.85 pts`
- `CRM`: `entry_filter`, `+8.41 pts`
- `GOOGL`: `exit_filter`, `+5.61 pts`
- `MSFT`: `combo`, `+4.57 pts`
- `JPM`: `exit_filter`, `+4.44 pts`

Clear weak fits:

- `META`
- `AMD`
- `ORCL`
- `WMT`
- `NFLX`

Interpretation:

- the DeepSeek rerun is more trustworthy than the earlier local-model run because the label bias is much lower
- `exit_filter` is the most reliable overlay type in this branch
- the LLM overlay should still be applied selectively by symbol
- the best daily-stock candidates for the later hourly follow-up are:
  - `AVGO / entry_filter`
  - `NVDA / exit_filter`
  - `TSLA / exit_filter`
  - `GOOGL / exit_filter`
  - `CRM / entry_filter`
  - `JPM / exit_filter`
2. Use `1.5%` trailing stop as the current hourly default for `SLV` momentum testing.
3. Keep `no same-day reentry` enabled for `SLV` momentum variants.
4. Do not apply the same no-same-day-reentry rule to `USO / uso_momentum_daily`.
5. Treat `XLE` as a paper/live monitoring candidate only, not as the strongest hourly research contender.

## Caveats

- These are hypothetical historical results, not live profits.
- The broader leaderboard still reflects the older daily-bar screening layer.
- The newer hourly branch uses more realistic execution assumptions, but it is still a model.
- Stop execution on hourly bars remains assumption-sensitive.
- The current leaderboard is suitable for screening and ranking contenders, not for making final live-capital claims.
- Promotion decisions should still combine:
  - historical consistency
  - internal parallel-ledger behavior
  - broker paper behavior

## Next Steps

1. Re-run the cost-aware hourly branch routinely for `SLV`, `USO`, and the next stock contender.
2. Promote only contenders that survive execution-aware testing, not just daily-bar screening.
3. Add benchmark comparisons and more formal out-of-sample reporting.
4. Continue ML/DL baseline experiments only where they improve on the rule-based screening layer.

## Initial ML/DL Baseline

This first ML/DL pass was run after the rule-based hourly execution-aware branch was stable enough to serve as a benchmark.

Setup used:

- market: `commodities`
- symbols: `GLD`, `SLV`, `USO`
- windows: `1y`, `2y`, `3y`
- timeframe: `hour`
- regular session only
- fixed stop loss: `5%`
- trailing stop: `1.5%`
- no same-day reentry: enabled
- execution-aware replay with the same spread, market impact, stop slippage, and fee assumptions as the rule-based hourly branch

Models tested:

- `logistic_regression`
- `hist_gradient_boosting`
- `mlp_classifier`
- `torch_mlp` on CUDA

Stored experiment reference:

- [ml_commodities_hourly_gpu_run.json](C:\SVNProjects\Algoding\reports\research\ml_commodities_hourly_gpu_run.json)

### ML/DL Baseline Result

Top initial ML leaderboard:

1. `commodities / SLV / slv_logistic_regression_ml`
   - profitable windows: `3/3`
   - average total return: `21.00%`
   - average annualized return: `14.19%`
   - average max drawdown: `-12.32%`
2. `commodities / SLV / slv_mlp_classifier_ml`
   - profitable windows: `3/3`
   - average total return: `25.91%`
   - average annualized return: `15.24%`
   - average max drawdown: `-10.31%`
3. `commodities / SLV / slv_torch_mlp_ml`
   - profitable windows: `3/3`
   - average total return: `17.32%`
   - average annualized return: `11.99%`
   - average max drawdown: `-9.47%`

Key interpretation:

- The CUDA path is working and reproducible.
- The first GPU-backed deep-learning baseline did `not` beat the simpler `SLV` baselines.
- On the current feature set, `logistic_regression` leads on composite score and `mlp_classifier` leads on average total return.
- The `torch_mlp` result is respectable on `SLV`, but weak on `GLD` and `USO`.
- This is the correct outcome to log: the current data/features matter more than model complexity.

### Updated Recommendation

For the ML/DL branch:

1. Keep `SLV` as the first ML/DL benchmark family.
2. Treat `logistic_regression` and `mlp_classifier` as the current baselines to beat.
3. Keep `torch_mlp` in the stack, but do not assume GPU training adds edge by itself.
4. Expand ML/DL research next by:
   - improving features
   - testing ETFs and stocks on the same execution-aware path
   - comparing ML outputs directly against the strongest rule-based contender for the same symbol

## ML/DL Extension To ETFs And Stocks

The same execution-aware setup was then extended to:

- markets: `etfs`, `stocks`
- symbols:
  - ETFs: `SPY`, `QQQ`, `IWM`, `TLT`, `XLF`, `XLE`, `XLV`, `SMH`
  - Stocks: `AAPL`, `MSFT`, `NVDA`, `AMZN`, `META`
- windows: `1y`, `2y`, `3y`
- timeframe: `hour`
- regular session only
- fixed stop loss: `5%`
- trailing stop: `1.5%`
- no same-day reentry: enabled

Stored experiment reference:

- [ml_etfs_stocks_hourly_gpu_run.json](C:\SVNProjects\Algoding\reports\research\ml_etfs_stocks_hourly_gpu_run.json)

### Result

Top extension findings:

1. `stocks / NVDA / nvda_mlp_classifier_ml`
   - profitable windows: `2/3`
   - average total return: `22.52%`
   - average max drawdown: `-10.39%`
2. `stocks / NVDA / nvda_hist_gradient_boosting_ml`
   - profitable windows: `3/3`
   - average total return: `12.22%`
   - average max drawdown: `-13.72%`
3. `stocks / NVDA / nvda_torch_mlp_ml`
   - profitable windows: `2/3`
   - average total return: `11.96%`
   - average max drawdown: `-11.05%`
4. `stocks / META / meta_hist_gradient_boosting_ml`
   - profitable windows: `3/3`
   - average total return: `8.57%`
   - average max drawdown: `-10.45%`
5. Best ETF-side ML contender:
   - `etfs / IWM / iwm_hist_gradient_boosting_ml`
   - profitable windows: `2/3`
   - average total return: `9.85%`
   - average max drawdown: `-9.29%`

### Interpretation

- The strongest ML/DL stock family right now is `NVDA`.
- `META` also shows a usable ML signal, but weaker than `NVDA`.
- The ETF basket is materially weaker under the current ML feature set than both:
  - the commodity ML branch
  - the rule-based ETF screening branch
- The CUDA model remains viable but still does not lead the leaderboard.

### Updated ML/DL Recommendation

1. Keep `SLV` as the primary ML/DL benchmark family.
2. Add `NVDA` as the first stock-side ML optimization target.
3. Keep `META` as a secondary stock-side ML candidate.
4. Do not prioritize ETF-side ML optimization yet.
5. Focus next on feature engineering and direct rule-based vs ML comparisons for:
   - `SLV`
   - `NVDA`

## Dedicated DL Branch

The project now has a separate CUDA-backed deep-learning research branch using:

- `torch_mlp`
- `torch_lstm`
- `torch_cnn`

Setup used:

- markets: `commodities`, `stocks`
- symbols:
  - Commodities: `GLD`, `SLV`, `USO`
  - Stocks: `AAPL`, `MSFT`, `NVDA`, `AMZN`, `META`
- windows: `1y`, `2y`, `3y`
- timeframe: `hour`
- regular-hours only
- fixed stop loss: `5%`
- trailing stop: `1.5%`
- no same-day reentry
- execution-aware replay with the same cost assumptions as the hourly rule-based and ML branches

Stored artifacts:

- [dl_commodities_stocks_hourly_gpu_run.json](C:\SVNProjects\Algoding\reports\research\dl_commodities_stocks_hourly_gpu_run.json)
- [dl_research_summary.md](C:\SVNProjects\Algoding\reports\research\dl_research_summary.md)

### DL Results

Top dedicated DL contenders:

1. `commodities / SLV / slv_torch_lstm_dl`
   - profitable windows: `3/3`
   - average total return: `26.64%`
   - average annualized return: `19.49%`
   - average max drawdown: `-10.98%`
2. `stocks / NVDA / nvda_torch_mlp_dl`
   - profitable windows: `2/3`
   - average total return: `22.46%`
   - average max drawdown: `-7.23%`
   - standout `3y` total return: `62.74%`
3. `commodities / SLV / slv_torch_cnn_dl`
   - profitable windows: `3/3`
   - average total return: `18.56%`
   - average max drawdown: `-12.82%`
4. `stocks / NVDA / nvda_torch_lstm_dl`
   - profitable windows: `3/3`
   - average total return: `12.18%`
   - average max drawdown: `-12.65%`

### DL Interpretation

- `SLV / torch_lstm` is the strongest dedicated DL result so far.
- `NVDA` remains the strongest stock-side learned-model family.
- The DL branch improves on the earlier learned baselines in some symbols, but not across the board.
- This is now a valid research branch, not a placeholder.

### Updated Recommendation

1. Keep `SLV / torch_lstm` as the lead DL contender.
2. Keep `NVDA / torch_mlp` and `NVDA / torch_lstm` as the lead stock-side DL contenders.
3. Compare DL directly against the best rule-based and ML contenders before promotion.
4. Do not broaden the DL branch further until those head-to-head comparisons are complete.

## DL Seven-Window Rerun Without No-Same-Day-Reentry

The dedicated DL branch was then rerun with:

- windows: `1m`, `3m`, `6m`, `9m`, `1y`, `2y`, `3y`
- timeframe: `hour`
- regular-hours only
- fixed stop loss: `5%`
- trailing stop: `1.5%`
- no same-day reentry: disabled
- the same execution-aware spread, slippage, fee, and partial-fill assumptions as the earlier hourly branch

Stored artifacts:

- [dl_commodities_stocks_hourly_gpu_no_reentry_off_7w_run.json](C:\SVNProjects\Algoding\reports\research\dl_commodities_stocks_hourly_gpu_no_reentry_off_7w_run.json)
- [research_strategy_comparison_dark.pdf](C:\SVNProjects\Algoding\output\pdf\research_strategy_comparison_dark.pdf)
- [research_strategy_comparison_dark.json](C:\SVNProjects\Algoding\output\pdf\research_strategy_comparison_dark.json)

### Result

Top no-rule DL contenders:

1. `commodities / SLV / slv_torch_cnn_dl`
   - profitable windows: `5/7`
   - average total return: `44.00%`
   - average annualized return: `41.83%`
   - average max drawdown: `-14.87%`
2. `commodities / SLV / slv_torch_lstm_dl`
   - profitable windows: `5/7`
   - average total return: `24.69%`
   - average annualized return: `15.35%`
   - average max drawdown: `-20.26%`
3. `commodities / SLV / slv_torch_mlp_dl`
   - profitable windows: `5/7`
   - average total return: `21.77%`
   - average annualized return: `17.43%`
   - average max drawdown: `-12.18%`
4. `commodities / USO / uso_torch_lstm_dl`
   - profitable windows: `6/7`
   - average total return: `24.43%`
   - average annualized return: `60.66%`
   - average max drawdown: `-9.18%`
5. `commodities / USO / uso_torch_cnn_dl`
   - profitable windows: `7/7`
   - average total return: `22.11%`
   - average annualized return: `97.23%`
   - average max drawdown: `-7.14%`

### Comparison To The Earlier No-Same-Day Version

- Disabling no-same-day reentry slightly improved the average DL report score across the matched strategy set.
- The biggest improvement was:
  - `SLV / slv_torch_cnn_dl`
  - average total return rose from `18.56%` to `44.00%`
- Other strong improvements:
  - `USO / uso_torch_mlp_dl`
    - from `-5.65%` to `14.05%`
  - `USO / uso_torch_cnn_dl`
    - from `6.89%` to `22.11%`
  - `SLV / slv_torch_mlp_dl`
    - from `8.39%` to `21.77%`

### Interpretation

- The new best dedicated DL contender is now `SLV / torch_cnn`, not `SLV / torch_lstm`.
- The no-same-day-reentry rule appears to be helpful for the tuned hourly rule-based `SLV` branch, but not automatically helpful for the DL branch.
- `SLV` remains the strongest learned-model family overall.
- `USO` improved materially in the no-rule DL run and remains the second-strongest commodity DL family.
- `NVDA` is still the main stock-side learned-model family, but commodities remain stronger under the current feature set.

### Updated Recommendation

1. Treat `SLV / torch_cnn` as the new lead DL contender.
2. Keep `SLV / torch_lstm` and `USO / torch_lstm` as secondary DL contenders.
3. Do not assume the no-same-day-reentry rule should carry from rule-based momentum into DL automatically.
4. Use the comparison PDF as the current cross-branch summary before moving to direct symbol-level head-to-head tests.

## Torch CNN Expansion To Untested Symbols

The `torch_cnn_dl` branch was expanded to the untested liquid symbols outside the earlier commodity and stock DL scope.

Stored artifacts:

- [dl_etfs_hourly_torch_cnn_no_reentry_off_7w_run.json](C:\SVNProjects\Algoding\reports\research\dl_etfs_hourly_torch_cnn_no_reentry_off_7w_run.json)
- [dl_fx_daily_torch_cnn_no_reentry_off_supported_run.json](C:\SVNProjects\Algoding\reports\research\dl_fx_daily_torch_cnn_no_reentry_off_supported_run.json)

### ETF Expansion

Setup:

- market: `etfs`
- symbols: `SPY`, `QQQ`, `IWM`, `TLT`, `XLF`, `XLE`, `XLV`, `SMH`
- windows: `1m`, `3m`, `6m`, `9m`, `1y`, `2y`, `3y`
- timeframe: `hour`
- regular-hours only
- fixed stop loss: `5%`
- trailing stop: `1.5%`
- no same-day reentry: disabled
- same execution-aware cost model as the current DL branch

Top ETF `torch_cnn_dl` results:

1. `SMH / smh_torch_cnn_dl`
   - profitable windows: `5/7`
   - average total return: `12.16%`
   - average annualized return: `6.59%`
   - average max drawdown: `-10.84%`
2. `XLE / xle_torch_cnn_dl`
   - profitable windows: `5/7`
   - average total return: `5.92%`
   - average annualized return: `12.36%`
   - average max drawdown: `-3.84%`
3. `QQQ / qqq_torch_cnn_dl`
   - profitable windows: `3/7`
   - average total return: `3.92%`
   - average max drawdown: `-6.08%`

Interpretation:

- `SMH` is the only ETF-side `torch_cnn_dl` result that looks genuinely interesting.
- `XLE` is positive and relatively shallow in drawdown, but much weaker than the top commodity DL branch.
- The rest of the ETF basket is weak or flat under the current feature set and assumptions.

### FX Expansion

Setup:

- market: `fx`
- symbols: `EURUSD`, `USDJPY`, `GBPUSD`
- windows: `6m`, `9m`, `1y`, `2y`, `3y`
- timeframe: `day`
- no same-day reentry: disabled
- same risk and execution-aware replay framework

Important limitation:

- `1m` and `3m` could not be used for the daily FX DL path because the sequence model requires more labeled samples than those windows can provide.

Top FX `torch_cnn_dl` results:

1. `USDJPY / usdjpy_torch_cnn_dl`
   - profitable windows: `3/5`
   - average total return: `5.59%`
   - average annualized return: `5.70%`
   - average max drawdown: `-2.03%`
2. `GBPUSD / gbpusd_torch_cnn_dl`
   - profitable windows: `3/5`
   - average total return: `2.41%`
   - average max drawdown: `-0.70%`
3. `EURUSD / eurusd_torch_cnn_dl`
   - profitable windows: `2/5`
   - average total return: `0.58%`
   - average max drawdown: `-0.95%`

Interpretation:

- FX is currently much weaker than commodities and the best ETF-side candidate.
- `USDJPY` is the only FX result worth keeping on the DL watchlist.
- The current DL feature set appears materially better suited to commodity proxies than to FX.

### Updated Recommendation

1. Keep `SLV / torch_cnn` as the lead DL contender overall.
2. Add `SMH / torch_cnn` as the best newly expanded ETF-side DL candidate.
3. Keep `USDJPY / torch_cnn` as a research-only FX DL watchlist candidate, not a promotion candidate.
4. Do not broaden the DL universe again until `SLV`, `SMH`, and `USDJPY` are compared head-to-head with their rule-based and ML baselines.

## Daily Torch CNN Expansion To Extended Commodities And Crypto

The next `torch_cnn_dl` expansion moved to daily execution-aware bars instead of hourly for:

- unexplored commodity proxies
- crypto spot pairs

Stored artifacts:

- [dl_commodities_extended_daily_torch_cnn_no_reentry_off_run.json](C:\SVNProjects\Algoding\reports\research\dl_commodities_extended_daily_torch_cnn_no_reentry_off_run.json)
- [dl_crypto_daily_torch_cnn_no_reentry_off_run.json](C:\SVNProjects\Algoding\reports\research\dl_crypto_daily_torch_cnn_no_reentry_off_run.json)

### Extended Commodities Daily Run

Setup:

- market: `commodities_extended`
- symbols: `DBC`, `PDBC`, `DBA`, `UNG`
- windows: `6m`, `9m`, `1y`, `2y`, `3y`
- timeframe: `day`
- fixed stop loss: `5%`
- trailing stop: `1.5%`
- no same-day reentry: disabled
- same execution-aware stock/ETF-style cost model as the rest of the daily non-crypto branch

Top results:

1. `PDBC / pdbc_torch_cnn_dl`
   - profitable windows: `5/5`
   - average total return: `0.36%`
   - average annualized return: `0.54%`
   - average max drawdown: `-0.15%`
2. `DBC / dbc_torch_cnn_dl`
   - profitable windows: `5/5`
   - average total return: `0.05%`
   - average annualized return: `0.09%`
   - average max drawdown: `-0.04%`

Interpretation:

- The newly added commodity proxies are effectively flat under the current DL feature set.
- `PDBC` is the cleanest of the group, but the edge is too small to matter.
- This branch does not challenge the existing `SLV` or `USO` hourly DL leaders.

### Crypto Daily DL Basket

The crypto DL branch was expanded beyond the initial `torch_cnn` baseline into a wider basket and stronger sequence models.

Stored artifact:

- [dl_crypto_basket_daily_sota_no_reentry_off_run.json](C:\SVNProjects\Algoding\reports\research\dl_crypto_basket_daily_sota_no_reentry_off_run.json)

Setup:

- market: `crypto`
- symbols:
  - `BTC/USD`
  - `ETH/USD`
  - `SOL/USD`
  - `DOGE/USD`
  - `LTC/USD`
  - `BCH/USD`
  - `AVAX/USD`
  - `LINK/USD`
  - `UNI/USD`
  - `AAVE/USD`
  - `XRP/USD`
- windows: `6m`, `9m`, `1y`, `2y`, `3y`
- timeframe: `day`
- fixed stop loss: `5%`
- trailing stop: `1.5%`
- no same-day reentry: disabled

DL models tested:

- `torch_cnn`
- `torch_gru`
- `torch_transformer`
- `torch_transformer_gru`

Model choice rationale:

- Transformer-based crypto forecasting has been actively studied in recent literature, including comparisons against LSTM/GRU baselines: [Enhancing Price Prediction in Cryptocurrency Using Transformer](https://arxiv.org/pdf/2403.03606)
- The Transformer-GRU hybrid in this repo is an engineering adaptation motivated by the same sequence-model family, intended as a stronger crypto-specific test than plain CNN alone.

Crypto execution-aware assumptions:

- Alpaca crypto taker fee modeled at `0.25%` per side, based on Alpaca's current tier-1 crypto fee schedule: [Alpaca Crypto Spot Trading Fees](https://docs.alpaca.markets/docs/crypto-fees)
- SEC and FINRA sell-side fees disabled for crypto
- quoted spread `3 bps`, market impact `2 bps`, extra stop slippage `6 bps`
  - these spread and slippage values are inference-based modeling assumptions, not quoted Alpaca guarantees

Top results:

1. `BTC/USD / btc/usd_torch_transformer_gru_dl`
   - profitable windows: `3/5`
   - average total return: `0.04%`
   - average max drawdown: `-0.09%`
2. `BCH/USD / bch/usd_torch_transformer_gru_dl`
   - profitable windows: `2/5`
   - average total return: `0.01%`
   - average max drawdown: `-0.01%`
3. `DOGE/USD / doge/usd_torch_transformer_dl`
   - profitable windows: `1/5`
   - average total return: effectively flat
   - average max drawdown: effectively flat

Cross-model summary:

- best average model family by composite score: `torch_transformer`
- strongest single strategy: `BTC/USD / torch_transformer_gru`
- every model family still averaged slightly negative total return across the full basket after fees

Interpretation:

- The stronger crypto-specific DL stack did better than the earlier three-pair `torch_cnn` test, but not by enough to matter economically.
- The best crypto result is still too small relative to fee drag and model risk.
- Crypto remains research-only and is still not a promotion candidate under the current feature set.

### Updated Recommendation

1. Do not promote the extended commodity daily DL branch.
2. Do not promote the current crypto daily DL branch.
3. Keep `SLV / torch_cnn` as the lead DL strategy and `SMH / torch_cnn` as the best newly expanded non-commodity side candidate.
4. If crypto remains a goal, it should move only after feature engineering, not after more brute-force reruns of the same feature set.

## Fifteen-Year Daily DL Benchmark For Prior Contenders

The strongest earlier non-crypto DL contenders were rerun on a long daily window using the four requested sequence models:

- `torch_cnn`
- `torch_gru`
- `torch_transformer`
- `torch_transformer_gru`

Stored artifact:

- [dl_contenders_15y_daily_no_reentry_off_run.json](C:\SVNProjects\Algoding\reports\research\dl_contenders_15y_daily_no_reentry_off_run.json)

Contender set:

- `SLV`
- `GLD`
- `USO`
- `SMH`
- `XLE`
- `NVDA`

Setup:

- window: `15y`
- timeframe: `day`
- cost-aware execution replay
- fixed stop loss: `5%`
- trailing stop: `1.5%`
- no same-day reentry: disabled

Long-history note:

- The current Alpaca source does not expose this depth in the workspace environment.
- For this run I used a Yahoo Finance fallback to obtain `3780` daily bars per symbol, which corresponds to an approximate `15` trading-year window.
- Actual window used for all six symbols: `2011-03-21` through `2026-03-31`

### Best Results

1. `SMH / smh_torch_transformer_gru_dl`
   - total return: `149.66%`
   - annualized return: `6.37%`
   - max drawdown: `-18.19%`
2. `SMH / smh_torch_transformer_dl`
   - total return: `130.02%`
   - annualized return: `5.78%`
   - max drawdown: `-23.59%`
3. `NVDA / nvda_torch_cnn_dl`
   - total return: `72.35%`
   - annualized return: `3.74%`
   - max drawdown: `-35.94%`
4. `XLE / xle_torch_cnn_dl`
   - total return: `50.72%`
   - annualized return: `2.81%`
   - max drawdown: `-7.89%`
5. `GLD / gld_torch_gru_dl`
   - total return: `50.46%`
   - annualized return: `2.79%`
   - max drawdown: `-8.92%`

### Important Losers

- `SLV` performed poorly across all four long-window DL models:
  - best `SLV` result: `slv_torch_transformer_gru_dl`
  - total return: `-0.23%`
- `USO` was mixed to poor:
  - best `USO` result: `uso_torch_transformer_gru_dl`
  - total return: `19.32%`
  - weakest `USO` transformer result: `-51.01%`

### Interpretation

- The short-horizon hourly DL branch and the long-horizon daily DL branch do not agree on the same winners.
- On the long daily horizon:
  - `SMH` is the clearest winner
  - `GLD` and `XLE` are respectable
  - `SLV` is not
- This suggests `SLV` is more of a short-horizon pattern family in the current DL setup, while `SMH` is stronger as a long-horizon daily DL candidate.

### Updated Recommendation

1. Keep `SLV / torch_cnn` as the lead short-horizon DL candidate.
2. Add `SMH / torch_transformer_gru` as the new lead long-horizon daily DL candidate.
3. Keep `GLD / torch_gru` and `XLE / torch_cnn` as secondary long-horizon daily DL candidates.
4. Do not treat long-horizon and short-horizon DL leaders as interchangeable; they are learning different regimes.

## Fourteen-Year Train / One-Year Test DL Benchmark

The earlier `15y` daily DL benchmark was a full-window walk-forward style result. The newer run below is stricter:

- train on the first approximate `14` trading years
- apply the trained model to the last approximate `1` trading year only
- no retraining during the final test year

Stored artifact:

- [dl_contenders_train14y_test1y_daily_no_reentry_off_run.json](C:\SVNProjects\Algoding\reports\research\dl_contenders_train14y_test1y_daily_no_reentry_off_run.json)

Contender set:

- `SLV`
- `GLD`
- `USO`
- `SMH`
- `XLE`
- `NVDA`

Models tested:

- `torch_cnn`
- `torch_gru`
- `torch_transformer`
- `torch_transformer_gru`

Setup:

- train bars: `3528`
- test bars: `252`
- timeframe: `day`
- cost-aware execution replay
- fixed stop loss: `5%`
- trailing stop: `1.5%`
- no same-day reentry: disabled

Data-depth note:

- A true `14y` train plus `1y` test run on hourly bars was not possible with the available history depth in the current data sources.
- This benchmark therefore uses daily bars with the Yahoo Finance long-history fallback.

### Last-Year Test Ranking

1. `SLV / slv_torch_transformer_dl`
   - last-year total return: `26.36%`
   - annualized return: `26.47%`
   - max drawdown: `-11.22%`
   - composite score: `0.404847`
2. `SMH / smh_torch_transformer_gru_dl`
   - last-year total return: `16.80%`
   - annualized return: `16.87%`
   - max drawdown: `-3.49%`
   - composite score: `0.367684`
3. `GLD / gld_torch_transformer_gru_dl`
   - last-year total return: `21.78%`
   - annualized return: `21.88%`
   - max drawdown: `-10.62%`
   - composite score: `0.365839`
4. `GLD / gld_torch_cnn_dl`
   - last-year total return: `22.52%`
   - annualized return: `22.62%`
   - max drawdown: `-9.83%`
   - composite score: `0.363286`
5. `GLD / gld_torch_transformer_dl`
   - last-year total return: `15.62%`
   - annualized return: `15.69%`
   - max drawdown: `-6.97%`
   - composite score: `0.336627`

### Interpretation

- The stricter fixed-split last-year test does not agree with the earlier full-window `15y` ranking.
- `SLV` was weak in the full-window long-horizon benchmark, but it is the strongest last-year out-of-sample DL contender in this stricter split.
- `SMH` remains strong, but it drops from the top spot to second on the last-year test.
- `GLD` becomes the most consistently strong family overall in this fixed-split view because three separate DL models rank near the top.

### Updated Recommendation

1. Treat the fixed-split `14y -> 1y` benchmark as the more decision-relevant long-horizon DL test.
2. Carry forward `SLV / torch_transformer`, `SMH / torch_transformer_gru`, and `GLD / torch_transformer_gru` as the strongest daily DL contenders.
3. Keep the earlier full-window `15y` result only as a secondary context check, not the primary ranking.

## Crypto Rule-Based Trend / Momentum Research

The crypto market had not yet been explored with the same rule-based momentum and breakout families used in the rest of the research stack. That gap is now closed with a cost-aware replay run.

Stored artifact:

- [crypto_rule_based_execution_aware_run.json](C:\SVNProjects\Algoding\reports\research\crypto_rule_based_execution_aware_run.json)

Setup:

- market: `crypto`
- symbols:
  - `BTC/USD`
  - `ETH/USD`
  - `SOL/USD`
  - `DOGE/USD`
  - `LTC/USD`
  - `BCH/USD`
  - `AVAX/USD`
  - `LINK/USD`
  - `UNI/USD`
  - `AAVE/USD`
  - `XRP/USD`
- windows: `6m`, `9m`, `1y`, `2y`, `3y`
- timeframe: `day`
- execution-aware replay
- fixed stop loss: `5%`
- trailing stop: `1.5%`
- no same-day reentry: disabled

Crypto execution assumptions matched the DL crypto branch for comparability:

- buy fee: `25 bps`
- sell fee: `25 bps`
- quoted spread: `3 bps`
- market impact: `2 bps`
- extra stop slippage: `6 bps`
- SEC / FINRA sell fees disabled for crypto

### Best Crypto Trend / Momentum Results

Highest-return but unstable winners:

1. `SOL/USD / sol/usd_breakout_daily`
   - profitable windows: `2/5`
   - average total return: `67.77%`
   - average max drawdown: `-14.05%`
   - best single window: `+339.33%`
2. `SOL/USD / sol/usd_momentum_aggressive`
   - profitable windows: `1/5`
   - average total return: `63.94%`
   - average max drawdown: `-13.78%`
   - best single window: `+320.24%`

More stable but much smaller winners:

1. `LINK/USD / link/usd_breakout_daily`
   - profitable windows: `5/5`
   - average total return: `0.12%`
   - average max drawdown: `-0.07%`
2. `ETH/USD / eth/usd_breakout_daily`
   - profitable windows: `4/5`
   - average total return: `0.12%`
   - average max drawdown: `-0.09%`
3. `DOGE/USD / doge/usd_momentum_daily`
   - profitable windows: `4/5`
   - average total return: `2.62%`
   - average max drawdown: `-2.70%`

### Interpretation

- Rule-based crypto trend/momentum performs better than the current crypto DL branch.
- But the edge is not broad or clean across the basket.
- `SOL` produced the strongest returns, but only in `1` to `2` windows, which means the result is regime-dependent rather than robust.
- The more consistent winners such as `LINK` and `ETH` are effectively flat after realistic crypto fees.
- `DOGE` showed some positive momentum behavior, but it is still modest and not yet strong enough for promotion.

### Comparison To Crypto DL

The best rule-based crypto result clearly beat the best current crypto DL result:

- best rule-based: `SOL/USD / breakout_daily`
  - average total return: `67.77%`
  - profitable windows: `2/5`
- best DL: `BTC/USD / torch_transformer_gru`
  - average total return: about `0.04%`
  - profitable windows: `3/5`

That comparison is directionally useful, but the practical conclusion is still conservative:

- rule-based crypto trend is more promising than our current crypto DL feature set
- the current crypto trend winners are still too unstable to treat as promotion candidates

### Updated Recommendation

1. Keep crypto as research-only.
2. Treat `SOL` breakout and aggressive momentum as speculative follow-up candidates, not promotion candidates.
3. If crypto remains a priority, the next correct step is a crypto-specific robustness pass:
   - more windows
   - separate bull / bear / sideways regime slices
   - direct head-to-head against buy-and-hold

## LLM News Sentiment Overlay Research

The first full historical LLM overlay branch is now implemented and tested on:

- `SLV`
- `GLD`
- `AAPL`
- `MSFT`
- `NVDA`
- `AMZN`
- `META`

Stored artifacts:

- [llm_news_sentiment_overlay_run.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_overlay_run.json)
- [llm_news_research_summary.md](C:\SVNProjects\Algoding\reports\research\llm_news_research_summary.md)

### Engine Summary

- baseline: `20`-day rule-based momentum
- local LLM: `google/flan-t5-base`
- data: Alpaca historical news headlines and summaries
- news alignment:
  - after-hours and weekend headlines mapped to the next trading day
  - short decay carry to reflect a `1-5` day impact assumption
- overlays tested:
  - `momentum_llm_entry_filter`
  - `momentum_llm_exit_filter`
  - `momentum_llm_combo`
- execution-aware replay:
  - daily bars
  - fixed stop loss `5%`
  - trailing stop `1.5%`
  - spread, market impact, stop slippage, and regulatory fee assumptions aligned with the stock/ETF replay path

### Best Results

Best overlay winners by symbol:

- `NVDA / nvda_momentum_llm_entry_filter`
  - average total return: `56.95%`
  - average max drawdown: `-13.24%`
- `MSFT / msft_momentum_llm_combo`
  - average total return: `18.18%`
  - average max drawdown: `-3.97%`
- `GLD / gld_momentum_llm_entry_filter`
  - average total return: `4.74%`
  - average max drawdown: `-5.46%`
- `AAPL / aapl_momentum_llm_exit_filter`
  - average total return: `8.28%`
  - average max drawdown: `-8.69%`
- `SLV / slv_momentum_llm_exit_filter`
  - average total return: `5.44%`
  - average max drawdown: `-25.81%`

Baseline remained best for:

- `META`

### Improvement Versus Baseline

Across `21` symbol-window pairs:

- entry filter beat baseline in `10`
- exit filter beat baseline in `11`
- combo beat baseline in `9`

Most important positive examples:

- `MSFT`
  - baseline `3y`: `0.10%`
  - combo `3y`: `25.06%`
- `AMZN`
  - baseline `3y`: `-10.44%`
  - entry filter `3y`: `7.50%`
  - combo `3y`: `5.33%`
- `GLD`
  - baseline `1y`: effectively flat
  - entry filter `1y`: `5.13%`
- `NVDA`
  - baseline `3y`: `121.17%`
  - entry filter `3y`: `128.77%`

### Interpretation

- The LLM branch is useful as a timing overlay, not as a standalone trading engine.
- It clearly helped several stock and gold cases.
- It did not help everything; `META` is the clearest negative case.
- The local model still has a positive-tone bias and does not produce many strongly negative days, so this is a valid first-pass result, not a final sentiment engine.

### Updated Recommendation

1. Keep rule-based momentum as the main production baseline.
2. Keep the LLM overlay as a research-only timing layer.
3. Carry forward `NVDA`, `MSFT`, `GLD`, `AAPL`, and `SLV` for deeper text-model comparison.
4. Next comparison should be `LLM overlay` vs `FinBERT / classic sentiment` on the same news bundles.

## DeepSeek Hourly LLM Overlay Follow-up

The DeepSeek daily stock-overlay branch has now been rerun on hourly bars for the same 16-stock universe.

Stored artifacts:

- [llm_news_sentiment_deepseek_stocks_hourly_run.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_stocks_hourly_run.json)
- [llm_news_deepseek_stocks_hourly_summary.md](C:\SVNProjects\Algoding\reports\research\llm_news_deepseek_stocks_hourly_summary.md)
- [deepseek_hourly_overlay_vs_baseline_dark.pdf](C:\SVNProjects\Algoding\output\pdf\deepseek_hourly_overlay_vs_baseline_dark.pdf)

Setup:

- timeframe: `hour`
- regular-hours only
- windows: `1y`, `2y`, `3y`
- execution-aware replay
- fixed stop loss `5%`
- trailing stop `1.5%`
- no same-day reentry: `false`

### Main Result

The strongest hourly rows overall remained baseline momentum rows:

- `AVGO / baseline`: `54.95%` average total return
- `NVDA / baseline`: `49.81%`

Across `48` symbol-window pairs:

- entry filter beat baseline in `26`
- exit filter beat baseline in `0`
- combo beat baseline in `28`

This is the key hourly change:

- `entry_filter` and `combo` remain useful
- `exit_filter` does not generalize to hourly bars in the current design

### Aggregate Comparison Across 16 Assets

Comparing the baseline to the best overlay per asset:

- mean average return:
  - baseline: `4.22%`
  - best overlay: `5.79%`
- return standard deviation:
  - baseline: `21.94%`
  - best overlay: `15.82%`
- mean average max drawdown:
  - baseline: `-16.68%`
  - best overlay: `-12.57%`
- drawdown standard deviation:
  - baseline: `6.52%`
  - best overlay: `5.94%`

The best overlay per asset beat the baseline on `9/16` assets and lagged on `7/16`.

### Best Hourly Overlay Improvements

- `GOOGL`: `+13.81 pts`
- `XOM`: `+10.62 pts`
- `AMZN`: `+10.14 pts`
- `AMD`: `+7.14 pts`
- `ORCL`: `+7.05 pts`
- `COST`: `+5.00 pts`

### Updated Recommendation

Carry forward these hourly overlay candidates:

- `GOOGL / combo`
- `XOM / entry_filter`
- `AMZN / combo`
- `AMD / entry_filter`
- `ORCL / entry_filter`
- `MSFT / combo`

Keep these names baseline-only for now:

- `AVGO`
- `NVDA`
- `NFLX`
- `TSLA`

And do not treat the hourly `exit_filter` as a default overlay until its logic is redesigned.

## DeepSeek FX Daily Overlay Follow-Up

Artifacts:

- [llm_news_sentiment_deepseek_fx_daily_run.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_fx_daily_run.json)
- [llm_news_deepseek_fx_daily_summary.md](C:\SVNProjects\Algoding\reports\research\llm_news_deepseek_fx_daily_summary.md)

Setup:

- model: `U:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B`
- symbols: `EURUSD`, `USDJPY`, `GBPUSD`, `AUDUSD`, `USDCAD`, `USDCHF`, `EURJPY`
- windows: `1y`, `2y`, `3y`
- timeframe: `day`
- execution-aware replay
- fixed stop loss `5%`
- trailing stop `1.5%`
- FX proxy-news mapping:
  - `EURUSD -> FXE, UUP`
  - `USDJPY -> FXY, UUP`
  - `GBPUSD -> FXB, UUP`
  - `AUDUSD -> FXA, UUP`
  - `USDCAD -> FXC, UUP`
  - `USDCHF -> FXF, UUP`
  - `EURJPY -> FXE, FXY`

### Main Result

Across `21` symbol-window pairs:

- `entry_filter` beat baseline in `18`
- `exit_filter` beat baseline in `3`
- `combo` beat baseline in `20`

Best overall FX overlay:

- `EURUSD / combo`
- average total return: `0.47%`
- average max drawdown: `-0.23%`
- profitable windows: `2/3`

Most improved pairs versus baseline:

- `AUDUSD`: `-7.18% -> -0.24%`
- `USDJPY`: `-6.34% -> -2.10%`
- `GBPUSD`: `-3.21% -> -1.17%`

### Interpretation

- DeepSeek improved timing on most FX pair-window tests, but the gains were mostly defensive.
- This branch reduced losses and drawdowns more than it created strong positive edge.
- `combo` and `entry_filter` remain the only credible FX overlay modes.
- `exit_filter` did not generalize.
- The only pair with positive average total return across `1y`, `2y`, and `3y` was `EURUSD`.

### Recommendation

- Keep `EURUSD / combo` as the primary FX LLM candidate.
- Keep `AUDUSD / combo`, `USDJPY / entry_filter`, and `GBPUSD / combo` as secondary research candidates.
- Do not promote FX to broker-paper from this branch yet.

## DeepSeek Daily Commodity Metals/Energy Overlay

Artifacts:

- [llm_news_sentiment_deepseek_commodities_daily_run.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_commodities_daily_run.json)
- [llm_news_deepseek_commodities_vs_stocks_summary.md](C:\SVNProjects\Algoding\reports\research\llm_news_deepseek_commodities_vs_stocks_summary.md)

Setup:

- symbols: `GLD`, `SLV`, `USO`, `UNG`
- windows: `1y`, `2y`, `3y`
- timeframe: `day`
- execution-aware replay
- fixed stop loss `5%`
- trailing stop `1.5%`
- model: `U:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B`

### Main Result

Across `12` symbol-window pairs:

- `entry_filter` beat baseline in `6`
- `exit_filter` beat baseline in `5`
- `combo` beat baseline in `5`

Best commodity overlays by symbol:

- `GLD / entry_filter`: `10.43%` average total return, `-4.53%` average max drawdown
- `SLV / exit_filter`: `5.35%`
- `UNG / combo`: `0.30%`
- `USO`: baseline remained best at `14.68%`

### Comparison with Large-Cap Stocks

Best-overlay-per-symbol statistics:

- stocks:
  - mean best average return: `20.76%`
  - mean overlay improvement over baseline: `4.32 pts`
  - mean best average max drawdown: `-9.64%`
- commodities:
  - mean best average return: `7.69%`
  - mean overlay improvement over baseline: `2.42 pts`
  - mean best average max drawdown: `-8.57%`

### Interpretation

- The DeepSeek overlay remains materially stronger on large-cap stocks than on commodity ETFs.
- `GLD` benefited clearly from the overlay.
- `USO` was already strong without the overlay; the overlay added no value there.
- `SLV` improved slightly but still carries deep drawdown.
- `UNG` is not a serious contender.

### Recommendation

- Keep `GLD / entry_filter` as the top commodity LLM overlay candidate.
- Keep `USO / baseline` as the stronger non-LLM energy benchmark.
- Keep `SLV / exit_filter` only as a secondary follow-up.
- Do not prioritize `UNG` for the next stage.

## DeepSeek Daily Non-U.S. Stock Overlay via Alpaca ADR/Global Listings

Artifacts:

- [llm_news_sentiment_deepseek_non_us_stocks_daily_run.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_non_us_stocks_daily_run.json)
- [llm_news_deepseek_non_us_stocks_summary.md](C:\SVNProjects\Algoding\reports\research\llm_news_deepseek_non_us_stocks_summary.md)

Scope note:

- This branch uses U.S.-listed ADRs/global stocks available through Alpaca.
- It is not direct non-U.S. exchange trading.

Basket:

- `ASML`, `TSM`, `NVO`, `SAP`, `BABA`, `PDD`, `HSBC`, `TM`, `RIO`, `SHEL`, `SONY`, `INFY`

Setup:

- windows: `1y`, `2y`, `3y`
- timeframe: `day`
- execution-aware replay
- fixed stop loss `5%`
- trailing stop `1.5%`
- model: `U:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B`

### Main Result

Across `36` symbol-window pairs:

- `entry_filter` beat baseline in `16`
- `exit_filter` beat baseline in `18`
- `combo` beat baseline in `17`

Best overall rows:

- `TSM / exit_filter`: `20.15%` average total return
- `BABA / exit_filter`: `12.52%`
- `ASML / baseline`: `9.04%`
- `SHEL / baseline`: `3.58%`

Largest symbol-level improvements:

- `PDD`: `-5.95% -> 2.14%`
- `TSM`: `13.35% -> 20.15%`
- `NVO`: `0.30% -> 3.72%`

### Comparison with U.S. Large-Cap Stock Branch

Best-overlay-per-symbol statistics:

- non-U.S. ADR/global stock basket:
  - mean best average return: `4.33%`
  - mean overlay improvement over baseline: `1.97 pts`
  - mean best average max drawdown: `-2.98%`
- U.S. large-cap stock basket:
  - mean best average return: `20.76%`
  - mean overlay improvement over baseline: `4.32 pts`
  - mean best average max drawdown: `-9.64%`

### Interpretation

- The DeepSeek overlay works on the ADR/global-stock basket, but the edge is materially smaller than on U.S. large caps.
- The branch is more defensive and lower-volatility.
- `exit_filter` is the strongest overlay mode here.
- `ASML`, `SHEL`, and `HSBC` were better left as baseline-only.

### Recommendation

- Keep `TSM / exit_filter`, `BABA / exit_filter`, `PDD / exit_filter`, and `NVO / exit_filter` as the strongest follow-ups.
- Keep `ASML`, `SHEL`, and `HSBC` as baseline-only benchmarks.
- Do not prioritize `TM`, `SONY`, `INFY`, or `RIO` for the next stage.

## DeepSeek Top 5 U.S. Stocks: Profit-Lock and Partial Take-Profit Sweep

Artifacts:

- [llm_deepseek_top5_us_profit_lock_partial_run.json](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_profit_lock_partial_run.json)
- [llm_deepseek_top5_us_profit_lock_partial_summary.md](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_profit_lock_partial_summary.md)

Top 5 daily DeepSeek stock contenders tested:

- `AVGO / entry_filter`
- `NVDA / exit_filter`
- `TSLA / exit_filter`
- `GOOGL / exit_filter`
- `XOM / exit_filter`

Rules tested:

- `current_best`
- `profit_lock_only`
- `profit_lock_partial_25_at_5`
- `profit_lock_partial_50_at_5`
- `profit_lock_partial_25_at_8`
- `profit_lock_partial_50_at_8`

Setup:

- windows: `1y`, `2y`, `3y`
- timeframe: `day`
- execution-aware replay
- fixed stop loss `5%`
- base trailing stop `1.5%`

### Main Result

Best aggregate variant across all `15` symbol-window slices:

- `profit_lock_partial_50_at_8`
  - average total return: `44.45%`
  - average max drawdown: `-10.81%`

Baseline aggregate:

- `current_best`
  - average total return: `44.36%`
  - average max drawdown: `-10.86%`

This means the aggregate edge is positive but small. The result is not a blanket replacement rule.

### Best Rule By Symbol

- `AVGO`: `profit_lock_only`
- `NVDA`: `profit_lock_partial_50_at_8`
- `TSLA`: `current_best`
- `GOOGL`: `profit_lock_partial_50_at_5`
- `XOM`: `current_best`

### Interpretation

- Profit-lock and partial exits help selected names.
- `GOOGL` showed the clearest improvement from partial take-profit.
- `NVDA` improved with a later partial take-profit at `+8%`.
- `AVGO` benefited from profit-lock tightening alone.
- `TSLA` and `XOM` should remain unchanged for now.

### Recommendation

- Keep `AVGO / entry_filter` but add `profit_lock_only`.
- Keep `NVDA / exit_filter` but add `profit_lock_partial_50_at_8`.
- Keep `GOOGL / exit_filter` but add `profit_lock_partial_50_at_5`.
- Leave `TSLA / exit_filter` and `XOM / exit_filter` unchanged.

## DeepSeek Top 5 U.S. Stocks: Rule Grid Sweep

Artifacts:

- [llm_deepseek_top5_us_rule_grid_run.json](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_rule_grid_run.json)
- [llm_deepseek_top5_us_rule_grid_summary.md](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_rule_grid_summary.md)

Top 5 daily DeepSeek stock contenders tested:

- `AVGO`
- `NVDA`
- `TSLA`
- `GOOGL`
- `XOM`

Rules tested:

- stop loss: `4%`, `5%`, `6%`
- trailing stop: `1%`, `1.5%`, `2%`
- break-even after `+3%`: `off`, `on`
- time stop: `10`, `20`, `30` bars
- reentry cooldown: `1`, `2`
- asymmetric entry threshold: `off`, `on`
- sentiment freshness: `1`, `2`, `3` trading days
- weak/neutral cluster block: `off`, `on`

Setup:

- windows: `1y`, `2y`, `3y`
- timeframe: `day`
- execution-aware replay
- total runs: `19,440`

### Main Result

Best-per-symbol aggregate versus prior DeepSeek daily leaders:

- prior mean average total return: `25.79%`
- new best-rule mean average total return: `26.50%`
- prior mean average max drawdown: `-12.54%`
- new best-rule mean average max drawdown: `-5.34%`

This was not a large return upgrade. The value of the sweep was much better drawdown control.

### Best Rule By Symbol

- `AVGO`: `sl_4_trail_1.5_be_off_time_10_cool_1_asym_off_weak_off`
- `NVDA`: `sl_4_trail_1_be_off_time_10_cool_2_asym_off_weak_off`
- `GOOGL`: `sl_4_trail_2_be_off_time_10_cool_1_asym_off_weak_off`
- `XOM`: `sl_4_trail_1.5_be_off_time_10_cool_1_asym_off_weak_on`
- `TSLA`: `sl_4_trail_1.5_be_off_time_10_cool_2_asym_off_weak_off`

### Parameter Scoring

- `trailing_stop_pct = 2%` was the best aggregate trailing-stop setting
- `reentry_cooldown = 1` slightly beat `2`
- `sentiment_freshness = 1` had the highest average return, while `3` was the more stable secondary choice
- `asymmetric stricter entry` hurt performance on average
- `weak_cluster_block` was only marginal overall, but helped `XOM`
- `break_even after +3%` had no measurable aggregate effect
- `stop_loss_pct` and `time_stop_bars` did not separate materially in this branch

### Recommendation

- Carry the optimized rule set forward for `AVGO`, `NVDA`, and `XOM`
- Keep `TSLA` and `GOOGL` under review because the optimized rules reduced drawdown but also reduced return
- Use this sweep as the starting point for the next hourly DeepSeek stock pass

## DeepSeek Top 5 U.S. Stocks: Best Combined Rule Set vs Previous Results

Artifact:

- [llm_deepseek_top5_us_best_combined_vs_previous.md](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_best_combined_vs_previous.md)

I also tested the best single shared rule set across all five names instead of using symbol-specific tuning.

Best shared rule set:

- `sl_4_trail_1.5_be_off_time_10_cool_2_asym_on_weak_off`
- freshness: `3` trading days

Result versus prior daily DeepSeek leaders:

- prior mean average total return: `25.79%`
- shared-rule mean average total return: `13.96%`
- prior mean average max drawdown: `-12.54%`
- shared-rule mean average max drawdown: `-4.62%`

Interpretation:

- the shared rule set is much more defensive
- but it gives up too much return
- only `NVDA` clearly improved under the shared policy

### Recommendation

- do **not** replace the symbol-specific best rules with one combined policy
- keep the combined policy only as a conservative fallback benchmark

## DeepSeek Top 5 U.S. Stocks: Hourly From Daily Rules vs Daily

Artifacts:

- [llm_deepseek_top5_us_hourly_from_daily_rules_run.json](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_hourly_from_daily_rules_run.json)
- [llm_deepseek_top5_us_hourly_vs_daily_summary.md](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_hourly_vs_daily_summary.md)

I applied the daily top-5 optimized rules to an hourly regular-hours DeepSeek stock run for:

- `AVGO`
- `NVDA`
- `TSLA`
- `GOOGL`
- `XOM`

To keep the rule meaning comparable, I mapped:

- `1` daily bar to `7` regular-session hourly bars
- daily time stop `10` bars to hourly time stop `70` bars
- daily cooldown `1/2` bars to hourly cooldown `7/14` bars

### Main Result

The daily-optimized rules did **not** transfer to hourly.

- mean daily optimized return: `26.50%`
- mean hourly return with transplanted daily rules: `-1.04%`
- mean daily optimized max drawdown: `-5.34%`
- mean hourly max drawdown with transplanted daily rules: `-10.87%`

### Interpretation

- the hourly branch needs its own native tuning
- reusing daily rules on hourly bars is materially worse than keeping daily and hourly branches separate
- the transplanted daily rules also failed to beat the earlier generic hourly branch on return

### Recommendation

- do **not** carry the daily optimized rules into the hourly branch
- keep hourly DeepSeek research as a separate optimization track

## Qwen3 vs DeepSeek on Top 5 U.S. Daily Stocks

Artifacts:

- [llm_news_sentiment_qwen3_top5_stocks_daily_run.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_qwen3_top5_stocks_daily_run.json)
- [llm_qwen3_top5_vs_deepseek_summary.md](C:\SVNProjects\Algoding\reports\research\llm_qwen3_top5_vs_deepseek_summary.md)

Open-source model work:

- downloaded `Qwen3-4B` to `U:\models\Qwen3-4B`
- downloaded `Qwen3-1.7B` to `U:\models\Qwen3-1.7B`

The `4B` model was too slow on the local `8GB` GPU because it spilled into CPU offload. The completed comparison used `Qwen3-1.7B`, which fit fully on GPU.

Setup:

- symbols: `AVGO`, `NVDA`, `TSLA`, `GOOGL`, `XOM`
- timeframe: `day`
- windows: `1y`, `2y`, `3y`
- execution-aware replay
- fixed stop loss: `5%`
- trailing stop: `1.5%`

### Main Result

Best-per-symbol aggregate comparison:

- Qwen3 mean average return: `34.35%`
- DeepSeek mean average return: `44.44%`
- Qwen3 mean average max drawdown: `-7.94%`
- DeepSeek mean average max drawdown: `-10.84%`

Interpretation:

- DeepSeek is still the stronger local LLM on return
- Qwen3 is more defensive and more bearish
- Qwen3 did not replicate DeepSeek’s strongest entry-filter wins, especially on `AVGO`

### Symbol highlights

- `NVDA`: Qwen3 stayed competitive with `exit_filter`
- `GOOGL`: Qwen3 remained close but still behind
- `XOM`: Qwen3 baseline slightly beat DeepSeek on return
- `AVGO`: DeepSeek remained clearly superior

### Recommendation

- keep DeepSeek as the primary local stock-news overlay model
- keep `Qwen3-1.7B` as a smaller full-GPU fallback baseline
- do not switch the stock branch away from DeepSeek

## DeepSeek Fine-Tuned News + Price Meta-Model

Artifacts:

- [deepseek_news_price_meta_plan.md](C:\SVNProjects\Algoding\reports\research\deepseek_news_price_meta_plan.md)
- [deepseek_news_price_meta_run.json](C:\SVNProjects\Algoding\reports\research\deepseek_news_price_meta_full\deepseek_news_price_meta_run.json)
- [deepseek_news_price_meta_summary.md](C:\SVNProjects\Algoding\reports\research\deepseek_news_price_meta_summary.md)

### What changed

This branch upgraded the research design from prompt-only sentiment gating to:

- future-return-derived labels
- `QLoRA` fine-tuning on `DeepSeek-R1-Distill-Qwen-1.5B`
- structured finance outputs
- fused `LLM + price` features
- strict chronological `train / validation / test`
- execution-aware replay on the later unseen test slice

### Setup

- train symbols: `AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, AVGO, AMD, NFLX, JPM, XOM, ORCL, CRM, WMT, COST`
- evaluation symbols: `AVGO, NVDA, TSLA, GOOGL, XOM`
- train split: `2023-01-25 -> 2024-12-19`
- validation split: `2024-12-20 -> 2025-08-13`
- test split: `2025-08-14 -> 2026-04-02`

Dataset sizes:

- train examples: `3537`
- validation examples: `2224`
- test examples: `1914`

Fine-tuning:

- capped train set used for QLoRA: `1200`
- validation set used for QLoRA: `300`
- train runtime: `752.148s`
- train loss: `0.144724`

### Strict test results

Aggregate comparison against the prompt-based DeepSeek overlay on the same test slice:

- fine-tuned news+price meta-model:
  - mean total return: `1.06%`
  - mean max drawdown: `-0.03%`
- prompt-based DeepSeek overlay:
  - mean total return: `3.43%`
  - mean max drawdown: `-2.98%`

Per symbol:

- `AVGO`
  - meta-model: `+0.87%`
  - prompt overlay: `+5.55%`
- `NVDA`
  - meta-model: `+4.41%`
  - prompt overlay: `+4.83%`
- `TSLA`
  - meta-model: `0.00%`
  - prompt overlay: `-1.00%`
- `GOOGL`
  - meta-model: `0.00%`
  - prompt overlay: `+4.48%`
- `XOM`
  - meta-model: `0.00%`
  - prompt overlay: `+3.27%`

### Interpretation

- the stronger research design is valid, but the first implementation is too selective
- it significantly reduced drawdown, but it also gave up too much return
- `NVDA` was the clearest useful case
- several symbols stayed flat in the test period, which indicates the current meta-thresholding is too conservative

### Comparison to earlier daily DeepSeek top-5 summary

The earlier looser daily summary across `1y/2y/3y` windows for the same top 5 had:

- mean average return: `44.60%`
- mean average max drawdown: `-10.91%`

That branch is not directly comparable because it is not the same strict later-only test design.

### Recommendation

- keep the fine-tuned branch as a valid research direction, but not as the new leader
- keep the current prompt-based DeepSeek overlay as the stronger stock-news branch for now
- next iteration should add:
  - raw prompt-based sentiment score as an additional fused feature
  - multi-horizon targets (`1d`, `3d`, `5d`)
  - lower meta-model entry thresholds
  - article density and sentiment-dispersion features
  - symbol-specific meta-models for `AVGO`, `NVDA`, and `GOOGL`

## DeepSeek Top-5 Strict Exact vs Meta Comparison

Artifacts:

- [deepseek_top5_strict_exact_vs_meta.json](C:\SVNProjects\Algoding\reports\research\deepseek_top5_strict_exact_vs_meta.json)
- [deepseek_top5_strict_exact_vs_meta_summary.md](C:\SVNProjects\Algoding\reports\research\deepseek_top5_strict_exact_vs_meta_summary.md)
- [deepseek_top5_strict_exact_vs_meta_dark.pdf](C:\SVNProjects\Algoding\output\pdf\deepseek_top5_strict_exact_vs_meta_dark.pdf)

Purpose:

- remove the earlier metric mixing
- rerun the exact earlier daily DeepSeek winners on the same strict later unseen test slice
- compare those exact prompt variants directly against the fine-tuned DeepSeek news+price meta-model

Exact earlier variants used:

- `AVGO / avgo_momentum_llm_entry_filter`, `entry_sentiment_min = 0.347`
- `NVDA / nvda_momentum_llm_exit_filter`, `exit_sentiment_max = 0.0`
- `TSLA / tsla_momentum_llm_exit_filter`, `exit_sentiment_max = 0.812801`
- `GOOGL / googl_momentum_llm_exit_filter`, `exit_sentiment_max = 0.0`
- `XOM / xom_momentum_llm_exit_filter`, `exit_sentiment_max = 0.0`

Aggregate:

- earlier multi-window average mean return: `44.60%`
- strict exact prompt mean return: `10.43%`
- strict meta-model mean return: `1.06%`

- earlier multi-window average mean max drawdown: `-10.91%`
- strict exact prompt mean max drawdown: `-3.11%`
- strict meta-model mean max drawdown: `-0.03%`

Result:

- the strict exact prompt variant beat the strict meta-model on return in `5/5` symbols
- the earlier DeepSeek stock winners remain valid as the headline results for the broad multi-window branch
- on the stricter unseen-test branch, those same prompt variants still outperform the current fine-tuned meta-model

## DeepSeek Directional Research v1

Artifacts:

- [deepseek_directional_run.json](C:\SVNProjects\Algoding\reports\research\deepseek_directional_v1\deepseek_directional_run.json)
- [deepseek_directional_summary.md](C:\SVNProjects\Algoding\reports\research\deepseek_directional_v1\deepseek_directional_summary.md)

Goal:

- train DeepSeek against future realized direction labels first
- score correctness first
- only then translate predictions into trades

Test-set directional metrics across `AVGO, NVDA, TSLA, GOOGL, XOM`:

- base directional DeepSeek:
  - mean direction accuracy: `33.72%`
  - mean direction macro-F1: `0.209953`
- fine-tuned directional DeepSeek:
  - mean direction accuracy: `37.56%`
  - mean direction macro-F1: `0.237399`

So the directional fine-tuning did improve correctness, but only modestly.

Trading replay:

- base directional branch:
  - mean total return: `10.68%`
  - mean max drawdown: `-1.51%`
- fine-tuned directional branch:
  - mean total return: `79.72%`
  - mean max drawdown: `-2.58%`

Comparison to earlier strict exact prompt branch:

- strict exact prompt mean return: `10.43%`
- strict exact prompt mean max drawdown: `-3.11%`

Interpretation:

- the new branch is closer to the intended research design
- however, the trading result is too strong relative to the still-weak prediction accuracy
- that means the current directional trade-mapping layer is too permissive and is likely benefitting from bullish skew and repeated risk-stop recycling
- keep this branch as the new correctness-first research foundation, but do not trust the v1 trading result as production-quality evidence

## DeepSeek Commodities V0 vs V2

Artifacts:

- [deepseek_directional_run.json](C:\SVNProjects\Algoding\reports\research\deepseek_directional_commodities_v2\deepseek_directional_run.json)
- [deepseek_commodities_v0_vs_v2.json](C:\SVNProjects\Algoding\reports\research\deepseek_commodities_v0_vs_v2.json)
- [deepseek_commodities_v0_vs_v2_summary.md](C:\SVNProjects\Algoding\reports\research\deepseek_commodities_v0_vs_v2_summary.md)

Scope:

- the current commodity DeepSeek overlay branch has `4` active contenders, not `5`: `GLD`, `SLV`, `USO`, `UNG`
- `V0` = strict exact prompt variants rerun on the later unseen test slice
- `V2` = fine-tuned directional branch rerun on the same `2025-08-14 -> 2026-04-02` strict test slice

Strict-test aggregates:

- `V0` mean total return: `1.90%`
- `V0` mean max drawdown: `-4.84%`
- `V2` mean direction accuracy: `42.35%`
- `V2` mean total return: `47.09%`
- `V2` mean max drawdown: `-1.07%`
- `V2` beat `V0` on return in `4/4` symbols

Per symbol:

- `GLD`
  - `V0`: `2.53%`, `-4.26%`
  - `V2`: `45.23%`, `-1.78%`, direction accuracy `52.38%`
- `SLV`
  - `V0`: `0.42%`, `-9.73%`
  - `V2`: `99.52%`, `-2.07%`, direction accuracy `47.92%`
- `USO`
  - `V0`: `5.27%`, `-3.69%`
  - `V2`: `40.72%`, `-0.25%`, direction accuracy `38.26%`
- `UNG`
  - `V0`: `-0.63%`, `-1.67%`
  - `V2`: `2.90%`, `-0.18%`, direction accuracy `30.85%`

Interpretation:

- the `V2` directional branch clearly dominates the commodity `V0` prompt branch on strict-test trading return and drawdown
- but the same caution from the stock directional branch still applies: the trading gains are disproportionately strong relative to the prediction accuracy
- `SLV` and `USO` are the clearest red flags here because the realized returns are far larger than the achieved classification quality would normally justify
- treat this as a strong research lead, not promotion-grade evidence

## Tightened Commodity V2

Artifacts:

- [deepseek_commodities_v0_vs_v2_tightened.json](C:\SVNProjects\Algoding\reports\research\deepseek_commodities_v0_vs_v2_tightened.json)
- [deepseek_commodities_v0_vs_v2_tightened_summary.md](C:\SVNProjects\Algoding\reports\research\deepseek_commodities_v0_vs_v2_tightened_summary.md)

Policy:

- only `medium/high` strength predictions were allowed
- relative confirmation was required
- momentum confirmation was required
- re-entry cooldowns of `3`, `5`, or `10` bars were applied
- the best constrained definition was selected on validation replay before testing

Strict-test aggregates:

- `V0` mean total return: `1.90%`
- original `V2` mean total return: `47.09%`
- tightened `V2` mean total return: `5.83%`
- `V0` mean max drawdown: `-4.84%`
- original `V2` mean max drawdown: `-1.07%`
- tightened `V2` mean max drawdown: `-2.82%`

Per symbol:

- `GLD`
  - chosen rule: `high + relative confirmation + momentum confirmation + cooldown 3`
  - tightened `V2`: `10.22%`, `-2.42%`
- `SLV`
  - chosen rule: `high + relative confirmation + momentum confirmation + cooldown 10`
  - tightened `V2`: `1.01%`, `-5.80%`
- `USO`
  - chosen rule: `high + relative confirmation + momentum confirmation + cooldown 3`
  - tightened `V2`: `11.24%`, `-2.78%`
- `UNG`
  - chosen rule: `high + relative confirmation + momentum confirmation + cooldown 3`
  - tightened `V2`: `0.86%`, `-0.27%`

Interpretation:

- the constrained mapper removes most of the suspiciously large original `V2` edge
- the tightened `V2` branch still beats `V0` on return in `4/4` symbols
- this tightened branch is materially more credible than the original loose `V2` commodity result
- `GLD` and `USO` remain the strongest commodity directional candidates after tightening

## Directional V3 Expansion

Artifacts:

- [deepseek_directional_run.json](C:\SVNProjects\Algoding\reports\research\deepseek_directional_v3\deepseek_directional_run.json)
- [v2_vs_v3_summary.md](C:\SVNProjects\Algoding\reports\research\deepseek_directional_v3\v2_vs_v3_summary.md)

Setup:

- expanded stock training universe from `16` to `34` names
- increased history from `800` bars to about `1400` available daily bars
- increased LoRA cap from `1200/300` train/eval to `3000/900`
- kept the same DeepSeek base checkpoint: `DeepSeek-R1-Distill-Qwen-1.5B`

Result:

- `V2` mean directional accuracy: `37.56%`
- `V3` mean directional accuracy: `37.02%`
- `V2` mean macro F1: `0.2374`
- `V3` mean macro F1: `0.2411`

Interpretation:

- simply expanding the universe, history, and train examples did not improve directional hit rate
- the directional branch still does not approach the `50%+` target band
- a larger checkpoint is not the first fix; labeling and trade mapping remain the main issues

## V3-3 and V3-4

Naming:

- `V3-3`: the loose `V3` stock branch using `medium/high` strength and only hard risk controls
- `V3-4`: `V3-3` plus `high` strength only and `momentum shift` entry instead of raw momentum thresholding

Artifacts:

- [v3_v23_like_top3_summary.md](C:\SVNProjects\Algoding\reports\research\deepseek_directional_v3\v3_v23_like_top3_summary.md)
- [v3_4_high_momentum_shift_top3.json](C:\SVNProjects\Algoding\reports\research\deepseek_directional_v3\v3_4_high_momentum_shift_top3.json)

Test set:

- `AVGO`, `NVDA`, `TSLA`
- strict test period: `2025-02-24 -> 2026-04-06`

Aggregates:

- `V3-3`
  - mean direction accuracy: `34.31%`
  - mean total return: `279.27%`
  - mean max drawdown: `-2.46%`
- `V3-4`
  - mean direction accuracy: `34.31%`
  - mean total return: `8.63%`
  - mean max drawdown: `-2.72%`

Per symbol `V3-4`:

- `AVGO`: `10.79%`, `-1.81%`
- `NVDA`: `-4.50%`, `-4.83%`
- `TSLA`: `19.59%`, `-1.51%`

Interpretation:

- `V3-3` remains too optimistic to trust
- `V3-4` removes most of the suspicious edge by forcing a fresh momentum-shift entry and `high` model confidence
- `TSLA` and `AVGO` remain viable follow-up names under the stricter mapper; `NVDA` weakens materially

## Directional V3-5

Artifacts:

- [v3_5_top5_trailing_1pct_no_same_day_reentry.json](C:\SVNProjects\Algoding\reports\research\deepseek_directional_v3\v3_5_top5_trailing_1pct_no_same_day_reentry.json)
- [v3_5_top5_trailing_1pct_no_same_day_reentry_summary.md](C:\SVNProjects\Algoding\reports\research\deepseek_directional_v3\v3_5_top5_trailing_1pct_no_same_day_reentry_summary.md)

Definition:

- `V3-5` = `V3-3` top-5 stock branch with:
  - `1.0%` trailing stop
  - `5%` stop loss
  - minimum strength `medium`
  - no relative confirmation
  - no momentum confirmation
  - `no same-day re-entry = true`

Test set:

- `AVGO`, `NVDA`, `TSLA`, `GOOGL`, `XOM`
- test period: `2025-02-25 -> 2026-04-07`

Aggregate:

- mean direction accuracy: `37.02%`
- mean macro F1: `0.2411`
- mean total return: `365.67%`
- mean max drawdown: `-1.28%`
- mean trades: `98.2`

Per symbol:

- `AVGO`: `645.22%`, `-1.04%`
- `NVDA`: `270.93%`, `-1.07%`
- `TSLA`: `735.40%`, `-1.08%`
- `GOOGL`: `90.12%`, `-1.26%`
- `XOM`: `86.70%`, `-1.94%`

Interpretation:

- adding `no same-day re-entry` did not fix the core credibility problem in the loose `V3-3` style mapper
- the branch remains dominated by permissive long exposure plus repeated stop-managed re-entry over time, not by a believable jump in directional skill
- `V3-5` should be treated as another sensitivity branch only, not a promotion benchmark

## V3-5 Paper Rollout

Artifacts:

- [portfolio config](C:\SVNProjects\Algoding\config\v35_paper_portfolio.json)
- [first weekly report](C:\SVNProjects\Algoding\reports\paper\v35\v35_weekly_report_20260407.md)
- [first weekly report JSON](C:\SVNProjects\Algoding\reports\paper\v35\v35_weekly_report_20260407.json)

Paper-trade setup:

- top symbols from the `V3-5` backtest branch: `TSLA`, `AVGO`, `NVDA`
- paper sleeve target capital: `$60,000`
- capital ratio frozen from live account equity at rollout: about `59.95%`
- equal sleeve weights when signals are active
- `5%` hard stop loss
- `1%` trailing stop
- `no same-day re-entry = true`
- duration target: `90` days

Initial live paper state:

- account was flat before rollout
- current `V3-5` signal wanted only `TSLA` long
- submitted `TSLA` buy notional: `$20,000`
- `AVGO` and `NVDA` remained flat because their current `V3-5` signal was `flat`

Initial broker result:

- `TSLA` quantity: about `58.2336`
- first post-entry snapshot equity: about `$100,083.62`
- first weekly report equity: about `$100,055.35`
- first unrealized P/L on `TSLA`: about `-$28.22`

Interpretation:

- the live paper rollout follows the current signal state rather than forcing all three names in at once
- unused sleeve capital remains in cash until `AVGO` or `NVDA` turn `long`
- this is a paper validation exercise only; it does not change the research view that `V3-5` is still a loose, low-credibility branch

## Directional V4

Artifacts:

- [deepseek_directional_v4_run.json](C:\SVNProjects\Algoding\reports\research\deepseek_directional_v4\deepseek_directional_v4_run.json)
- [deepseek_directional_v4_summary.md](C:\SVNProjects\Algoding\reports\research\deepseek_directional_v4\deepseek_directional_v4_summary.md)

Goal:

- target mean directional accuracy of `50%+`

Method changes:

- sector-specific models instead of one pooled stock model
- rebalanced train-set directional labels with reduced neutral skew
- multi-horizon labels at `1d`, `3d`, `5d`
- fused raw DeepSeek prompt sentiment scores with price features
- direct comparison against a price-only directional baseline

Aggregate result:

- fused `V4` mean direction accuracy: `33.31%`
- price-only baseline mean direction accuracy: `34.69%`
- fused `V4` mean macro F1: `0.2797`
- price-only baseline mean macro F1: `0.3085`
- fused `V4` beat price-only accuracy on `2/5` symbols
- symbols reaching `50%+` accuracy: `0/5`

Per symbol:

- `AVGO`
  - fused: `38.91%`
  - price-only: `37.09%`
- `NVDA`
  - fused: `26.55%`
  - price-only: `31.27%`
- `TSLA`
  - fused: `39.27%`
  - price-only: `39.27%`
- `GOOGL`
  - fused: `28.36%`
  - price-only: `32.73%`
- `XOM`
  - fused: `33.45%`
  - price-only: `33.09%`

Interpretation:

- `V4` is methodologically cleaner than the earlier loose directional branches
- but it did not achieve the target directional accuracy
- on this design, raw DeepSeek prompt sentiment did not improve mean direction accuracy over a price-only sector model
- `AVGO` and `XOM` are the only modest positive fusion cases

## Directional V4-3

Artifacts:

- [deepseek_directional_v43_run.json](C:\SVNProjects\Algoding\reports\research\deepseek_directional_v4\deepseek_directional_v43_run.json)
- [deepseek_directional_v43_summary.md](C:\SVNProjects\Algoding\reports\research\deepseek_directional_v4\deepseek_directional_v43_summary.md)

Definition:

- `V4-3` = `V4` fused directional predictions with the old `V3-3`-style loose mapper
- minimum strength `medium`
- no relative confirmation
- no momentum confirmation
- model-driven entry/exit
- stop loss `5%`
- trailing stop `1.5%`

Aggregate:

- mean direction accuracy: `33.31%`
- mean macro F1: `0.2797`
- mean total return: `3.31%`
- mean max drawdown: `-0.18%`
- mean trades: `1.8`

Interpretation:

- `V4-3` is far more conservative than the earlier loose branches because the `V4` fused classifier itself is less trade-happy
- only `AVGO` and `NVDA` produced non-zero returns under this mapper
- this is a more believable directional branch than `V2-3` / `V3-3`, but it still does not approach the directional accuracy target

## Directional V3-5 Hourly

Artifacts:

- [deepseek_directional_hourly_run.json](C:\SVNProjects\Algoding\reports\research\deepseek_directional_v3_hourly\deepseek_directional_hourly_run.json)
- [deepseek_directional_hourly_summary.md](C:\SVNProjects\Algoding\reports\research\deepseek_directional_v3_hourly\deepseek_directional_hourly_summary.md)

Definition:

- `V3-5-Hourly` = full hourly retrain and rerun of the `V3-5` branch
- timeframe `hour`
- entry threshold `medium/high`
- no relative confirmation
- no momentum confirmation
- stop loss `5%`
- trailing stop `1.0%`
- `no same-day re-entry = true`

Universe:

- training symbols: `AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, AVGO, AMD, NFLX, JPM, XOM, ORCL, CRM, WMT, COST`
- evaluation symbols: `AVGO, NVDA, TSLA, GOOGL, XOM`

Hourly split:

- train: `2024-07-30T14:00:00+00:00 -> 2025-08-14T20:00:00+00:00`
- validation: `2025-08-15T12:00:00+00:00 -> 2025-12-09T21:00:00+00:00`
- test: `2025-12-10T13:00:00+00:00 -> 2026-04-07T18:00:00+00:00`

Aggregate:

- mean direction accuracy: `30.77%`
- mean macro F1: `0.1757`
- mean total return: `9.00%`
- mean max drawdown: `-3.21%`
- mean trades: `54.8`

Per symbol:

- `AVGO`
  - direction accuracy: `28.57%`
  - total return: `23.36%`
  - max drawdown: `-3.77%`
- `NVDA`
  - direction accuracy: `29.64%`
  - total return: `7.07%`
  - max drawdown: `-3.42%`
- `TSLA`
  - direction accuracy: `29.43%`
  - total return: `3.13%`
  - max drawdown: `-5.85%`
- `GOOGL`
  - direction accuracy: `27.27%`
  - total return: `-0.01%`
  - max drawdown: `-1.95%`
- `XOM`
  - direction accuracy: `38.93%`
  - total return: `11.45%`
  - max drawdown: `-1.07%`

Interpretation:

- the hourly rerun is materially less explosive than the daily `V3-5` branch
- the result is more believable than the earlier loose daily mapper, but predictive quality is still weak
- `AVGO` and `XOM` are the strongest hourly cases
- `GOOGL` is effectively flat under the hourly branch

## Crypto Directional C_V-0

Artifacts:

- [deepseek_directional_crypto_run.json](C:\SVNProjects\Algoding\reports\research\deepseek_directional_crypto_cv0\deepseek_directional_crypto_run.json)
- [deepseek_directional_crypto_summary.md](C:\SVNProjects\Algoding\reports\research\deepseek_directional_crypto_cv0\deepseek_directional_crypto_summary.md)

Definition:

- `C_V-0` = crypto directional DeepSeek branch similar to `V3-5`
- broad crypto basket for training
- evaluation on `BTC/USD`, `ETH/USD`, `SOL/USD`
- minimum strength `medium`
- no relative confirmation
- no momentum confirmation
- stop loss `5%`
- trailing stop `1.0%`
- `no same-day re-entry = true`

Universe and split:

- training symbols: `BTC/USD, ETH/USD, SOL/USD, DOGE/USD, LTC/USD, BCH/USD, AVAX/USD, LINK/USD, UNI/USD, AAVE/USD, XRP/USD`
- evaluation symbols: `BTC/USD, ETH/USD, SOL/USD`
- train: `2024-01-30 -> 2025-05-23`
- validation: `2025-05-24 -> 2025-10-30`
- test: `2025-10-31 -> 2026-04-08`

Aggregate:

- base mean direction accuracy: `41.94%`
- fine-tuned mean direction accuracy: `27.50%`
- base mean return: `0.01%`
- fine-tuned mean return: `0.17%`
- base mean max drawdown: `-0.00%`
- fine-tuned mean max drawdown: `-0.02%`
- fine-tuned beats base on return: `3/3`

Per symbol:

- `BTC/USD`
  - base accuracy: `41.38%`
  - fine-tuned accuracy: `29.66%`
  - base return: `0.01%`
  - fine-tuned return: `0.33%`
- `ETH/USD`
  - base accuracy: `40.00%`
  - fine-tuned accuracy: `29.60%`
  - base return: `-0.00%`
  - fine-tuned return: `0.11%`
- `SOL/USD`
  - base accuracy: `44.44%`
  - fine-tuned accuracy: `23.23%`
  - base return: `0.01%`
  - fine-tuned return: `0.08%`

Interpretation:

- the fine-tuned crypto branch improved trading return slightly over the base model on all three evaluated names
- but the absolute edge is tiny after realistic crypto costs
- directional accuracy deteriorated materially after fine-tuning
- this is not a promotion-ready branch; it remains a research-only crypto result
