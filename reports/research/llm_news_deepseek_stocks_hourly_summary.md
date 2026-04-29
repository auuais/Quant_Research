# DeepSeek LLM Stocks Overlay Research - Hourly

Last updated: 2026-04-02

## Scope

This run extends the DeepSeek daily stock-overlay branch to hourly bars.

Artifacts:

- `reports/research/llm_news_sentiment_deepseek_stocks_hourly_run.json`
- `output/pdf/deepseek_hourly_overlay_vs_baseline_dark.pdf`
- `output/pdf/deepseek_hourly_overlay_vs_baseline_dark.json`

Model:

- `U:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B`

Universe:

- `AAPL`
- `MSFT`
- `NVDA`
- `AMZN`
- `META`
- `GOOGL`
- `TSLA`
- `AVGO`
- `AMD`
- `NFLX`
- `JPM`
- `XOM`
- `ORCL`
- `CRM`
- `WMT`
- `COST`

Setup:

- windows: `1y`, `2y`, `3y`
- timeframe: `hour`
- regular-hours only
- execution-aware replay
- fixed stop loss: `5%`
- trailing stop: `1.5%`
- no same-day reentry: `false`

## Headline Findings

- The best hourly rows overall were still baseline momentum rows:
  - `AVGO / baseline`: `54.95%` average total return
  - `NVDA / baseline`: `49.81%`
- The strongest overlay rows were:
  - `AVGO / llm_exit_filter`: `40.22%`
  - `AVGO / llm_combo`: `31.25%`
  - `ORCL / llm_entry_filter`: `25.07%`
  - `NVDA / llm_exit_filter`: `37.53%`

## Overlay Vs Baseline

Across `48` symbol-window pairs:

- `entry_filter` beat baseline in `26`
- `exit_filter` beat baseline in `0`
- `combo` beat baseline in `28`

This is the main hourly conclusion:

- `entry_filter` and `combo` are the useful hourly overlay modes
- `exit_filter` does not generalize to hourly bars under the current setup

## Aggregate Stats Across 16 Assets

Using the best overlay per asset:

### Average total return

- baseline mean: `4.22%`
- best-overlay mean: `5.79%`
- baseline variance: `0.048129`
- best-overlay variance: `0.025023`
- baseline std dev: `21.94%`
- best-overlay std dev: `15.82%`
- baseline min: `-24.70%`
- best-overlay min: `-14.55%`
- baseline max: `54.95%`
- best-overlay max: `40.22%`

### Average max drawdown

- baseline mean: `-16.68%`
- best-overlay mean: `-12.57%`
- baseline variance: `0.004255`
- best-overlay variance: `0.003523`
- baseline std dev: `6.52%`
- best-overlay std dev: `5.94%`
- baseline min: `-28.42%`
- best-overlay min: `-23.38%`
- baseline max: `-7.01%`
- best-overlay max: `-3.65%`

Interpretation:

- the best overlay per asset improved the average return
- it also reduced the average drawdown and dispersion
- but it did not beat the strongest baseline leaders such as `AVGO` and `NVDA`

## Asset-Level Result

Best overlay beat baseline on `9/16` assets and lagged on `7/16`.

Largest positive return deltas:

- `GOOGL`: `+13.81 pts`
- `XOM`: `+10.62 pts`
- `AMZN`: `+10.14 pts`
- `AMD`: `+7.14 pts`
- `ORCL`: `+7.05 pts`
- `COST`: `+5.00 pts`

Largest negative return deltas:

- `AVGO`: `-14.73 pts`
- `NVDA`: `-12.28 pts`
- `NFLX`: `-3.24 pts`
- `META`: `-2.62 pts`
- `AAPL`: `-2.09 pts`
- `TSLA`: `-1.94 pts`

## Recommendation

For the hourly branch, carry forward:

- `GOOGL / combo`
- `XOM / entry_filter`
- `AMZN / combo`
- `AMD / entry_filter`
- `ORCL / entry_filter`
- `MSFT / combo`
- `COST / combo`

Keep baseline-only for now on the strongest hourly leaders:

- `AVGO`
- `NVDA`
- `NFLX`
- `TSLA`

And avoid the hourly `exit_filter` as a default overlay until the design changes.
