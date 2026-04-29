# DeepSeek Top 5 U.S. Stocks: Profit-Lock and Partial Take-Profit Sweep

Artifacts:

- [run JSON](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_profit_lock_partial_run.json)

Universe:

- `AVGO / entry_filter`
- `NVDA / exit_filter`
- `TSLA / exit_filter`
- `GOOGL / exit_filter`
- `XOM / exit_filter`

Setup:

- model: `U:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B`
- windows: `1y`, `2y`, `3y`
- timeframe: `day`
- execution-aware replay
- fixed stop loss: `5%`
- base trailing stop: `1.5%`

Rules tested:

- `current_best`
- `profit_lock_only`
  - once trade is up `+5%`, trailing stop tightens from `1.5%` to `1.0%`
- `profit_lock_partial_25_at_5`
- `profit_lock_partial_50_at_5`
- `profit_lock_partial_25_at_8`
- `profit_lock_partial_50_at_8`

Total runs:

- `5` symbols
- `6` rule variants
- `3` windows
- `90` total runs

## Aggregate result

Best aggregate variant across all `15` symbol-window slices:

- `profit_lock_partial_50_at_8`
  - average total return: `44.45%`
  - average max drawdown: `-10.81%`

Baseline comparison:

- `current_best`
  - average total return: `44.36%`
  - average max drawdown: `-10.86%`

Interpretation:

- The aggregate improvement is real, but small.
- The new rules do not dominate uniformly.
- These exit rules are symbol-specific rather than universal.

## Best rule by symbol

- `AVGO`
  - best: `profit_lock_only`
  - average return: `61.61%`
  - delta vs current best: `+0.82 pts`

- `NVDA`
  - best: `profit_lock_partial_50_at_8`
  - average return: `62.77%`
  - delta vs current best: `+1.46 pts`

- `TSLA`
  - best: `current_best`
  - no tested profit-lock or partial rule improved it

- `GOOGL`
  - best: `profit_lock_partial_50_at_5`
  - average return: `36.66%`
  - delta vs current best: `+2.97 pts`

- `XOM`
  - best: `current_best`
  - all tested variants were effectively unchanged

## Practical conclusion

Keep:

- `AVGO / entry_filter` with `profit_lock_only`
- `NVDA / exit_filter` with `profit_lock_partial_50_at_8`
- `GOOGL / exit_filter` with `profit_lock_partial_50_at_5`

Do not change yet:

- `TSLA / exit_filter`
- `XOM / exit_filter`

The main takeaway is that profit-lock and partial exits can help, but only on selected names. They should not be applied uniformly across the whole DeepSeek stock basket.
