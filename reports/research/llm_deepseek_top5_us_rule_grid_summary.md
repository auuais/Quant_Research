# DeepSeek Top 5 U.S. Stocks Rule Grid

Artifacts:

- [llm_deepseek_top5_us_rule_grid_run.json](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_rule_grid_run.json)
- [llm_news_sentiment_deepseek_stocks_daily_run_full.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_stocks_daily_run_full.json)

Universe:

- `AVGO`
- `NVDA`
- `TSLA`
- `GOOGL`
- `XOM`

Base DeepSeek overlays:

- `AVGO / entry_filter`
- `NVDA / exit_filter`
- `TSLA / exit_filter`
- `GOOGL / exit_filter`
- `XOM / exit_filter`

Setup:

- timeframe: `day`
- windows: `1y`, `2y`, `3y`
- execution-aware replay
- model: `U:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B`
- total runs: `19,440`

Grid tested:

- stop loss: `4%`, `5%`, `6%`
- trailing stop: `1%`, `1.5%`, `2%`
- break-even after `+3%`: `off`, `on`
- time stop: `10`, `20`, `30` bars
- reentry cooldown: `1`, `2` bars
- asymmetric threshold: `off`, `on`
- sentiment freshness: `1`, `2`, `3` trading days
- weak/neutral cluster block: `off`, `on`

## Aggregate result

Best-per-symbol aggregate versus prior DeepSeek daily leaders:

- prior mean average total return: `25.79%`
- new best-rule mean average total return: `26.50%`
- delta: `+0.71 pts`
- prior mean average max drawdown: `-12.54%`
- new best-rule mean average max drawdown: `-5.34%`
- drawdown improvement: `+7.20 pts`

Interpretation:

- the rule sweep improved the average return only slightly
- the main gain was materially lower drawdown
- rule optimization helped `AVGO`, `NVDA`, and `XOM`
- it hurt `TSLA` and `GOOGL` on return, even while reducing drawdown

## Best rule by symbol

| Symbol | Best rule | Freshness | Avg return | Avg max drawdown |
|---|---|---:|---:|---:|
| `AVGO` | `sl_4_trail_1.5_be_off_time_10_cool_1_asym_off_weak_off` | `1` | `56.47%` | `-7.24%` |
| `NVDA` | `sl_4_trail_1_be_off_time_10_cool_2_asym_off_weak_off` | `3` | `36.01%` | `-6.08%` |
| `GOOGL` | `sl_4_trail_2_be_off_time_10_cool_1_asym_off_weak_off` | `1` | `19.27%` | `-4.12%` |
| `XOM` | `sl_4_trail_1.5_be_off_time_10_cool_1_asym_off_weak_on` | `2` | `9.47%` | `-1.79%` |
| `TSLA` | `sl_4_trail_1.5_be_off_time_10_cool_2_asym_off_weak_off` | `3` | `11.28%` | `-7.45%` |

## Parameter scores

Scored by average replay score across all runs.

| Parameter | Best value | Avg return | Avg max drawdown | Avg score |
|---|---|---:|---:|---:|
| `stop_loss_pct` | no separation | `12.28%` | `-5.82%` | `0.1834` |
| `trailing_stop_pct` | `2%` | `12.83%` | `-6.22%` | `0.1866` |
| `time_stop_bars` | no separation | `12.28%` | `-5.82%` | `0.1834` |
| `reentry_cooldown_bars` | `1` | `12.64%` | `-5.93%` | `0.1848` |
| `sentiment_freshness_days` | `1` by return, `3` by stability | `14.54%` / `12.23%` | `-5.33%` / `-5.68%` | `0.1994` / `0.1888` |
| `break_even_enabled` | no effect | `12.28%` | `-5.82%` | `0.1834` |
| `asymmetric_threshold_enabled` | `False` | `12.64%` | `-5.66%` | `0.1868` |
| `weak_cluster_block` | `True` by score | `12.25%` | `-5.79%` | `0.1839` |

## Interpretation

- `4%` stop loss dominated the symbol-level winners, but aggregate stop-loss scoring was flat because stops rarely determined the final path under this branch.
- `2%` trailing stop won on aggregate score, but symbol-level winners still split between `1%`, `1.5%`, and `2%`.
- `10`-bar time stop did not separate from `20` or `30` bars in the aggregate results.
- `1`-bar cooldown slightly outperformed `2`.
- `1`-day sentiment freshness gave the highest average return.
- `3`-day freshness was more stable and helped `NVDA` and `TSLA`.
- `break-even after +3%` added no measurable edge in this sweep.
- `asymmetric stricter entry` hurt this top-5 basket on average.
- `weak/neutral cluster blocking` was only marginally helpful overall, but it did help `XOM`.

## Recommendation

- Keep `AVGO` with the optimized rule set.
- Keep `NVDA` with the optimized rule set.
- Keep `XOM` with weak-cluster blocking enabled.
- Do not force the optimized grid onto `TSLA` or `GOOGL` without a separate objective that explicitly prioritizes lower drawdown over return.
- For the next hourly pass, start from:
  - stop loss `4%`
  - trailing stop `1.5%` and `2%`
  - cooldown `1`
  - sentiment freshness `1` and `3`
  - asymmetric threshold `off`
  - weak-cluster block `off` by default, `on` for `XOM`
