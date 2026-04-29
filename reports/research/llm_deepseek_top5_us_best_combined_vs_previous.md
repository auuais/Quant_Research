# DeepSeek Top 5 U.S. Stocks: Best Combined Rule Set vs Previous Results

Artifacts:

- [llm_deepseek_top5_us_rule_grid_run.json](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_rule_grid_run.json)
- [llm_news_sentiment_deepseek_stocks_daily_run_full.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_stocks_daily_run_full.json)
- [llm_deepseek_top5_us_rule_grid_summary.md](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_rule_grid_summary.md)

## Global best combined rule set

Best single rule set across all five names from the rule-grid leaderboard:

- variant: `sl_4_trail_1.5_be_off_time_10_cool_2_asym_on_weak_off`
- sentiment freshness: `3` trading days
- stop loss: `4%`
- trailing stop: `1.5%`
- break-even after `+3%`: `off`
- time stop: `10` bars
- reentry cooldown: `2`
- asymmetric stricter entry threshold: `on`
- weak/neutral cluster block: `off`

This was the highest-scoring *single shared policy* across the top-5 basket.

## Aggregate comparison

Compared with the prior best daily DeepSeek result already in use for each symbol:

- prior mean average total return: `25.79%`
- global combined-rule mean average total return: `13.96%`
- delta: `-11.82 pts`

- prior mean average max drawdown: `-12.54%`
- global combined-rule mean average max drawdown: `-4.62%`
- drawdown improvement: `+7.91 pts`

## Per-symbol comparison

| Symbol | Prior best | Prior return | Combined return | Delta return | Prior max DD | Combined max DD | Delta DD |
|---|---|---:|---:|---:|---:|---:|---:|
| `AVGO` | `avgo_momentum_baseline` | `38.94%` | `16.69%` | `-22.25 pts` | `-11.57%` | `-6.63%` | `+4.94 pts` |
| `NVDA` | `nvda_momentum_llm_combo` | `15.98%` | `27.54%` | `+11.56 pts` | `-16.81%` | `-4.17%` | `+12.63 pts` |
| `TSLA` | `tsla_momentum_baseline` | `39.71%` | `11.28%` | `-28.43 pts` | `-21.73%` | `-7.45%` | `+14.28 pts` |
| `GOOGL` | `googl_momentum_baseline` | `27.96%` | `12.03%` | `-15.93 pts` | `-9.60%` | `-2.95%` | `+6.65 pts` |
| `XOM` | `xom_momentum_llm_combo` | `6.34%` | `2.27%` | `-4.07 pts` | `-2.97%` | `-1.92%` | `+1.06 pts` |

## Interpretation

- A single shared policy is **not** the best way to carry these names forward.
- The combined rule set made the basket much more defensive, but it gave up too much return on `AVGO`, `TSLA`, and `GOOGL`.
- `NVDA` is the only clear winner under the combined policy.
- This means the earlier conclusion still holds: the rule improvements are mostly **symbol-specific**, not universal.

## Practical decision

Use the **per-symbol optimized rules** from the rule-grid sweep, not the single global combined rule set.

Recommended next step:

- carry the per-symbol optimized rules into the hourly DeepSeek stock pass
- keep the combined policy only as a conservative fallback benchmark
