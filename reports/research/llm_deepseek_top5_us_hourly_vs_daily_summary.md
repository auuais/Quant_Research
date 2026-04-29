# DeepSeek Top 5 U.S. Stocks: Hourly From Daily Rules vs Daily

Artifacts:

- [llm_deepseek_top5_us_hourly_from_daily_rules_run.json](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_hourly_from_daily_rules_run.json)
- [llm_deepseek_top5_us_rule_grid_run.json](C:\SVNProjects\Algoding\reports\research\llm_deepseek_top5_us_rule_grid_run.json)
- [llm_news_sentiment_deepseek_stocks_hourly_run.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_stocks_hourly_run.json)

## Setup

Applied the **daily top-5 optimized rule sets** to an **hourly regular-hours-only** DeepSeek stock run for:

- `AVGO`
- `NVDA`
- `TSLA`
- `GOOGL`
- `XOM`

Comparison basis:

- daily optimized rule results from the top-5 rule grid
- earlier generic hourly DeepSeek branch for the same symbols

Assumption used for portability:

- `1` daily bar = `7` regular-session hourly bars
- so:
  - daily time stop `10` bars -> hourly time stop `70` bars
  - daily cooldown `1`/`2` bars -> hourly cooldown `7`/`14` hourly bars

## Aggregate comparison: hourly vs daily optimized

- mean daily optimized return: `26.50%`
- mean hourly return with daily rules transplanted: `-1.04%`
- delta: `-27.54 pts`

- mean daily optimized max drawdown: `-5.34%`
- mean hourly max drawdown with daily rules transplanted: `-10.87%`
- delta: `-5.54 pts`

Conclusion:

- the daily-optimized rule sets **do not transfer** to hourly execution
- the hourly branch should remain independently tuned

## Per-symbol hourly vs daily optimized

| Symbol | Daily optimized return | Hourly from daily rules | Delta return | Daily max DD | Hourly max DD | Delta DD |
|---|---:|---:|---:|---:|---:|---:|
| `AVGO` | `56.47%` | `14.60%` | `-41.87 pts` | `-7.24%` | `-10.50%` | `-3.26 pts` |
| `NVDA` | `36.01%` | `-9.53%` | `-45.55 pts` | `-6.08%` | `-13.34%` | `-7.26 pts` |
| `TSLA` | `11.28%` | `-1.89%` | `-13.17 pts` | `-7.45%` | `-15.37%` | `-7.92 pts` |
| `GOOGL` | `19.27%` | `-1.57%` | `-20.83 pts` | `-4.12%` | `-7.65%` | `-3.53 pts` |
| `XOM` | `9.47%` | `-6.81%` | `-16.29 pts` | `-1.79%` | `-7.51%` | `-5.72 pts` |

## Comparison to earlier generic hourly branch

The transplanted daily rules also failed to beat the earlier hourly branch on return:

| Symbol | Prior hourly best | Prior hourly return | Daily-rule hourly return | Delta return |
|---|---|---:|---:|---:|
| `AVGO` | `avgo_momentum_baseline` | `54.95%` | `14.60%` | `-40.35 pts` |
| `NVDA` | `nvda_momentum_baseline` | `49.81%` | `-9.53%` | `-59.34 pts` |
| `TSLA` | `tsla_momentum_baseline` | `13.00%` | `-1.89%` | `-14.89 pts` |
| `GOOGL` | `googl_momentum_llm_combo` | `1.07%` | `-1.57%` | `-2.64 pts` |
| `XOM` | `xom_momentum_llm_entry_filter` | `-5.41%` | `-6.81%` | `-1.41 pts` |

The transplanted daily rules were only better on drawdown for some names, but not enough to justify replacing the hourly branch.

## Recommendation

- do **not** reuse the daily top-5 rules in hourly trading research
- keep the hourly DeepSeek branch separate
- next hourly work should start from the earlier hourly winners and tune **hourly-native** exits/risk gates instead
