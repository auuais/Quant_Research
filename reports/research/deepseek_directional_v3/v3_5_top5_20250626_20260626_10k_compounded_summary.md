# V3-5 Last 1 Year With $10,000 Compounded Accounting

Source artifact: `reports\research\deepseek_directional_v3\v3_5_top5_20250626_20260626.json`

Window requested: `2025-06-26 -> 2026-06-26`

Policy:

- minimum strength: `medium`
- relative confirmation: `false`
- momentum confirmation: `false`
- stop loss: `5%`
- trailing stop: `1.0%`
- no same-day re-entry: `true`

Note: the replay engine already compounds by reinvesting the full normalized equity after each trade. This report converts those compounded returns to dollar terms.

## $10,000 Per Symbol

| Symbol | Start | Final equity | Profit | Return | Max DD | Trades | Latest signal |
|---|---:|---:|---:|---:|---:|---:|---|
| `TSLA` | `$10,000.00` | `$57,091.30` | `$47,091.30` | `470.91%` | `-1.08%` | `100` | `long` |
| `AVGO` | `$10,000.00` | `$49,621.66` | `$39,621.66` | `396.22%` | `-1.04%` | `111` | `flat` |
| `NVDA` | `$10,000.00` | `$27,718.56` | `$17,718.56` | `177.19%` | `-2.17%` | `82` | `long` |
| `XOM` | `$10,000.00` | `$18,013.91` | `$8,013.91` | `80.14%` | `-1.94%` | `107` | `flat` |
| `GOOGL` | `$10,000.00` | `$15,780.80` | `$5,780.80` | `57.81%` | `-1.26%` | `55` | `flat` |

Per-symbol case aggregate:

- start capital: `$50,000.00`
- final equity: `$168,226.23`
- profit: `$118,226.23`
- return: `236.45%`

## $10,000 Total Equal-Weight Basket

| Symbol | Start allocation | Final equity | Profit | Return |
|---|---:|---:|---:|---:|
| `TSLA` | `$2,000.00` | `$11,418.26` | `$9,418.26` | `470.91%` |
| `AVGO` | `$2,000.00` | `$9,924.33` | `$7,924.33` | `396.22%` |
| `NVDA` | `$2,000.00` | `$5,543.71` | `$3,543.71` | `177.19%` |
| `XOM` | `$2,000.00` | `$3,602.78` | `$1,602.78` | `80.14%` |
| `GOOGL` | `$2,000.00` | `$3,156.16` | `$1,156.16` | `57.81%` |

Equal-weight basket aggregate:

- start capital: `$10,000.00`
- final equity: `$33,645.24`
- profit: `$23,645.24`
- return: `236.45%`
