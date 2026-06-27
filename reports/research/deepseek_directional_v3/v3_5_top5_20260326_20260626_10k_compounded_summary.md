# V3-5 Last 3 Months With $10,000 Compounded Accounting

Source artifact: `reports\research\deepseek_directional_v3\v3_5_top5_20260326_20260626.json`

Window requested: `2026-03-26 -> 2026-06-26`

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
| `TSLA` | `$10,000.00` | `$18,442.83` | `$8,442.83` | `84.43%` | `-0.53%` | `29` | `long` |
| `AVGO` | `$10,000.00` | `$16,347.94` | `$6,347.94` | `63.48%` | `-0.60%` | `29` | `flat` |
| `NVDA` | `$10,000.00` | `$14,232.85` | `$4,232.85` | `42.33%` | `-2.17%` | `28` | `long` |
| `GOOGL` | `$10,000.00` | `$14,051.77` | `$4,051.77` | `40.52%` | `-1.02%` | `30` | `flat` |
| `XOM` | `$10,000.00` | `$12,903.32` | `$2,903.32` | `29.03%` | `-0.73%` | `30` | `flat` |

Per-symbol case aggregate:

- start capital: `$50,000.00`
- final equity: `$75,978.71`
- profit: `$25,978.71`
- return: `51.96%`

## $10,000 Total Equal-Weight Basket

| Symbol | Start allocation | Final equity | Profit | Return |
|---|---:|---:|---:|---:|
| `TSLA` | `$2,000.00` | `$3,688.57` | `$1,688.57` | `84.43%` |
| `AVGO` | `$2,000.00` | `$3,269.59` | `$1,269.59` | `63.48%` |
| `NVDA` | `$2,000.00` | `$2,846.57` | `$846.57` | `42.33%` |
| `GOOGL` | `$2,000.00` | `$2,810.35` | `$810.35` | `40.52%` |
| `XOM` | `$2,000.00` | `$2,580.66` | `$580.66` | `29.03%` |

Equal-weight basket aggregate:

- start capital: `$10,000.00`
- final equity: `$15,195.74`
- profit: `$5,195.74`
- return: `51.96%`
