# V3-4 Trailing Stop Sweep on Top 3 Stocks

Policy:

- model: `V3` finetuned directional branch
- symbols: `AVGO`, `NVDA`, `TSLA`
- test period: `2025-02-24 -> 2026-04-06`
- entry: `high` confidence + fresh `momentum shift`
- exit: `model-predicted exit` plus hard `stop loss` / `trailing stop`
- fixed stop loss: `5%`

## Best trailing-stop settings by composite score

| Rank | Trailing stop | Mean return | Mean max drawdown | Mean trades | Composite score |
|---|---:|---:|---:|---:|---:|
| 1 | `1.0%` | `12.72%` | `-1.84%` | `8.33` | `0.244680` |
| 2 | `1.5%` | `8.63%` | `-2.72%` | `8.33` | `0.196334` |
| 3 | `2.0%` | `4.64%` | `-3.74%` | `8.33` | `0.135715` |
| 4 | `2.5%` | `2.62%` | `-4.96%` | `8.33` | `0.103376` |
| 5 | `3.0%` | `0.54%` | `-6.51%` | `8.00` | `0.049399` |

## Interpretation

- tighter trailing stops worked better in this branch
- `1.0%` is the best tested stop level on the aggregate score
- performance deteriorated steadily as the trailing stop was loosened
- direction accuracy did not change across the sweep; the difference came from exit/risk handling, not better prediction
- `NVDA` remained the weak leg across the stop sweep, while `AVGO` and `TSLA` held up better
