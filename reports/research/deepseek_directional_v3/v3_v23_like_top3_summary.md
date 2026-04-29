# V3 with V2-3-like Policy on Top 3 Stocks

Policy:

- minimum strength: `medium`
- relative confirmation: `false`
- momentum confirmation: `false`
- stop loss: `5%`
- trailing stop: `1.5%`

Test period:

- `2025-02-24 -> 2026-04-06 (280 days)`

## Per Symbol

| Symbol | Direction accuracy | Macro F1 | Total return | Max drawdown | Trades |
|---|---:|---:|---:|---:|---:|
| `AVGO` | `30.56%` | `0.1743` | `287.00%` | `-1.86%` | `146` |
| `NVDA` | `35.68%` | `0.2702` | `204.74%` | `-3.79%` | `115` |
| `TSLA` | `36.69%` | `0.2842` | `346.08%` | `-1.73%` | `102` |

## Aggregate

- mean direction accuracy: `34.31%`
- mean macro F1: `0.2429`
- mean total return: `279.27%`
- mean max drawdown: `-2.46%`

## Interpretation

- This behaves like the earlier loose directional variants: trading returns are far too high relative to the still-weak directional accuracy.
- So this is useful as a sensitivity check, but not a credible production result.
