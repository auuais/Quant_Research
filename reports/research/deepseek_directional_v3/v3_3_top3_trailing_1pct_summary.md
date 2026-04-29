# V3-3 Top 3 Stocks with 1.0% Trailing Stop

Policy:

- minimum strength: `medium`
- relative confirmation: `false`
- momentum confirmation: `false`
- stop loss: `5%`
- trailing stop: `1.0%`

Test period:

- `2025-02-24 -> 2026-04-06 (280 days)`

## Per Symbol

| Symbol | Direction accuracy | Macro F1 | Total return | Max drawdown | Trades |
|---|---:|---:|---:|---:|---:|
| `AVGO` | `30.56%` | `0.1743` | `365.64%` | `-1.43%` | `147` |
| `NVDA` | `35.68%` | `0.2702` | `351.27%` | `-1.53%` | `102` |
| `TSLA` | `36.69%` | `0.2842` | `448.56%` | `-1.49%` | `102` |

## Aggregate

- mean direction accuracy: `34.31%`
- mean macro F1: `0.2429`
- mean total return: `388.49%`
- mean max drawdown: `-1.48%`

## Comparison to V3-3 at 1.5%

- `1.5%` trailing stop mean return: `279.27%`
- `1.0%` trailing stop mean return: `388.49%`
- `1.5%` trailing stop mean max drawdown: `-2.46%`
- `1.0%` trailing stop mean max drawdown: `-1.48%`

Interpretation:

- the looser `V3-3` branch becomes even more optimistic with a tighter trailing stop
- this strengthens the earlier conclusion that the branch is dominated by permissive entry plus repeated stop-managed re-entry, not strong directional forecasting skill
