# DeepSeek Directional V4-3

Definition:

- `V4-3` = `V4` fused directional predictions with a `V3-3`-style mapper
- entry/hold policy:
  - minimum strength `medium`
  - no relative confirmation
  - no momentum confirmation
  - model-driven entry/exit
  - hard stop loss `5%`
  - trailing stop `1.5%`

Test period:

- `2025-02-24 -> 2026-04-06 (280 days)`

## Aggregate

- mean direction accuracy: `33.31%`
- mean macro F1: `0.2797`
- mean total return: `3.31%`
- mean max drawdown: `-0.18%`
- mean trades: `1.8`

## Per Symbol

| Symbol | Direction accuracy | Macro F1 | Total return | Max drawdown | Trades |
|---|---:|---:|---:|---:|---:|
| `AVGO` | `38.91%` | `0.3747` | `11.70%` | `-0.90%` | `6` |
| `NVDA` | `26.55%` | `0.2455` | `4.84%` | `-0.02%` | `3` |
| `TSLA` | `39.27%` | `0.2687` | `0.00%` | `0.00%` | `0` |
| `GOOGL` | `28.36%` | `0.2516` | `0.00%` | `0.00%` | `0` |
| `XOM` | `33.45%` | `0.2582` | `0.00%` | `0.00%` | `0` |

## Interpretation

- `V4-3` is materially more conservative than the earlier loose directional branches
- the fused `V4` model only produced meaningful tradable behavior in `AVGO` and `NVDA` under this mapper
- the branch remains well below the directional accuracy goal of `50%+`
