# DeepSeek Directional V4

Goal:

- target mean directional accuracy: `50%+`

Design changes vs prior directional branches:

- rebalanced directional labels using sector-specific train-set quantile bands
- reduced neutral skew with a `30%` neutral band
- added multi-horizon labels: `1d`, `3d`, `5d`
- fused raw DeepSeek prompt sentiment with price features
- trained sector-specific models instead of one pooled stock model
- compared fused model against a price-only directional baseline

Test symbols:

- `AVGO`, `NVDA`, `TSLA`, `GOOGL`, `XOM`

Sector universes:

- `semis`: `AVGO`, `NVDA`, `AMD`, `QCOM`, `INTC`, `AMAT`, `LRCX`, `KLAC`
- `growth`: `GOOGL`, `TSLA`, `META`, `AMZN`, `NFLX`, `AAPL`, `MSFT`, `CRM`, `ORCL`, `ADBE`
- `energy`: `XOM`, `CVX`, `COP`, `SLB`, `EOG`, `MPC`

Splits:

- train: `2020-09-08 -> 2024-01-09 (840 days)`
- validation: `2024-01-10 -> 2025-02-21 (280 days)`
- test: `2025-02-24 -> 2026-04-06 (280 days)`

## Aggregate

| Metric | Fused V4 | Price-only baseline |
|---|---:|---:|
| Mean direction accuracy | `33.31%` | `34.69%` |
| Mean macro F1 | `0.2797` | `0.3085` |
| Mean trade return | `0.15%` | `0.08%` |

Headline result:

- `V4` did **not** reach the `50%` directional accuracy goal
- fused DeepSeek prompt features beat price-only accuracy on `2/5` symbols

## Per Symbol

| Symbol | Fused acc | Price-only acc | Fused macro F1 | Price-only macro F1 |
|---|---:|---:|---:|---:|
| `AVGO` | `38.91%` | `37.09%` | `0.3747` | `0.3707` |
| `NVDA` | `26.55%` | `31.27%` | `0.2455` | `0.2984` |
| `TSLA` | `39.27%` | `39.27%` | `0.2687` | `0.3220` |
| `GOOGL` | `28.36%` | `32.73%` | `0.2516` | `0.3136` |
| `XOM` | `33.45%` | `33.09%` | `0.2582` | `0.2376` |

## Interpretation

- the sector-specific split and multi-horizon design are cleaner than earlier directional branches
- but the fused DeepSeek prompt signal did not improve mean directional accuracy over price-only
- `AVGO` and `XOM` are the only clean positive cases for the fused model
- `NVDA` and `GOOGL` clearly favored the price-only baseline
- the branch is more credible than the earlier loose mappers because the trading layer is conservative and near-flat

## Conclusion

- `V4` is a valid research improvement in methodology
- `V4` is **not** a predictive breakthrough
- the next step should focus on better labels and market-context features, not just more data or a larger checkpoint
