# V3-Q3E Qwen3-Embedding Directional Run

- model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- role: `frozen Qwen3 embedding encoder + downstream logistic directional classifiers`
- CUDA used: `True`
- encoder: `{"model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "max_length": 512, "batch_size": 1, "quantization": "4bit"}`
- train rows used: `10000` (class balance `{'neutral': 3721, 'bullish': 3333, 'bearish': 2946}`)
- sentiment features used: `False` (coverage `{'train': 0.114, 'validation': 0.154, 'test': 0.1572}`)

## Date splits (with purge/embargo)

- train: `2020-11-27 -> 2024-03-25 (835 days)`
- embargo: `2024-03-26 -> 2025-05-14 (10 days)`
- validation: `2024-04-03 -> 2025-05-07 (275 days)`
- test: `2025-05-15 -> 2026-06-26 (280 days)`

## Comparison vs V3-5 (medium threshold, net of costs; loose 5% exit, market-on-close)

| Branch | Mean dir. acc | Balanced acc | Macro F1 | Mean net return | Mean net max DD |
|---|---:|---:|---:|---:|---:|
| `V3-5` | `37.92%` | `33.11%` | `0.2444` | `23.22%` | `-20.45%` |
| `price_only_baseline` | `34.69%` | `35.96%` | `0.3205` | `0.14%` | `-0.42%` |
| `q3e_text_only` | `34.76%` | `34.24%` | `0.3335` | `12.25%` | `-11.05%` |
| `q3e_fused` | `34.16%` | `33.55%` | `0.3260` | `10.70%` | `-12.32%` |
| `q3e_sector` | `34.57%` | `32.46%` | `0.3126` | `6.57%` | `-14.65%` |

Best variant: `price_only_baseline` — max mean balanced accuracy, then mean net return at the primary (medium) threshold (beats V3-5 balanced accuracy: `True`).

## price_only_baseline

| Symbol | Dir. acc | Balanced acc | Macro F1 | Net return (med) | Net max DD (med) | Trades |
|---|---:|---:|---:|---:|---:|---:|
| `AVGO` | `39.20%` | `37.80%` | `0.3647` | `0.00%` | `0.00%` | `0` |
| `NVDA` | `34.25%` | `38.14%` | `0.3155` | `0.00%` | `0.00%` | `0` |
| `TSLA` | `34.77%` | `34.58%` | `0.3000` | `0.00%` | `0.00%` | `0` |
| `GOOGL` | `30.82%` | `37.40%` | `0.3121` | `0.22%` | `-0.00%` | `1` |
| `XOM` | `34.43%` | `31.87%` | `0.3102` | `0.49%` | `-2.12%` | `12` |

## q3e_text_only

| Symbol | Dir. acc | Balanced acc | Macro F1 | Net return (med) | Net max DD (med) | Trades |
|---|---:|---:|---:|---:|---:|---:|
| `AVGO` | `33.20%` | `32.58%` | `0.3205` | `0.05%` | `-12.60%` | `52` |
| `NVDA` | `38.36%` | `37.87%` | `0.3773` | `35.93%` | `-8.68%` | `43` |
| `TSLA` | `33.20%` | `32.83%` | `0.3290` | `31.76%` | `-9.39%` | `51` |
| `GOOGL` | `38.36%` | `36.55%` | `0.3560` | `7.12%` | `-10.95%` | `41` |
| `XOM` | `30.66%` | `31.38%` | `0.2848` | `-13.62%` | `-13.62%` | `34` |

## q3e_fused

| Symbol | Dir. acc | Balanced acc | Macro F1 | Net return (med) | Net max DD (med) | Trades |
|---|---:|---:|---:|---:|---:|---:|
| `AVGO` | `33.60%` | `32.01%` | `0.3168` | `0.76%` | `-18.34%` | `44` |
| `NVDA` | `38.81%` | `38.25%` | `0.3811` | `34.03%` | `-8.68%` | `43` |
| `TSLA` | `33.59%` | `33.23%` | `0.3322` | `23.02%` | `-12.41%` | `49` |
| `GOOGL` | `36.99%` | `35.68%` | `0.3463` | `5.02%` | `-11.37%` | `37` |
| `XOM` | `27.83%` | `28.59%` | `0.2535` | `-9.33%` | `-10.82%` | `30` |

## q3e_sector

| Symbol | Dir. acc | Balanced acc | Macro F1 | Net return (med) | Net max DD (med) | Trades |
|---|---:|---:|---:|---:|---:|---:|
| `AVGO` | `32.80%` | `29.67%` | `0.2934` | `1.16%` | `-17.61%` | `62` |
| `NVDA` | `32.88%` | `32.33%` | `0.3204` | `23.06%` | `-9.60%` | `52` |
| `TSLA` | `34.38%` | `33.64%` | `0.3335` | `-12.47%` | `-25.14%` | `48` |
| `GOOGL` | `38.36%` | `31.19%` | `0.3215` | `20.59%` | `-12.29%` | `33` |
| `XOM` | `34.43%` | `35.46%` | `0.2944` | `0.52%` | `-8.59%` | `29` |

## Charts

- `reports\research\qwen_embedding_v3_7b\charts\returns_by_variant.png`
- `reports\research\qwen_embedding_v3_7b\charts\drawdown_by_variant.png`

