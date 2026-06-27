# V3-Q3E Qwen3-Embedding Directional Run

- model: `Qwen/Qwen3-Embedding-4B`
- role: `frozen Qwen3 embedding encoder + downstream logistic directional classifiers`
- CUDA used: `True`
- encoder: `{"model_name": "Qwen/Qwen3-Embedding-4B", "max_length": 768, "batch_size": 4, "quantization": "4bit"}`
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
| `q3e_text_only` | `32.39%` | `31.68%` | `0.2887` | `6.19%` | `-13.91%` |
| `q3e_fused` | `33.13%` | `32.56%` | `0.2932` | `-0.90%` | `-15.83%` |
| `q3e_sector` | `32.54%` | `33.03%` | `0.2984` | `18.25%` | `-15.45%` |

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
| `AVGO` | `30.80%` | `28.93%` | `0.2891` | `16.06%` | `-24.76%` | `51` |
| `NVDA` | `32.42%` | `31.30%` | `0.3128` | `-2.62%` | `-17.28%` | `47` |
| `TSLA` | `32.81%` | `31.24%` | `0.2999` | `-2.37%` | `-18.91%` | `39` |
| `GOOGL` | `32.88%` | `31.56%` | `0.3062` | `14.95%` | `-7.56%` | `36` |
| `XOM` | `33.02%` | `35.39%` | `0.2354` | `4.94%` | `-1.06%` | `8` |

## q3e_fused

| Symbol | Dir. acc | Balanced acc | Macro F1 | Net return (med) | Net max DD (med) | Trades |
|---|---:|---:|---:|---:|---:|---:|
| `AVGO` | `33.60%` | `30.71%` | `0.3046` | `-14.19%` | `-34.59%` | `46` |
| `NVDA` | `34.25%` | `33.86%` | `0.3338` | `12.00%` | `-16.36%` | `56` |
| `TSLA` | `33.98%` | `31.99%` | `0.2980` | `-18.76%` | `-19.59%` | `33` |
| `GOOGL` | `30.82%` | `30.83%` | `0.2940` | `11.49%` | `-7.56%` | `32` |
| `XOM` | `33.02%` | `35.39%` | `0.2354` | `4.94%` | `-1.06%` | `8` |

## q3e_sector

| Symbol | Dir. acc | Balanced acc | Macro F1 | Net return (med) | Net max DD (med) | Trades |
|---|---:|---:|---:|---:|---:|---:|
| `AVGO` | `38.00%` | `37.98%` | `0.3753` | `57.15%` | `-14.44%` | `46` |
| `NVDA` | `31.51%` | `31.08%` | `0.3040` | `2.02%` | `-27.39%` | `55` |
| `TSLA` | `31.25%` | `31.78%` | `0.3092` | `7.16%` | `-22.24%` | `50` |
| `GOOGL` | `30.82%` | `31.40%` | `0.2900` | `22.59%` | `-11.26%` | `35` |
| `XOM` | `31.13%` | `32.91%` | `0.2135` | `2.31%` | `-1.91%` | `11` |

## Charts

- `reports\research\qwen_embedding_v3_q3e\charts\returns_by_variant.png`
- `reports\research\qwen_embedding_v3_q3e\charts\drawdown_by_variant.png`

