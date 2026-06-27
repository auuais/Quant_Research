# V3-Q3E roadmap results: V3-5 vs 4B vs 7B (Items 1–3)

All rows are **net of costs**, medium signal threshold, identical window **2025-05-15 → 2026-06-26 (280 days)**,
identical policy: **loose 5% fixed exit, no trailing, market-on-close execution, conviction sizing,
band labels, no calibration**. Only the encoder differs between the 4B and 7B blocks.

## Net return / risk

| Branch | Encoder | Balanced acc | Macro F1 | **Net return** | Net max DD | Trades |
|---|---|---:|---:|---:|---:|---:|
| **Buy & hold** | — | — | — | **+49.3%** | (passive) | 1 |
| **V3-5** (DeepSeek directional) | — | 33.1% | 0.244 | **+23.2%** | −20.5% | 43.8 |
| q3e_text_only | Qwen3-Embedding-4B | 31.7% | 0.289 | +6.2% | −13.9% | 36.2 |
| q3e_fused | Qwen3-Embedding-4B | 32.6% | 0.293 | −0.9% | −15.8% | 35.0 |
| q3e_sector | Qwen3-Embedding-4B | 33.0% | 0.298 | +18.3% | −15.5% | 39.4 |
| q3e_text_only | DeepSeek-R1-Distill-Qwen-7B | 34.2% | **0.334** | **+12.3%** | −11.1% | — |
| q3e_fused | DeepSeek-R1-Distill-Qwen-7B | 33.6% | 0.326 | +10.7% | −12.3% | — |
| q3e_sector | DeepSeek-R1-Distill-Qwen-7B | 32.5% | 0.313 | +6.6% | −14.7% | — |

## Read

- **Nothing beats buy-and-hold (+49.3%).** Every active strategy — V3-5 and all embedding variants —
  underperforms simply holding these five trending mega-caps, net of costs. The active strategies' only
  edge is lower drawdown (−11 to −16% via conviction sizing vs the full B&H ride).
- **The 7B encoder modestly improves the pooled text variants over the 4B**: macro F1 0.33 vs 0.29,
  net return +12.3% vs +6.2% (text_only) and +10.7% vs −0.9% (fused), with lower drawdown. So the
  larger reasoning-distilled encoder produces better text representations here — even as a frozen
  causal-LM hidden state. (q3e_sector is the exception: the smaller per-sector models are noisier and
  the 4B happened to land a more long-biased fit, +18% vs +7%.)
- **V3-5 (the DeepSeek directional model) still leads net return (+23.2%)** among active strategies, and
  the embeddings have not beaten it. Balanced accuracy across the board sits at ~33% (≈ random for 3
  classes), so none of these classifiers has a strong directional edge; positive returns are dominated
  by long exposure to uptrending names.

## Item verdicts

- **Item 1 (drop 1% trailing → loose 5% + MOC):** done; the new default is net-positive and Alpaca-executable.
- **Item 2:** conviction sizing = keep (cut net DD ~5pts). Probability calibration and tight triple-barrier
  labels = reverted to opt-in — both pushed the weak classifiers into degenerate modes (all-neutral /
  all-bullish). Sentiment fused feature still off (needs a full 34-symbol DeepSeek cache regen; only the
  5 eval symbols were extended to 2026-06-26).
- **Item 3 (V3-7B):** ran on the same harness; the 7B helps the text variants but doesn't reach V3-5 or B&H.
