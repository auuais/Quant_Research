# DeepSeek News + Price Meta-Model Summary

Artifact:

- [deepseek_news_price_meta_run.json](C:\SVNProjects\Algoding\reports\research\deepseek_news_price_meta_full\deepseek_news_price_meta_run.json)

Plan reference:

- [deepseek_news_price_meta_plan.md](C:\SVNProjects\Algoding\reports\research\deepseek_news_price_meta_plan.md)

## Setup

- Base model: `U:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B`
- Fine-tuning: `QLoRA`, `4-bit`
- Train symbols: `AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, AVGO, AMD, NFLX, JPM, XOM, ORCL, CRM, WMT, COST`
- Evaluation symbols: `AVGO, NVDA, TSLA, GOOGL, XOM`
- Daily bars
- Cost-aware replay
- Risk:
  - stop loss `5%`
  - trailing stop `1.5%`

Date split:

- train: `2023-01-25 -> 2024-12-19`
- validation: `2024-12-20 -> 2025-08-13`
- test: `2025-08-14 -> 2026-04-02`

Dataset sizes:

- train examples: `3537`
- validation examples: `2224`
- test examples: `1914`

Training:

- fine-tune train examples used: `1200`
- eval examples used: `300`
- train runtime: `752.148s`
- train loss: `0.144724`

## Strict test comparison

Aggregate:

- DeepSeek news+price meta-model:
  - mean total return: `1.06%`
  - mean max drawdown: `-0.03%`
- Existing prompt-based DeepSeek overlay:
  - mean total return: `3.43%`
  - mean max drawdown: `-2.98%`
- Meta-model beat the prompt overlay on return in `1/5` symbols

Per symbol:

- `AVGO`
  - meta-model: `+0.87%`, drawdown `-0.02%`
  - prompt overlay: `+5.55%`, drawdown `-3.38%`
- `NVDA`
  - meta-model: `+4.41%`, drawdown `-0.11%`
  - prompt overlay: `+4.83%`, drawdown `-3.92%`
- `TSLA`
  - meta-model: `0.00%`, drawdown `0.00%`
  - prompt overlay: `-1.00%`, drawdown `-5.66%`
- `GOOGL`
  - meta-model: `0.00%`, drawdown `0.00%`
  - prompt overlay: `+4.48%`, drawdown `-0.02%`
- `XOM`
  - meta-model: `0.00%`, drawdown `0.00%`
  - prompt overlay: `+3.27%`, drawdown `-1.93%`

## Comparison to earlier daily DeepSeek top-5 summary

Earlier top-5 daily DeepSeek overlay summary from the broader, looser window averaging branch:

- mean average return: `44.60%`
- mean average max drawdown: `-10.91%`

That is **not** apples-to-apples with the strict split above. The earlier summary averaged `1y/2y/3y` runs, while the new branch evaluates only the later unseen test period.

## Conclusion

The improved research design is stronger, but the first implementation did **not** beat the existing prompt-based DeepSeek overlay on return.

What it did achieve:

- materially lower drawdown
- much more conservative behavior
- a clean end-to-end path for:
  - return-derived labeling
  - QLoRA fine-tuning
  - fused LLM + price features
  - strict date-based evaluation

Main weakness of this first pass:

- the meta-model is too selective and often stays flat
- several symbols ended with `0` trades in the test period

Best next improvements:

1. include the raw prompt-based DeepSeek sentiment score as an additional feature instead of only structured outputs
2. train the meta-model on more than one target horizon, e.g. `1d`, `3d`, `5d`
3. lower the meta entry threshold search range and allow more flexible exits
4. add article density, sentiment dispersion, and recency-decay features
5. run a second pass with symbol-specific meta-models for `AVGO`, `NVDA`, and `GOOGL`
