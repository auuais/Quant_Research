# DeepSeek News + Price Research Plan

## Why this plan

The stronger pattern in public research is not "LLM alone decides trades." It is:

1. Use the language model to interpret news or produce finance-aware text features.
2. Combine those outputs with market state and price features.
3. Evaluate with strict chronological splits and realistic trading costs.

Primary references that motivate this design:

- EMNLP Industry 2024, *Fine-Tuning Large Language Models for Stock Return Prediction Using Newsflow*:
  [https://aclanthology.org/2024.emnlp-industry.77/](https://aclanthology.org/2024.emnlp-industry.77/)
- QLoRA, *Efficient Finetuning of Quantized LLMs*:
  [https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
- FinBERT, domain-adapted financial sentiment:
  [https://arxiv.org/abs/2006.08097](https://arxiv.org/abs/2006.08097)
- Deep momentum / optimize toward trading outcomes:
  [https://arxiv.org/abs/1904.04912](https://arxiv.org/abs/1904.04912)

## Improved concrete plan

### 1. News labeler

Build daily news bundles aligned to trading days and label them from future excess returns instead of subjective sentiment.

Output fields:

- `symbol`
- `trading_day`
- `session_label`
- `event_type`
- `forward_return_5d`
- `forward_excess_return_5d`
- `label` in `{bullish, bearish, neutral}`
- `strength` in `{low, medium, high}`

Labeling rule:

- use next `5` trading day excess return versus `SPY`
- use a dynamic neutral band based on realized volatility plus a minimum tradability threshold

### 2. DeepSeek fine-tune dataset

Create instruction-style prompt/completion pairs with structured JSON outputs.

Prompt includes:

- symbol context
- trading day
- pre/post/intraday session context
- article count
- bundled headlines

Completion includes:

- `label`
- `strength`
- `event_type`
- `session`
- `horizon`

### 3. QLoRA fine-tune

Fine-tune `DeepSeek-R1-Distill-Qwen-1.5B` with:

- `4-bit` quantization
- LoRA adapters on attention and MLP projections
- chronological train / validation split only

### 4. News + price feature builder

Score bundles with the fine-tuned model, then combine those structured outputs with price features:

- `1/3/5/10/20` day returns
- distance to `SMA10` and `SMA20`
- `20` day realized volatility
- volume z-score
- structured LLM outputs:
  - signed label
  - strength
  - event type
  - session type

### 5. Overlay meta-model

Train a separate numeric model on the fused features.

Current implementation:

- `MLPClassifier`

Target:

- whether next `5` trading day excess return exceeds the tradability threshold

Trade logic:

- momentum gate from the existing baseline
- entry only when meta probability exceeds validation-selected threshold
- exit on momentum failure, probability deterioration, or risk stops

### 6. Strict evaluation

Use date-based splits across all symbols:

- train
- validation
- test

Then evaluate only on later unseen dates with:

- same execution-aware slippage model already used in the repo
- same stop loss and trailing stop settings
- direct comparison against the existing prompt-based DeepSeek overlay

## Success criteria

The branch is considered an improvement only if it beats the prompt-based DeepSeek overlay on the strict test slice in at least one of these ways:

- higher mean return with similar drawdown
- similar return with materially lower drawdown
- better symbol-level consistency across the top evaluation names
