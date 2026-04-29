# DeepSeek Directional Research Plan

## Objective

Build a separate LLM research branch where the primary target is `future direction correctness`, not immediate trading return.

## Rules of the branch

1. Label the future first.
   - use later realized `5d` direction with a volatility-scaled neutral band
   - also label relative performance vs `SPY`

2. Train on chronological data only.
   - no random split
   - train / validation / test by date

3. Score correctness first.
   - direction accuracy
   - balanced accuracy
   - macro-F1
   - relative-to-market accuracy
   - joint accuracy

4. Translate to trades second.
   - convert predictions into long/flat actions
   - replay with the same execution-aware assumptions as the rest of the repo

5. Compare with the current benchmark.
   - strict exact DeepSeek prompt winners on the same unseen test slice

## v1 implementation

- training symbols: `AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, AVGO, AMD, NFLX, JPM, XOM, ORCL, CRM, WMT, COST`
- evaluation symbols: `AVGO, NVDA, TSLA, GOOGL, XOM`
- model: `U:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B`
- training: `QLoRA`
- split:
  - train `60%`
  - validation `20%`
  - test `20%`

## Success criteria

This branch is considered a genuine improvement only if:

1. prediction correctness improves materially over the unfine-tuned base model
2. trading results remain strong after stricter entry gating
3. returns are consistent with prediction quality rather than obviously inflated by overexposure
