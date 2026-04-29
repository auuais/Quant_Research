# DeepSeek Directional V2 vs V3

## Setup

- `V2`
  - training symbols: `16`
  - train split: `2023-01-25 -> 2024-12-19 (480 days)`
  - validation split: `2024-12-20 -> 2025-08-13 (160 days)`
  - test split: `2025-08-14 -> 2026-04-02 (160 days)`
  - available examples: train `3537`, validation `2224`, test `1914`
  - LoRA cap used: train `1200`, eval `300`

- `V3`
  - training symbols: `34`
  - train split: `2020-09-08 -> 2024-01-09 (840 days)`
  - validation split: `2024-01-10 -> 2025-02-21 (280 days)`
  - test split: `2025-02-24 -> 2026-04-06 (280 days)`
  - available examples: train `11179`, validation `6171`, test `6877`
  - LoRA cap used: train `3000`, eval `900`

## Aggregates

| Metric | V2 | V3 |
|---|---:|---:|
| Mean directional accuracy | `37.56%` | `37.02%` |
| Mean directional macro F1 | `0.2374` | `0.2411` |
| Mean trade return | `79.72%` | `187.03%` |
| Mean max drawdown | `-2.58%` | `-2.52%` |

## Interpretation

- `V3` did **not** improve mean directional accuracy.
- `V3` did improve macro F1 slightly, which suggests a marginally better class balance, but not a better headline hit rate.
- The much higher trade return in `V3` is not trustworthy by itself because the directional accuracy stayed weak; it is more likely a trade-mapper/exposure artifact than a real predictive jump.
- Conclusion: simply expanding the universe, history, and LoRA sample cap is **not enough** to reach the target directional accuracy band of `50%+`.
