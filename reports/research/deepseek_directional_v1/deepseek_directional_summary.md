# DeepSeek Directional Research v1

Artifact:

- [deepseek_directional_run.json](C:\SVNProjects\Algoding\reports\research\deepseek_directional_v1\deepseek_directional_run.json)

Purpose:

- move the LLM branch to a true `predict direction first, trade second` workflow
- train DeepSeek against future realized direction labels
- score correctness with classification metrics before looking at trading returns
- compare that branch against the earlier strict DeepSeek prompt winners

## Design

- train symbols: `AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA, AVGO, AMD, NFLX, JPM, XOM, ORCL, CRM, WMT, COST`
- evaluation symbols: `AVGO, NVDA, TSLA, GOOGL, XOM`
- daily bars
- strict chronological split:
  - train: `2023-01-25 -> 2024-12-19`
  - validation: `2024-12-20 -> 2025-08-13`
  - test: `2025-08-14 -> 2026-04-02`
- label:
  - `direction` in `{bullish, bearish, neutral}` from next `5d` realized return
  - `relative_to_spy` in `{outperform, underperform, inline}` from next `5d` excess return vs `SPY`
- risk:
  - stop loss `5%`
  - trailing stop `1.5%`

## Main directional result

Test-set direction metrics across the top 5:

- base DeepSeek directional predictor:
  - mean direction accuracy: `33.72%`
  - mean direction macro-F1: `0.209953`
- fine-tuned DeepSeek directional predictor:
  - mean direction accuracy: `37.56%`
  - mean direction macro-F1: `0.237399`

So the fine-tuned directional branch improved prediction correctness, but only modestly.

## Trading translation result

Test-set cost-aware replay across the top 5:

- base directional branch:
  - mean total return: `10.68%`
  - mean max drawdown: `-1.51%`
- fine-tuned directional branch:
  - mean total return: `79.72%`
  - mean max drawdown: `-2.58%`

Compared with the earlier strict exact DeepSeek prompt winners:

- strict exact prompt mean return: `10.43%`
- strict exact prompt mean max drawdown: `-3.11%`

## Important caveat

The trading result is much stronger than the directional accuracy result. That means the current v1 directional trading policy is likely benefitting from:

- strong bullish bias in the fine-tuned outputs on this later test slice
- long-heavy exposure in names that had favorable realized trends
- risk exits recycling exposure without requiring high classification quality

So the `79.72%` mean trading return should **not** be read as proof that the model became a strong directional forecaster. It is stronger evidence that the current trade-mapping layer is too permissive.

## Per symbol

- `AVGO`
  - strict prompt return: `5.39%`
  - directional fine-tuned accuracy: `21.92%`
  - directional fine-tuned return: `137.77%`
  - interpretation: return is far too high relative to prediction accuracy; this is a red-flag overexposure case

- `NVDA`
  - strict prompt return: `4.83%`
  - directional fine-tuned accuracy: `32.74%`
  - directional fine-tuned return: `83.86%`

- `TSLA`
  - strict prompt return: `8.80%`
  - directional fine-tuned accuracy: `31.43%`
  - directional fine-tuned return: `151.37%`
  - interpretation: same red-flag pattern as `AVGO`

- `GOOGL`
  - strict prompt return: `23.18%`
  - directional fine-tuned accuracy: `51.72%`
  - directional fine-tuned return: `2.80%`
  - interpretation: this is the cleanest true directional-improvement case

- `XOM`
  - strict prompt return: `9.95%`
  - directional fine-tuned accuracy: `50.00%`
  - directional fine-tuned return: `22.81%`
  - interpretation: plausible improvement, but still with bullish skew

## Conclusion

This new branch is closer to the intended research design and does improve prediction correctness over the unfine-tuned base directional model.

But the first end-to-end trading mapping is too permissive and produces returns that are out of proportion to the actual classification quality. The branch is therefore useful as a directional research foundation, but not yet trustworthy as a trading strategy.

## Recommended next step

Keep this branch, but tighten it before trusting the trade results:

1. require `medium` or `high` strength for entries
2. require `relative_to_spy != underperform`
3. add a no-trade rule when predicted class entropy is weak or neutral-dominant
4. compare against a simple directional baseline like price-only momentum classification
5. add a capped holding period and no immediate re-entry rule
