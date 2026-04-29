# LLM News Sentiment Overlay Research

Last updated: 2026-04-02

Artifact:

- [llm_news_sentiment_overlay_run.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_overlay_run.json)

## Scope

Symbols tested:

- Commodities via ETFs: `SLV`, `GLD`
- Major stocks: `AAPL`, `MSFT`, `NVDA`, `AMZN`, `META`

Windows:

- `1y`
- `2y`
- `3y`

## Engine

Baseline:

- `20`-day rule-based momentum

LLM overlay:

- local open-weight model: `google/flan-t5-base`
- Alpaca historical news headlines and summaries
- news bundled by effective trading day
- after-hours and weekend headlines mapped to the next trading day
- sentiment score carried forward with short decay to reflect a `1-5` day impact assumption

Overlay variants:

- `momentum_baseline`
- `momentum_llm_entry_filter`
- `momentum_llm_exit_filter`
- `momentum_llm_combo`

Execution-aware assumptions:

- daily bars
- fixed stop loss: `5%`
- trailing stop: `1.5%`
- spread: `1 bp`
- market impact: `1 bp`
- extra stop slippage: `2 bps`
- FINRA TAF enabled
- SEC sell fee assumed `0.0` for this replay path

## News Coverage

- `AAPL`: `7300` articles, `556` news-trading days
- `MSFT`: `3900` articles, `324` news-trading days
- `NVDA`: `6950` articles, `400` news-trading days
- `AMZN`: `4050` articles, `334` news-trading days
- `META`: `2650` articles, `258` news-trading days
- `GLD`: `1867` articles, `720` news-trading days
- `SLV`: `668` articles, `509` news-trading days

## Headline Findings

Best overall overlay winners by symbol:

- `NVDA`: `nvda_momentum_llm_entry_filter`
  - avg total return: `56.95%`
  - avg max drawdown: `-13.24%`
- `MSFT`: `msft_momentum_llm_combo`
  - avg total return: `18.18%`
  - avg max drawdown: `-3.97%`
- `GLD`: `gld_momentum_llm_entry_filter`
  - avg total return: `4.74%`
  - avg max drawdown: `-5.46%`
- `AAPL`: `aapl_momentum_llm_exit_filter`
  - avg total return: `8.28%`
  - avg max drawdown: `-8.69%`
- `SLV`: `slv_momentum_llm_exit_filter`
  - avg total return: `5.44%`
  - avg max drawdown: `-25.81%`

Symbols where baseline remained strongest:

- `META`

Weak symbol:

- `AMZN` remained poor overall, but the LLM overlays materially reduced the loss and turned the `3y` window positive.

## Improvement Versus Baseline

Across `21` symbol-window pairs:

- entry filter improved return in `10`
- exit filter improved return in `11`
- combo improved return in `9`

Most convincing improvements:

- `MSFT`
  - baseline `3y`: `0.10%`
  - combo `3y`: `25.06%`
- `AMZN`
  - baseline `3y`: `-10.44%`
  - entry filter `3y`: `7.50%`
  - combo `3y`: `5.33%`
- `GLD`
  - baseline `1y`: effectively flat
  - entry filter `1y`: `5.13%`
- `NVDA`
  - baseline `3y`: `121.17%`
  - entry filter `3y`: `128.77%`

Clear failures:

- `META`
  - all LLM overlays underperformed the baseline in `1y`, `2y`, and `3y`
- `SLV`
  - entry filter and combo hurt performance in all windows

## Interpretation

The LLM overlay is promising as a secondary timing filter, not as a standalone strategy.

What worked:

- entry filtering improved several stock and gold cases
- exit filtering improved `SLV`, `AAPL`, `AMZN`, `MSFT`, and `NVDA`
- the overlay clearly helped some weak baselines, especially `MSFT` and `AMZN`

What did not work:

- the overlay was not universally beneficial
- `META` is a clear counterexample
- the model still shows a positive-tone bias: very few strongly negative days were produced

## Trust Level

This is a valid first-pass historical LLM overlay result, but not yet a production-grade sentiment engine.

Main limitations:

- local small open-weight model, not a frontier finance-tuned LLM
- no article-source weighting yet
- no event-type separation yet
- positive sentiment bias remains visible
- no benchmark yet against non-LLM text models like FinBERT or dictionary sentiment

## Recommendation

1. Keep rule-based momentum as the production backbone.
2. Keep the LLM overlay as a research-only timing layer.
3. Carry forward `NVDA`, `MSFT`, `GLD`, `AAPL`, and `SLV` for deeper text-model comparison.
4. Next comparison should be:
   - `LLM overlay` vs `FinBERT / classic sentiment` on the same news bundles
   - `LLM overlay` vs baseline on a fixed out-of-sample split
