# Qwen3 vs DeepSeek on Top 5 U.S. Daily Stocks

Artifacts:

- [Qwen run](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_qwen3_top5_stocks_daily_run.json)
- [DeepSeek run](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_stocks_daily_run_full.json)

Model paths:

- `Qwen3-1.7B`: `U:\models\Qwen3-1.7B`
- `DeepSeek-R1-Distill-Qwen-1.5B`: `U:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B`

Important hardware note:

- `Qwen3-4B` was also downloaded to `U:\models\Qwen3-4B`
- but on the local `RTX 3070 Ti 8GB`, it spilled to CPU offload and was too slow for the full historical pass
- the final comparison below uses `Qwen3-1.7B` because it fits fully on GPU

Universe:

- `AVGO`
- `NVDA`
- `TSLA`
- `GOOGL`
- `XOM`

Setup:

- timeframe: `day`
- windows: `1y`, `2y`, `3y`
- execution-aware replay
- fixed stop loss: `5%`
- trailing stop: `1.5%`

## Aggregate comparison

Best-per-symbol result for each model:

- Qwen3 mean average return: `34.35%`
- DeepSeek mean average return: `44.44%`
- delta: `-10.10 pts`

- Qwen3 mean average max drawdown: `-7.94%`
- DeepSeek mean average max drawdown: `-10.84%`
- delta: `+2.90 pts`

Interpretation:

- DeepSeek is still stronger on return
- Qwen3 is more defensive
- Qwen3 did not beat DeepSeek overall on this top-5 stock set

## Best strategy by symbol

| Symbol | Qwen3 best | Qwen3 return | Qwen3 max DD | DeepSeek best | DeepSeek return | DeepSeek max DD |
|---|---|---:|---:|---|---:|---:|
| `AVGO` | `baseline` | `35.04%` | `-9.01%` | `entry_filter` | `62.16%` | `-5.59%` |
| `NVDA` | `exit_filter` | `50.72%` | `-8.21%` | `exit_filter` | `58.71%` | `-15.98%` |
| `TSLA` | `exit_filter` | `37.52%` | `-14.76%` | `exit_filter` | `51.56%` | `-21.70%` |
| `GOOGL` | `exit_filter` | `30.72%` | `-4.99%` | `exit_filter` | `33.57%` | `-7.84%` |
| `XOM` | `baseline` | `17.74%` | `-2.73%` | `entry_filter` | `16.23%` | `-3.06%` |

## Overlay hit rate

Across `15` symbol-window pairs:

- Qwen3:
  - `entry_filter` beat baseline in `1/15`
  - `exit_filter` beat baseline in `9/15`
  - `combo` beat baseline in `1/15`

Interpretation:

- Qwen3 behaves much more like an `exit_filter` model than a broad entry-timing model
- it did not produce the same entry-filter gains that DeepSeek produced on `AVGO`
- `XOM` and `AVGO` reverted to baseline under Qwen3

## Sentiment distribution

Top-5 universe average news-day labels:

- Qwen3:
  - bullish: `30.66%`
  - bearish: `66.85%`
  - neutral: `2.49%`
- DeepSeek:
  - bullish: `45.03%`
  - bearish: `46.43%`
  - neutral: `8.54%`

Interpretation:

- Qwen3 is materially more bearish than DeepSeek on the same news bundles
- this likely explains why Qwen3 was more defensive and weaker on upside capture

## Conclusion

- DeepSeek remains the stronger local LLM for this stock-overlay task
- Qwen3-1.7B is a credible fallback model when a smaller full-GPU model is needed
- but it should not replace DeepSeek as the primary local stock-news model
