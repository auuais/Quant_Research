# DeepSeek Daily Overlay: Non-U.S. Stocks via Alpaca ADR/Global Listings

Artifacts:

- [run JSON](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_non_us_stocks_daily_run.json)

Important scope note:

- This branch uses `U.S.-listed ADRs/global stocks` available through Alpaca.
- It is not direct trading on foreign exchanges.

Basket:

- `ASML`, `TSM`, `NVO`, `SAP`, `BABA`, `PDD`, `HSBC`, `TM`, `RIO`, `SHEL`, `SONY`, `INFY`

Setup:

- model: `U:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B`
- windows: `1y`, `2y`, `3y`
- timeframe: `day`
- execution-aware replay
- fixed stop loss: `5%`
- trailing stop: `1.5%`

## Overlay hit rate

Across `36` symbol-window pairs:

- `entry_filter` beat baseline in `16/36`
- `exit_filter` beat baseline in `18/36`
- `combo` beat baseline in `17/36`

The branch is viable, but weaker than the U.S. large-cap stock branch.

## Top outcomes

Best overall rows:

- `TSM / exit_filter`: `20.15%` average total return, `-5.70%` average max drawdown
- `BABA / exit_filter`: `12.52%`, `-7.55%`
- `ASML / baseline`: `9.04%`, `-6.10%`
- `SHEL / baseline`: `3.58%`, `-0.84%`

Best overlay improvement by symbol:

- `PDD`: `-5.95% -> 2.14%` with `exit_filter`
- `TSM`: `13.35% -> 20.15%` with `exit_filter`
- `NVO`: `0.30% -> 3.72%` with `exit_filter`
- `INFY`: `-3.14% -> -0.86%` with `entry_filter`

## Aggregate comparison vs U.S. large-cap stock branch

Best-overlay-per-symbol statistics:

- non-U.S. ADR/global stock basket:
  - mean best average return: `4.33%`
  - mean overlay improvement over baseline: `1.97 pts`
  - mean best average max drawdown: `-2.98%`
- U.S. large-cap stock basket:
  - mean best average return: `20.76%`
  - mean overlay improvement over baseline: `4.32 pts`
  - mean best average max drawdown: `-9.64%`

Interpretation:

- The non-U.S. ADR/global basket is more defensive and lower-volatility than the U.S. large-cap basket.
- The LLM overlay still adds value, but the edge is smaller and more selective.
- `exit_filter` is the best mode in this branch, unlike the broader U.S. stock branch where entry and combo were also strong.

## Current recommendation

Carry forward:

- `TSM / exit_filter`
- `BABA / exit_filter`
- `PDD / exit_filter`
- `NVO / exit_filter`

Keep as baseline-only:

- `ASML`
- `SHEL`
- `HSBC`

Do not prioritize:

- `TM`
- `SONY`
- `INFY`
- `RIO`
