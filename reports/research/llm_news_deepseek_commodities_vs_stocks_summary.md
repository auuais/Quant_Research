# DeepSeek Daily Overlay: Commodities vs Large-Cap Stocks

Artifacts:

- [commodities run](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_commodities_daily_run.json)
- [stocks run](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_stocks_daily_run_full.json)

Setup:

- model: `U:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B`
- windows: `1y`, `2y`, `3y`
- timeframe: `day`
- execution-aware replay
- fixed stop loss: `5%`
- trailing stop: `1.5%`

Commodity basket:

- `GLD`, `SLV`, `USO`, `UNG`

Large-cap stock basket:

- `AAPL`, `MSFT`, `NVDA`, `AMZN`, `META`, `GOOGL`, `TSLA`, `AVGO`, `AMD`, `NFLX`, `JPM`, `XOM`, `ORCL`, `CRM`, `WMT`, `COST`

## Commodity results

Overlay-vs-baseline summary across `12` symbol-window pairs:

- `entry_filter` beat baseline in `6/12`
- `exit_filter` beat baseline in `5/12`
- `combo` beat baseline in `5/12`

Best commodity overlays by symbol:

- `GLD / entry_filter`: `10.43%` average total return, `-4.53%` average max drawdown
- `SLV / exit_filter`: `5.35%` average total return, `-25.80%` average max drawdown
- `UNG / combo`: `0.30%` average total return, `-0.92%` average max drawdown
- `USO`: baseline remained best at `14.68%`

## Stocks vs commodities

Best-overlay-per-symbol comparison:

- stocks:
  - mean best average return: `20.76%`
  - mean overlay improvement over baseline: `4.32 pts`
  - mean best average max drawdown: `-9.64%`
- commodities:
  - mean best average return: `7.69%`
  - mean overlay improvement over baseline: `2.42 pts`
  - mean best average max drawdown: `-8.57%`

Variation:

- stocks:
  - return standard deviation: `21.60 pts`
  - improvement standard deviation: `6.11 pts`
- commodities:
  - return standard deviation: `6.23 pts`
  - improvement standard deviation: `3.90 pts`

## Interpretation

- The DeepSeek overlay is still stronger on large-cap stocks than on commodity ETFs.
- Commodities are more mixed:
  - `GLD` benefited clearly from the overlay.
  - `USO` was already strong without the overlay.
  - `SLV` improved only slightly and still carried deep drawdown.
  - `UNG` stayed close to flat.
- Stocks offer more upside and more symbol-specific dispersion.
- Commodities offer narrower but sometimes cleaner defensive improvements, especially on `GLD`.

## Current recommendation

Carry forward:

- stocks:
  - `AVGO / entry_filter`
  - `NVDA / exit_filter`
  - `TSLA / exit_filter`
  - `GOOGL / exit_filter`
- commodities:
  - `GLD / entry_filter`
  - `USO / baseline`
  - `SLV / exit_filter` as a lower-priority research branch

Do not prioritize `UNG` for the next stage.
