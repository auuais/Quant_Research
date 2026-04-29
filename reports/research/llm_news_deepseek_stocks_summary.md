# DeepSeek LLM Stocks Overlay Research

Last updated: 2026-04-02

## Objective

Rerun the historical LLM news overlay with a stronger local model, broader stock coverage, and the fixes identified after the earlier local-model run.

Model used:

- `U:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B`

Artifacts:

- `reports/research/llm_news_sentiment_deepseek_stocks_run.json`
- `reports/research/llm_news_sentiment_deepseek_stocks_report.json`

## Adjustments From The Prior Run

The earlier local run using `google/flan-t5-base` had a clear positive/neutral bias. This rerun addresses that by:

- switching the sentiment engine to support causal LMs and scoring label likelihood directly
- using DeepSeek instead of the earlier seq2seq model
- aligning after-hours and weekend news to the next trading day
- carrying sparse-news sentiment forward with decay instead of forcing hard zeros
- calibrating sentiment thresholds per symbol from the observed score distribution
- broadening the stock universe to cover more liquid large-cap names

## Setup

Universe:

- `AAPL`
- `MSFT`
- `NVDA`
- `AMZN`
- `META`
- `GOOGL`
- `TSLA`
- `AVGO`
- `AMD`
- `NFLX`
- `JPM`
- `XOM`
- `ORCL`
- `CRM`
- `WMT`
- `COST`

Backtest setup:

- timeframe: `day`
- windows: `1y`, `2y`, `3y`
- execution-aware replay: `true`
- fixed stop loss: `5%`
- trailing stop: `1.5%`
- no same-day reentry: `false`
- regulatory fees, spread, market impact, stop slippage, and partial-fill constraints: enabled

Run size:

- `16` symbols
- `4` strategy variants per symbol
- `3` windows
- `192` total runs

## Sentiment Distribution

Average day-level label mix across the full stock universe:

- bullish: `39.5%`
- bearish: `51.4%`
- neutral: `9.1%`

This is materially better than the earlier local run because the model now produces a realistic number of bearish news days. It removes the main trust issue from the earlier FLAN pass.

## Best Results

Top aggregate overlay contenders:

1. `AVGO / avgo_momentum_llm_entry_filter`
   - profitable windows: `3/3`
   - average total return: `62.16%`
   - average max drawdown: `-5.59%`
   - composite score: `0.816942`
2. `NVDA / nvda_momentum_llm_exit_filter`
   - profitable windows: `3/3`
   - average total return: `58.71%`
   - average max drawdown: `-15.98%`
   - composite score: `0.682209`
3. `TSLA / tsla_momentum_llm_exit_filter`
   - profitable windows: `3/3`
   - average total return: `51.56%`
   - average max drawdown: `-21.70%`
   - composite score: `0.578712`
4. `GOOGL / googl_momentum_llm_exit_filter`
   - profitable windows: `3/3`
   - average total return: `33.57%`
   - average max drawdown: `-7.84%`
   - composite score: `0.478970`
5. `XOM / xom_momentum_llm_exit_filter`
   - profitable windows: `3/3`
   - average total return: `17.00%`
   - average max drawdown: `-3.41%`
   - composite score: `0.358472`

## Overlay Vs Baseline

Across `48` symbol-window pairs:

- entry filter beat baseline in `19`
- exit filter beat baseline in `25`
- combo beat baseline in `17`

Best symbol-level improvements versus the momentum baseline:

- `AVGO`
  - best overlay: `entry_filter`
  - average return improvement: `+23.22 pts`
- `TSLA`
  - best overlay: `exit_filter`
  - average return improvement: `+11.85 pts`
- `CRM`
  - best overlay: `entry_filter`
  - average return improvement: `+8.41 pts`
- `GOOGL`
  - best overlay: `exit_filter`
  - average return improvement: `+5.61 pts`
- `MSFT`
  - best overlay: `combo`
  - average return improvement: `+4.57 pts`
- `JPM`
  - best overlay: `exit_filter`
  - average return improvement: `+4.44 pts`

Interpretation:

- `exit_filter` is the strongest general-purpose overlay type in this DeepSeek run
- `entry_filter` works very well for selected names such as `AVGO` and `CRM`
- `combo` is more selective, but it is useful on `MSFT` and `AMZN`

## Weak Fits

The overlay underperformed the baseline on these names:

- `META`
- `AMD`
- `ORCL`
- `WMT`
- `NFLX`

This means the LLM layer should not be applied uniformly across the whole stock basket.

## Assessment

The DeepSeek rerun is more credible than the earlier local-model pass because the bearish/neutral coverage problem is largely fixed. The tradeoff is that some of the earlier headline wins no longer hold up once the overlay becomes stricter and more realistic.

The correct reading is:

- trust in the sentiment engine improved
- broad-market average improvement is still selective, not universal
- the overlay is best used as a symbol-specific timing layer on top of momentum, not as a blanket replacement for the baseline

## Recommendation

Best daily-stock LLM overlay candidates for the next hourly follow-up:

- `AVGO / entry_filter`
- `NVDA / exit_filter`
- `TSLA / exit_filter`
- `GOOGL / exit_filter`
- `CRM / entry_filter`
- `JPM / exit_filter`
- `XOM / exit_filter`

Names that should stay baseline-only for now:

- `META`
- `AMD`
- `ORCL`
- `WMT`
- `NFLX`
