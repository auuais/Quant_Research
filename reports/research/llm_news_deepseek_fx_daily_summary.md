# DeepSeek FX Daily Overlay Summary

Artifacts:

- [llm_news_sentiment_deepseek_fx_daily_run.json](C:\SVNProjects\Algoding\reports\research\llm_news_sentiment_deepseek_fx_daily_run.json)

Setup:

- model: `U:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B`
- market: `fx`
- symbols: `EURUSD`, `USDJPY`, `GBPUSD`, `AUDUSD`, `USDCAD`, `USDCHF`, `EURJPY`
- windows: `1y`, `2y`, `3y`
- timeframe: `day`
- execution-aware replay
- fixed stop loss: `5%`
- trailing stop: `1.5%`
- no same-day reentry: not used in this branch

FX-specific assumptions:

- Yahoo daily OHLC bars for the currency pairs
- proxy-news mapping:
  - `EURUSD -> FXE, UUP`
  - `USDJPY -> FXY, UUP`
  - `GBPUSD -> FXB, UUP`
  - `AUDUSD -> FXA, UUP`
  - `USDCAD -> FXC, UUP`
  - `USDCHF -> FXF, UUP`
  - `EURJPY -> FXE, FXY`
- cost-aware execution:
  - quoted spread: `1.5 bps`
  - market impact: `0.5 bps`
  - stop slippage: `1.5 bps`
  - no SEC/FINRA fees

Main result:

- overlay symbol-window pairs tested: `21`
- `entry_filter` beat baseline in `18/21`
- `exit_filter` beat baseline in `3/21`
- `combo` beat baseline in `20/21`

Best overall FX overlay:

- `EURUSD / eurusd_momentum_llm_combo`
- average total return: `0.47%`
- average max drawdown: `-0.23%`
- profitable windows: `2/3`

Most improved pairs versus their own baseline:

- `AUDUSD`: baseline `-7.18%` -> best overlay `-0.24%` using `combo`
- `USDJPY`: baseline `-6.34%` -> best overlay `-2.10%` using `entry_filter`
- `GBPUSD`: baseline `-3.21%` -> best overlay `-1.17%` using `combo`

Interpretation:

- The DeepSeek overlay helps FX timing more than the raw baseline for most pair-window tests.
- The improvement is mostly defensive: it reduces losses and drawdowns rather than creating large absolute returns.
- `combo` and `entry_filter` are the only viable FX overlay modes from this run.
- `exit_filter` is weak on FX in the current design.
- Only `EURUSD` was net positive on an average-return basis across `1y`, `2y`, and `3y`.

Current recommendation:

- Keep `EURUSD / combo` as the primary FX LLM overlay candidate.
- Keep `AUDUSD / combo`, `USDJPY / entry_filter`, and `GBPUSD / combo` as secondary research candidates.
- Do not promote the FX overlay branch yet; the absolute edge is still too small.
