# Deep Learning Research Summary

Date: 2026-04-01

## Scope

This report summarizes the first dedicated CUDA-backed deep-learning research branch.

Markets tested:

- `commodities`
- `stocks`

Symbols covered:

- Commodities: `GLD`, `SLV`, `USO`
- Stocks: `AAPL`, `MSFT`, `NVDA`, `AMZN`, `META`

Windows:

- `1y`
- `2y`
- `3y`

Execution and risk assumptions:

- timeframe: `hour`
- regular-hours only
- fixed stop loss: `5%`
- trailing stop: `1.5%`
- no same-day reentry
- cost-aware replay with:
  - spread
  - market impact
  - extra stop slippage
  - FINRA TAF
  - partial-fill cap

Models tested:

- `torch_mlp`
- `torch_lstm`
- `torch_cnn`

Primary artifact:

- [dl_commodities_stocks_hourly_gpu_run.json](C:\SVNProjects\Algoding\reports\research\dl_commodities_stocks_hourly_gpu_run.json)

## Top DL Contenders

1. `SLV / torch_lstm`
   - profitable windows: `3/3`
   - average total return: `26.64%`
   - average annualized return: `19.49%`
   - average max drawdown: `-10.98%`

2. `SLV / torch_mlp`
   - profitable windows: `3/3`
   - average total return: `8.39%`
   - average max drawdown: `-8.86%`

3. `NVDA / torch_mlp`
   - profitable windows: `2/3`
   - average total return: `22.46%`
   - average max drawdown: `-7.23%`
   - standout `3y` total return: `62.74%`

4. `SLV / torch_cnn`
   - profitable windows: `3/3`
   - average total return: `18.56%`
   - average max drawdown: `-12.82%`

5. `NVDA / torch_lstm`
   - profitable windows: `3/3`
   - average total return: `12.18%`
   - average max drawdown: `-12.65%`

6. `META / torch_mlp`
   - profitable windows: `3/3`
   - average total return: `6.98%`
   - average max drawdown: `-7.80%`

7. `USO / torch_lstm`
   - profitable windows: `3/3`
   - average total return: `13.74%`
   - average max drawdown: `-7.13%`

8. `GLD / torch_cnn`
   - profitable windows: `3/3`
   - average total return: `7.85%`
   - average max drawdown: `-11.14%`

## Cross-Branch Interpretation

- `SLV` remains the strongest learned-model family overall.
- The best dedicated DL result is `SLV / torch_lstm`.
- `NVDA` is still the strongest stock-side DL research family.
- `USO / torch_lstm` is cleaner than the weaker tabular DL variants for `USO`.
- `GLD / torch_cnn` is the most credible DL candidate inside the gold branch.

## Comparison To Earlier ML

Most important takeaways:

- The DL branch is now real and GPU-backed.
- DL does add value in some symbols, especially `SLV`.
- The gains are not universal.

Examples:

- `SLV`
  - `torch_lstm` beats the earlier simple ML baselines on average total return and average annualized return.
  - This is the clearest positive DL result so far.

- `NVDA`
  - `torch_mlp` is strong, but not clearly dominant over the earlier best ML baseline.
  - `torch_lstm` is more consistent than some NVDA ML variants, but less explosive than the best `3y` tabular DL run.

- `META`
  - DL is usable, but not clearly stronger than the earlier tree baseline.

- `USO`
  - sequence DL is better than the weaker DL tabular variant, but still not obviously better than the earlier tree baseline.

## Current Conclusion

The current DL recommendation is:

1. Keep `SLV / torch_lstm` as the lead DL contender.
2. Keep `NVDA / torch_mlp` and `NVDA / torch_lstm` as the lead stock-side DL contenders.
3. Keep `USO / torch_lstm` and `GLD / torch_cnn` as secondary commodity DL candidates.
4. Do not broaden the DL branch further until we compare these directly against the best rule-based and ML contenders for the same symbols.

## Next Comparison Set

Direct head-to-head comparisons should now focus on:

- `SLV / slv_momentum_aggressive` vs `SLV / torch_lstm`
- `NVDA / nvda_mean_reversion_daily` vs `NVDA / torch_mlp`
- `USO / uso_momentum_daily` vs `USO / torch_lstm`
