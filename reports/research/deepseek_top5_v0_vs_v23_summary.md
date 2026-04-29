# Stock Top-5 V0 vs V2-3

Artifacts:

- [comparison JSON](C:\SVNProjects\Algoding\reports\research\deepseek_top5_v0_vs_v23.json)

Policy for `V2-3`:

- keep hard stop loss `5%`
- keep trailing stop `1.5%`
- allow only `medium` or `high` strength predictions
- no relative confirmation
- no momentum confirmation
- no cooldown

Aggregate strict-test comparison:

- `V0`: `10.43%` return, `-3.11%` max drawdown
- `V2-3`: `79.25%` return, `-2.61%` max drawdown
- `V2-3` mean direction accuracy: `37.56%`
- `V2-3` beat `V0` on return in `4/5` symbols

Per symbol:

- `AVGO`
  `V0`: `5.39%` / `-3.36%`
  `V2-3`: `137.31%` / `-1.88%` / accuracy `21.92%`
- `NVDA`
  `V0`: `4.83%` / `-3.92%`
  `V2-3`: `83.33%` / `-4.11%` / accuracy `32.74%`
- `TSLA`
  `V0`: `8.80%` / `-4.61%`
  `V2-3`: `149.42%` / `-1.79%` / accuracy `31.43%`
- `GOOGL`
  `V0`: `23.18%` / `-1.86%`
  `V2-3`: `2.76%` / `-2.64%` / accuracy `51.72%`
- `XOM`
  `V0`: `9.95%` / `-1.81%`
  `V2-3`: `23.40%` / `-2.64%` / accuracy `50.00%`
