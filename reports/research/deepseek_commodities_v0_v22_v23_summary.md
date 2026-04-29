# Commodity V0 vs V2 vs V2-2 vs V2-3

Artifacts:

- [comparison JSON](C:\SVNProjects\Algoding\reports\research\deepseek_commodities_v0_v22_v23.json)

Policy for `V2-3`:

- keep hard stop loss `5%`
- keep trailing stop `1.5%`
- allow only `medium` or `high` strength predictions
- no relative confirmation
- no momentum confirmation
- no cooldown

Aggregate strict-test comparison:

- `V0`: `1.90%` return, `-4.84%` max drawdown
- `V2`: `47.09%` return, `-1.07%` max drawdown
- `V2-2`: `5.83%` return, `-2.82%` max drawdown
- `V2-3`: `62.63%` return, `-1.07%` max drawdown
- `V2-3` mean direction accuracy: `42.35%`
- `V2-3` beat `V0` on return in `4/4` symbols
- `V2-3` beat `V2-2` on return in `4/4` symbols

Per symbol:

- `GLD`
  `V0`: `2.53%` / `-4.26%`
  `V2`: `45.23%` / `-1.78%`
  `V2-2`: `10.22%` / `-2.42%`
  `V2-3`: `45.26%` / `-1.78%` / accuracy `52.38%`
- `SLV`
  `V0`: `0.42%` / `-9.73%`
  `V2`: `99.52%` / `-2.07%`
  `V2-2`: `1.01%` / `-5.80%`
  `V2-3`: `161.85%` / `-2.05%` / accuracy `47.92%`
- `USO`
  `V0`: `5.27%` / `-3.69%`
  `V2`: `40.72%` / `-0.25%`
  `V2-2`: `11.24%` / `-2.78%`
  `V2-3`: `40.54%` / `-0.25%` / accuracy `38.26%`
- `UNG`
  `V0`: `-0.63%` / `-1.67%`
  `V2`: `2.90%` / `-0.18%`
  `V2-2`: `0.86%` / `-0.27%`
  `V2-3`: `2.89%` / `-0.18%` / accuracy `30.85%`
