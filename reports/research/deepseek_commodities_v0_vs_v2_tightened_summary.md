# Tightened Commodity V2 vs V0

Artifacts:

- [tightened compare JSON](C:\SVNProjects\Algoding\reports\research\deepseek_commodities_v0_vs_v2_tightened.json)

Policy:

- only `medium/high` strength predictions
- require relative confirmation
- require momentum confirmation
- apply re-entry cooldowns of `3/5/10` bars
- select the best definition on validation replay before testing

Aggregate strict-test comparison:

- `V0` mean total return: `1.90%`
- `V0` mean max drawdown: `-4.84%`
- original `V2` mean total return: `47.09%`
- original `V2` mean max drawdown: `-1.07%`
- tightened `V2` mean total return: `5.83%`
- tightened `V2` mean max drawdown: `-2.82%`
- tightened `V2` beat `V0` in `4/4` symbols
- tightened `V2` beat original `V2` in `0/4` symbols

Per symbol:

- `GLD`
  chosen rule: `high`, `rel=True`, `mom=True`, `cooldown=3`
  V0: `2.53%` / `-4.26%`
  original V2: `45.23%` / `-1.78%`
  tightened V2: `10.22%` / `-2.42%`
- `SLV`
  chosen rule: `high`, `rel=True`, `mom=True`, `cooldown=10`
  V0: `0.42%` / `-9.73%`
  original V2: `99.52%` / `-2.07%`
  tightened V2: `1.01%` / `-5.80%`
- `USO`
  chosen rule: `high`, `rel=True`, `mom=True`, `cooldown=3`
  V0: `5.27%` / `-3.69%`
  original V2: `40.72%` / `-0.25%`
  tightened V2: `11.24%` / `-2.78%`
- `UNG`
  chosen rule: `high`, `rel=True`, `mom=True`, `cooldown=3`
  V0: `-0.63%` / `-1.67%`
  original V2: `2.90%` / `-0.18%`
  tightened V2: `0.86%` / `-0.27%`
