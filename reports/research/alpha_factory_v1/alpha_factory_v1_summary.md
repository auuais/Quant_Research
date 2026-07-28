# ALPHA-FACTORY-V1 (Direction D3)

- thesis: `automated search over a formulaic price/volume DSL finds factor ensembles with higher validation RankIC than the hand-built winners, and they survive a once-run test period`
- panel: `2005-01-03 -> 2026-07-27` (`5424` days x `100` symbols)
- splits: train `2005-01-03 -> 2024-06-30`, validation `2024-07-01 -> 2025-06-30`, test `2025-07-01 -> 2026-07-27` (embargo `5` days)
- declared search budget: `4000`, realized evaluations: `4000`
- decision: **`WATCHLIST`**

## Severe caveat on this panel

the panel is UNIVERSE_100 loaded from Yahoo over ~21 years, which is far longer than the ~3.5-year Alpaca panel the registry's null assumed; it is also a currently-listed mega-cap set, so survivorship bias is SEVERE over this span and any accepted factor partly encodes 'these particular names went up'

## Search

- expressions scored: `3778`
- passing train thresholds: `482`
- best absolute train RankIC: `0.02677`
- generator mix: `{'seed': 17, 'mutation': 1701, 'random': 1678, 'crossover': 604}`
- LLM proposer: `not requested`

## Validation-Period Context (read this before the factor table)

Whether the acceptance stage can discriminate at all depends on whether the *incumbent* factors work over the validation window. Rebuilt on this same panel:

| Baseline | Validation IC | Validation ICIR | Test IC | Test ICIR |
|---|---:|---:|---:|---:|
| `baseline_ret_20d` | `-0.03834` | `-2.5596` | `0.002304` | `0.1749` |
| `baseline_resid_mom_60d` | `-0.008793` | `-0.5897` | `0.032616` | `2.3346` |

the incumbent factors also fail the acceptance bar over this validation window, so the window is hostile to the entire cross-sectional price/volume family; a single contiguous validation year cannot distinguish 'mining produces nothing' from 'nothing worked that year'

Required fix for the next registration: replace the single contiguous validation year with purged multi-fold walk-forward validation so acceptance is not decided by one regime

## Accepted Factors (validation-accepted, then scored once on test)

| Expression | Origin | Dir | Train IC | Val IC | Val ICIR | Test IC |
|---|---|---:|---:|---:|---:|---:|
| `cs_rank(mul(mul(abs(low), ts_rank(open, 20)), sub(close, low)))` | `crossover` | `-1` | `-0.0257` | `+0.0249` | `+2.183` | `+0.0190` |

## Validation -> Test Decay (the pre-registered kill test)

- accepted factors: `1`
- mean validation RankIC: `0.024858`
- mean test RankIC: `0.018989`
- decay: `0.2361` against a kill threshold of `0.5`
- kill rule fired: **`False`**
- fraction of accepted factors with positive test IC: `1.0`

## Book Comparison (top-20 long only, net of primary costs)

| Book | Period | Net CAGR | Net Sharpe | Worst DD | Beta vs EW | Turnover |
|---|---|---:|---:|---:|---:|---:|
| `baseline_resid_mom_60d` | `test` | `62.88%` | `2.54` | `-7.2%` | `1.1098` | `0.38` |
| `baseline_resid_mom_60d` | `validation` | `24.61%` | `1.45` | `-14.6%` | `0.6945` | `0.45` |
| `baseline_ret_20d` | `test` | `48.27%` | `2.31` | `-7.6%` | `1.0645` | `0.78` |
| `baseline_ret_20d` | `validation` | `12.91%` | `0.78` | `-14.7%` | `0.7608` | `0.79` |
| `ensemble_rank_average` | `test` | `26.50%` | `1.90` | `-6.8%` | `0.9178` | `0.82` |
| `ensemble_rank_average` | `validation` | `13.64%` | `0.67` | `-16.2%` | `1.1549` | `0.77` |

## Decision

- status: **`WATCHLIST`**
- ensemble test Sharpe: `1.898`
- incumbent `resid_mom_60d` test Sharpe: `2.545`
- `ret_20d` test Sharpe: `2.3064`

the kill rule is evaluated first and overrides a favourable book comparison: an ensemble whose component ICs decayed more than 50% is not a finding, it is an overfit on too small a panel
