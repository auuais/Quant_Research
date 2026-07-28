# EVENT-PREMIA-V1 (Direction D2)

- thesis: `documented calendar/announcement premia persist and are harvestable with daily close-decision orders on index ETFs`
- history: `2005-01-03 -> 2026-07-27` (`5424` SPY sessions)
- declared variants: `14`, realized: `14`
- confirmation rule: `bootstrap 95% CI excludes zero in the early 70% AND late 30% independently, same sign, and mean effect >= 2x round-trip cost`
- decision: **`WATCHLIST`**

## Hypothesis Results

| Hypothesis | Events | Mean effect | Early CI excl. 0 | Late CI excl. 0 | > 2x cost | Verdict |
|---|---:|---:|:---:|:---:|:---:|---|
| `E1_pre_fomc_SPY` | `130` | `0.7` bps | no | no | no | **DROPPED** |
| `E1_pre_fomc_QQQ` | `130` | `14.2` bps | no | no | yes | **DROPPED** |
| `E2_overnight_SPY` | `5423` | `3.4` bps | yes | no | no | **DROPPED** |
| `E2_overnight_QQQ` | `5423` | `4.6` bps | yes | yes | yes | **SURVIVES** |
| `E2_overnight_panel` | `5423` | `4.4` bps | yes | yes | yes | **SURVIVES** |
| `E3_turn_of_month_base_last4_first3` | `260` | `61.8` bps | yes | yes | yes | **SURVIVES** |
| `E3_turn_of_month_wide_last5_first4` | `260` | `58.7` bps | yes | no | yes | **DROPPED** |
| `E4_opex_week` | `259` | `7.7` bps | no | no | yes | **DROPPED** |
| `E4_opex_day` | `259` | `-1.8` bps | no | no | no | **DROPPED** |
| `E4_quarter_end_day` | `87` | `11.8` bps | no | no | yes | **DROPPED** |
| `E5_gap_pead_5d` | `882` | `71.2` bps | yes | yes | yes | **SURVIVES** |
| `E5_gap_pead_20d` | `879` | `213.4` bps | yes | yes | yes | **SURVIVES** |
| `E6_vix_spike_p90` | `69` | `35.4` bps | no | no | yes | **DROPPED** |
| `E6_vix_spike_p95` | `56` | `51.7` bps | no | no | yes | **DROPPED** |

## E2: Overnight vs Intraday Decomposition

| Series | Overnight mean | Intraday mean | Overnight total | Intraday total | Overnight Sharpe | Intraday Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| `SPY` | `3.35` bps | `1.39` bps | `435.4%` | `68.3%` | `0.74` | `0.24` |
| `QQQ` | `4.62` bps | `1.82` bps | `936.5%` | `92.9%` | `0.93` | `0.26` |
| `panel` | `4.41` bps | `2.56` bps | `846.4%` | `213.7%` | `0.97` | `0.43` |

## Dropped Hypotheses

| Hypothesis | Reason |
|---|---|
| `E1_pre_fomc_SPY` | failed split-sample confirmation |
| `E1_pre_fomc_QQQ` | failed split-sample confirmation |
| `E2_overnight_SPY` | failed split-sample confirmation |
| `E3_turn_of_month_wide_last5_first4` | failed split-sample confirmation |
| `E4_opex_week` | failed split-sample confirmation |
| `E4_opex_day` | failed split-sample confirmation |
| `E4_quarter_end_day` | failed split-sample confirmation |
| `E6_vix_spike_p90` | failed split-sample confirmation |
| `E6_vix_spike_p95` | failed split-sample confirmation |

## Survivor Sleeves (net of costs)

| Sleeve | Gross Sharpe | Net Sharpe | Net CAGR | Worst DD | Time in market |
|---|---:|---:|---:|---:|---:|
| `E2_overnight_QQQ|primary` | `0.93` | `0.53` | `6.00%` | `-30.8%` | `100%` |
| `E2_overnight_QQQ|stress` | `0.93` | `-0.28` | `-4.16%` | `-67.0%` | `100%` |
| `E2_overnight_panel|primary` | `0.74` | `0.30` | `2.80%` | `-29.8%` | `100%` |
| `E2_overnight_panel|stress` | `0.74` | `-0.59` | `-7.06%` | `-80.8%` | `100%` |
| `E3_turn_of_month_base_last4_first3` | `0.49` | `0.47` | `4.51%` | `-33.6%` | `33%` |
| `E5_gap_pead_5d|primary` | `0.71` | `0.69` | `14.88%` | `-31.1%` | `45%` |
| `E5_gap_pead_5d|stress` | `0.71` | `0.64` | `13.57%` | `-31.2%` | `45%` |
| `E5_gap_pead_20d|primary` | `0.79` | `0.77` | `18.79%` | `-51.1%` | `86%` |
| `E5_gap_pead_20d|stress` | `0.79` | `0.74` | `17.87%` | `-51.3%` | `86%` |

## Data and Multiple Testing

- FOMC announcements scraped from federalreserve.gov: `130` (`2011-01-26` -> `2026-06-17`)
- years whose statement count differs from 8 scheduled meetings: `[2019, 2020, 2025]` (the Fed also publishes statements for unscheduled actions, so a year with more than 8 statement dates includes non-scheduled announcements)
- with `14` pre-registered variants over `5424` sessions, best-of-N selection alone is expected to produce an annualized Sharpe of about `0.3747` under a zero-edge null

## Decision

- survived confirmation: `['E2_overnight_QQQ', 'E2_overnight_panel', 'E3_turn_of_month_base_last4_first3', 'E5_gap_pead_5d', 'E5_gap_pead_20d']`
- dropped permanently: `['E1_pre_fomc_SPY', 'E1_pre_fomc_QQQ', 'E2_overnight_SPY', 'E3_turn_of_month_wide_last5_first4', 'E4_opex_week', 'E4_opex_day', 'E4_quarter_end_day', 'E6_vix_spike_p90', 'E6_vix_spike_p95']`
- sleeves meeting the standalone gate: `[]`
- watchlist: `['E2_overnight_QQQ|primary', 'E2_overnight_QQQ|stress', 'E2_overnight_panel|primary', 'E2_overnight_panel|stress', 'E3_turn_of_month_base_last4_first3', 'E5_gap_pead_5d|primary', 'E5_gap_pead_5d|stress', 'E5_gap_pead_20d|primary', 'E5_gap_pead_20d|stress']`

surviving an effect test is necessary but not sufficient: the sleeve must also clear the standalone Sharpe/drawdown gate net of costs
