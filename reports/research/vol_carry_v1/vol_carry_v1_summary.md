# VOL-CARRY-V1 (Direction D1)

- thesis: `the VIX term-structure slope plus a realized-vol forecast times short-volatility ETP exposure well enough to beat both always-short-vol and cash after costs, including crises`
- execution: `signals known at T close; position held for the T+1 close-to-close return of the real ETP; turnover charged at the VIX-ETP cost profile`
- evaluation: `2013-11-06` -> `2026-07-27` (`3197` days)
- costs: primary `3.0` bps/side, stress `10.0` bps/side
- pre-registered variants: `12`
- decision: **`WATCHLIST`** (best variant `slope_02_vrp`)

## Methodology amendment

Pre-registration said the tradeable leg would be reconstructed from ETP returns. During implementation the reconstruction was tested and rejected: a synthetic -1x series rebuilt from VXX loses 48.7% over Volmageddon while the real -1x product lost 90.4%, because VIX futures moved between the 16:00 equity close and 16:15 futures settlement. The backtest therefore uses the real closes of the product actually holdable (SVXY throughout, whose target exposure genuinely fell from -1x to -0.5x on 2018-02-28). This tightens the methodology and was decided before any strategy results were examined.

## Full-History Net Results (primary costs)

| Strategy | Total | CAGR | Ann vol | Sharpe | Worst DD | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_cash_only` | `24.6%` | `1.75%` | `0.3%` | `6.55` | `-0.3%` | `6.66` |
| `baseline_spy` | `420.3%` | `13.88%` | `17.1%` | `0.85` | `-33.7%` | `0.41` |
| `slope_02` | `391.6%` | `13.37%` | `40.5%` | `0.52` | `-61.2%` | `0.22` |
| `slope_02_vrp` | `348.6%` | `12.56%` | `38.9%` | `0.51` | `-61.7%` | `0.20` |
| `slope_00` | `294.9%` | `11.43%` | `42.0%` | `0.48` | `-66.1%` | `0.17` |
| `slope_00_vrp` | `258.6%` | `10.59%` | `40.5%` | `0.46` | `-66.6%` | `0.16` |
| `slope_05` | `122.5%` | `6.51%` | `37.7%` | `0.36` | `-66.1%` | `0.10` |
| `baseline_always_short_vol` | `-3.5%` | `-0.28%` | `53.4%` | `0.36` | `-95.2%` | `-0.00` |
| `slope_05_vrp` | `83.4%` | `4.90%` | `36.1%` | `0.32` | `-59.3%` | `0.08` |
| `slope_02_longvol` | `47.7%` | `3.12%` | `51.1%` | `0.32` | `-78.9%` | `0.04` |
| `slope_02_vrp_longvol` | `34.8%` | `2.38%` | `49.8%` | `0.30` | `-73.6%` | `0.03` |
| `slope_00_longvol` | `18.6%` | `1.35%` | `52.3%` | `0.29` | `-81.3%` | `0.02` |
| `slope_00_vrp_longvol` | `7.7%` | `0.59%` | `51.1%` | `0.27` | `-76.5%` | `0.01` |
| `slope_05_longvol` | `-33.2%` | `-3.12%` | `48.9%` | `0.18` | `-80.3%` | `-0.04` |
| `slope_05_vrp_longvol` | `-44.9%` | `-4.59%` | `47.7%` | `0.14` | `-80.0%` | `-0.06` |

## Window Aggregates (primary costs, excludes full-history window)

| Strategy | Windows | Mean total | Mean Sharpe | Worst DD | Profitable |
|---|---:|---:|---:|---:|---:|
| `baseline_cash_only` | `26` | `0.97%` | `7.85` | `-0.09%` | `73%` |
| `baseline_spy` | `26` | `7.54%` | `1.28` | `-33.72%` | `81%` |
| `baseline_always_short_vol` | `26` | `9.44%` | `0.75` | `-93.07%` | `69%` |
| `slope_02_vrp` | `26` | `7.97%` | `0.68` | `-51.08%` | `58%` |
| `slope_02` | `26` | `9.57%` | `0.67` | `-58.58%` | `62%` |
| `slope_00_vrp` | `26` | `7.51%` | `0.65` | `-49.62%` | `65%` |
| `slope_00` | `26` | `9.15%` | `0.63` | `-54.12%` | `65%` |
| `slope_02_vrp_longvol` | `26` | `5.04%` | `0.42` | `-62.12%` | `46%` |
| `slope_02_longvol` | `26` | `6.73%` | `0.41` | `-69.22%` | `46%` |
| `slope_00_vrp_longvol` | `26` | `4.86%` | `0.40` | `-62.59%` | `46%` |
| `slope_00_longvol` | `26` | `6.58%` | `0.39` | `-69.23%` | `50%` |
| `slope_05_vrp` | `26` | `3.74%` | `0.38` | `-46.49%` | `54%` |
| `slope_05` | `26` | `5.36%` | `0.38` | `-54.73%` | `62%` |
| `slope_05_longvol` | `26` | `2.79%` | `0.14` | `-59.42%` | `50%` |
| `slope_05_vrp_longvol` | `26` | `1.16%` | `0.12` | `-52.65%` | `46%` |

## Crisis Slices (primary costs, net total return)

| Strategy | `aug_2015_vol_spike` | `brexit_2016` | `volmageddon_2018_02` | `covid_2020_03` | `bear_2022_h1` | `yen_unwind_2024_08` | `tariff_2025_04` |
|---|---|---|---|---|---|---|---|
| `slope_02_vrp` | `-24.9%` | `6.0%` | `-39.2%` | `-17.6%` | `-22.0%` | `-20.3%` | `0.1%` |
| `baseline_always_short_vol` | `-54.2%` | `-11.1%` | `-90.4%` | `-55.3%` | `-21.7%` | `-21.7%` | `-25.3%` |
| `baseline_cash_only` | `0.0%` | `-0.0%` | `0.0%` | `0.3%` | `0.1%` | `0.2%` | `0.1%` |
| `baseline_spy` | `-8.4%` | `0.5%` | `-5.9%` | `-33.7%` | `-20.0%` | `-2.1%` | `-6.2%` |

## Leverage-Era Split (the -1x era is the honest stress test)

| Strategy | Era | Total | CAGR | Sharpe | Worst DD |
|---|---|---:|---:|---:|---:|
| `slope_02_vrp` | `minus_1x_era_to_2018_02_27` | `56.2%` | `10.92%` | `0.47` | `-61.7%` |
| `slope_02_vrp` | `minus_0p5x_era_from_2018_02_28` | `187.2%` | `13.41%` | `0.58` | `-38.8%` |
| `baseline_always_short_vol` | `minus_1x_era_to_2018_02_27` | `-58.0%` | `-18.26%` | `0.31` | `-93.1%` |
| `baseline_always_short_vol` | `minus_0p5x_era_from_2018_02_28` | `129.7%` | `10.43%` | `0.46` | `-62.2%` |
| `baseline_cash_only` | `minus_1x_era_to_2018_02_27` | `0.8%` | `0.18%` | `0.66` | `-0.3%` |
| `baseline_cash_only` | `minus_0p5x_era_from_2018_02_28` | `23.6%` | `2.56%` | `10.40` | `-0.2%` |
| `baseline_spy` | `minus_1x_era_to_2018_02_27` | `69.6%` | `13.07%` | `1.06` | `-13.0%` |
| `baseline_spy` | `minus_0p5x_era_from_2018_02_28` | `206.8%` | `14.30%` | `0.80` | `-33.7%` |

## Gate Checks

| Check | Result |
|---|---|
| `mean_sharpe_at_least_1_0` | FAIL |
| `profitable_windows_at_least_70pct` | FAIL |
| `full_history_worst_dd_within_25pct` | FAIL |
| `beats_always_short_vol_dd_2018_02` | PASS |
| `beats_always_short_vol_dd_2020_03` | PASS |
| `clears_multiple_testing_hurdle` | PASS |
| `beats_always_short_vol_sharpe_full_history` | PASS |

Multiple-testing hurdle: with `12` pre-registered variants over `3197` days, best-of-N selection alone is expected to produce an annualized Sharpe of about `0.4674` under a zero-edge null.

## Overnight Gap Scenario

| Variant | Shock date | Total before | Total after | Worst DD after |
|---|---|---:|---:|---:|
| `slope_00` | `2018-02-05` | `294.9%` | `132.4%` | `-76.9%` |
| `slope_00_longvol` | `2018-02-05` | `18.6%` | `-30.2%` | `-82.2%` |
| `slope_00_vrp` | `2018-02-05` | `258.6%` | `111.0%` | `-71.9%` |
| `slope_00_vrp_longvol` | `2018-02-05` | `7.7%` | `-36.6%` | `-79.2%` |
| `slope_02` | `2018-02-05` | `391.6%` | `189.2%` | `-77.2%` |
| `slope_02_longvol` | `2018-02-05` | `47.7%` | `-13.1%` | `-82.8%` |
| `slope_02_vrp` | `2018-02-05` | `348.6%` | `164.0%` | `-73.1%` |
| `slope_02_vrp_longvol` | `2018-02-05` | `34.8%` | `-20.7%` | `-79.7%` |
| `slope_05` | `2018-02-05` | `122.5%` | `30.9%` | `-80.1%` |
| `slope_05_longvol` | `2018-02-05` | `-33.2%` | `-60.7%` | `-87.9%` |
| `slope_05_vrp` | `2018-02-05` | `83.4%` | `7.9%` | `-76.1%` |
| `slope_05_vrp_longvol` | `2018-02-05` | `-44.9%` | `-67.6%` | `-85.2%` |

## Look-Ahead Sensitivity (the decisive robustness test)

VIX cash indices settle at 16:15 ET while the ETP trades at the 16:00 close, so a same-close decision (lag 1) can use up to 15 minutes of future information. Feb-2018's damage happened inside that window, so lag 2 is used for every headline number.

| Variant | Sharpe (lag 2, headline) | Sharpe (lag 1, optimistic) | Volmageddon lag 2 | Volmageddon lag 1 |
|---|---:|---:|---:|---:|
| `slope_00` | `0.48` | `0.72` | `-39.2%` | `-10.6%` |
| `slope_00_longvol` | `0.29` | `0.74` | `-32.7%` | `32.0%` |
| `slope_00_vrp` | `0.46` | `0.65` | `-39.2%` | `-10.6%` |
| `slope_00_vrp_longvol` | `0.27` | `0.68` | `-32.7%` | `32.0%` |
| `slope_02` | `0.52` | `0.55` | `-39.2%` | `-10.6%` |
| `slope_02_longvol` | `0.32` | `0.60` | `-32.7%` | `32.0%` |
| `slope_02_vrp` | `0.51` | `0.49` | `-39.2%` | `-10.6%` |
| `slope_02_vrp_longvol` | `0.30` | `0.56` | `-32.7%` | `32.0%` |
| `slope_05` | `0.36` | `0.50` | `-39.2%` | `-10.6%` |
| `slope_05_longvol` | `0.18` | `0.57` | `-32.7%` | `32.0%` |
| `slope_05_vrp` | `0.32` | `0.44` | `-39.2%` | `-10.6%` |
| `slope_05_vrp_longvol` | `0.14` | `0.52` | `-32.7%` | `32.0%` |

## Signal Diagnostics

- contango fraction (slope > 0): `93.2%`
- VRP positive fraction: `85.3%`
- HAR-RV out-of-sample correlation with realized variance: `0.3407`
- HAR-RV beats an expanding-mean baseline on RMSE: `False`
- HAR-RV refits: `153`, first forecast `2013-11-06`

## Data Provenance

```json
{
  "cboe_indices": {
    "vix": "1990-01-02 -> 2026-07-27 (9237 rows)",
    "vix9d": "2011-01-04 -> 2026-07-27 (3912 rows)",
    "vix3m": "2009-09-18 -> 2026-07-27 (4238 rows)",
    "vix6m": "2008-01-02 -> 2026-07-27 (4670 rows)",
    "vvix": "2006-03-06 -> 2026-07-27 (5069 rows)"
  },
  "etp_coverage": {
    "VXX": "2018-01-25 -> 2026-07-27",
    "SVXY": "2011-10-04 -> 2026-07-27",
    "UVXY": "2011-10-04 -> 2026-07-27",
    "SVIX": "2022-03-30 -> 2026-07-27",
    "BIL": "2007-05-30 -> 2026-07-27",
    "SPY": "2005-01-03 -> 2026-07-27"
  },
  "futures_index_reconstruction": {
    "method": "index return = VXX return where available (+1x, no leverage assumption); otherwise -SVXY/leverage, else UVXY/leverage. SVXY leverage 1.0 before 2018-02-28, 0.5 after; UVXY 2.0 then 1.5.",
    "source_day_counts": {
      "vxx_direct": 2135,
      "uvxy_derived": 1587
    },
    "leverage_diagnostics": {
      "svxy_derived": {
        "overlap_days": 2135,
        "beta_vs_vxx": 1.0037,
        "correlation": 0.9148,
        "reads_as_expected": true
      },
      "uvxy_derived": {
        "overlap_days": 2135,
        "beta_vs_vxx": 0.9928,
        "correlation": 0.9804,
        "reads_as_expected": true
      }
    },
    "caveat": "Pre-2018 index returns are derived from a leveraged ETP, so they inherit its tracking error and fee drag. Daily-rebalanced leverage rescaling is exact for daily returns but not for multi-day compounding of the source product.",
    "tradeable_legs": {
      "short_vol": {
        "policy": "hold SVXY throughout (real closes); target exposure -1x before 2018-02-28, -0.5x after",
        "alternative_policy": "svix_when_available (SVIX from 2022-03-30, else SVXY) is reported as a robustness check",
        "svxy_days": 3722,
        "minus_1x_era_days": 1609,
        "minus_0p5x_era_days": 2113
      },
      "long_vol": {
        "policy": "VXX real closes where available (2018-01-25+); before that UVXY deleveraged by its 2x target",
        "caveat": "pre-2018 long-vol returns are synthetic (deleveraged UVXY) and inherit the crisis-timing problem",
        "vxx_days": 2135
      },
      "why_real_not_synthetic": "Synthetic leverage rescaling was tested and rejected: a -1x series rebuilt from VXX loses 48.7% over 2018-02-01..2018-02-12 while the real -1x product (SVXY) lost 90.7%. VIX futures moved violently between the 16:00 equity close and the 16:15 futures settlement, so ETPs striking NAV at different times are not interchangeable in exactly the crises that decide this sleeve. Backtests therefore use the real closes of the product
```

## SVIX Instrument-Policy Robustness

Period `2022-03-31 -> 2026-07-27`, variant `slope_02_vrp`.

| Instrument | Total | CAGR | Sharpe | Worst DD |
|---|---:|---:|---:|---:|
| `SVIX (-1x real)` | `76.1%` | `14.07%` | `0.53` | `-65.2%` |
| `SVXY (-0.5x real)` | `74.2%` | `13.78%` | `0.62` | `-38.8%` |
