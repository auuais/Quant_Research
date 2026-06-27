# V3-5 price-aware execution controls

- window (test): `2025-03-14 -> 2026-04-10 (270 days)`
- symbols: `AVGO, NVDA, TSLA, GOOGL, XOM`
- fixed stop: `5%` · random seeds: `20`

## Cost model (Alpaca-aligned)

- commission `$0.00` (commission-free), data feed `iex`
- slippage `1.0`bps spread + `1.0`bps impact (+`2.0`bps on stops)
- FINRA TAF `$0.000195`/sh cap `$9.79`, SEC 31 `$27.8`/$1M sold
- participation cap `0.0500%` of $-volume, volume scaled `x30` to undo IEX under-reporting

## Strategy comparison (mean over symbols)

| Strategy | Gross return | **Net return** | Net max DD | Trades | Partial fills |
|---|---:|---:|---:|---:|---:|
| `buy_and_hold` | `64.09%` | `64.06%` | `-23.48%` | `0.0` | `0` |
| `always_long_1pct` | `600.63%` | `-9.48%` | `-21.20%` | `134.2` | `0` |
| `random_entry_1pct` | `366.09%` | `-1.58%` | `-15.48%` | `98.75` | `0` |
| `v35_model_trail_1pct` | `311.92%` | `-4.03%` | `-15.42%` | `94.6` | `0` |
| `v35_model_trail_2pct` | `76.88%` | `-17.36%` | `-23.95%` | `87.0` | `0` |
| `v35_model_trail_3pct` | `36.03%` | `-13.08%` | `-24.68%` | `76.0` | `0` |
| `v35_model_trail_5pct` | `20.77%` | `1.49%` | `-25.78%` | `59.4` | `0` |
| `v35_model_trail_none` | `38.02%` | `33.59%` | `-19.82%` | `49.2` | `0` |
| `v35_model_exit_only` | `45.98%` | `43.88%` | `-19.78%` | `45.2` | `0` |

## Cheaper execution: stop-market vs MOC/close (levers #2 + #3)

MOC profile: `0.5`bps spread + `0.5`bps impact, `0.0`bps stop-extra (close-to-close decisions, fills at the close auction, no intraday stop-gap).

| Model strategy | Stop-market net | Stop-mkt trades | **MOC/close net** | MOC trades |
|---|---:|---:|---:|---:|
| `v35_model_trail_1pct` | `-4.03%` | `94.6` | `27.26%` | `64.4` |
| `v35_model_trail_2pct` | `-17.36%` | `87.0` | `27.30%` | `57.2` |
| `v35_model_trail_3pct` | `-13.08%` | `76.0` | `34.30%` | `53.4` |
| `v35_model_trail_5pct` | `1.49%` | `59.4` | `45.06%` | `50.0` |
| `v35_model_trail_none` | `33.59%` | `49.2` | `44.51%` | `47.8` |
| `v35_model_exit_only` | `43.88%` | `45.2` | `44.83%` | `45.2` |

## Entry edge (forward 5-day return on bullish entry days vs unconditional)

- mean edge (entry − unconditional): `0.05%`
- symbols with positive entry edge: `3/5`
- per-symbol edge: `{'AVGO': '-0.15%', 'NVDA': '0.30%', 'TSLA': '0.13%', 'GOOGL': '0.18%', 'XOM': '-0.19%'}`

## Verdicts

- V3-5 (1% trailing) gross vs net: `311.92%` -> `-4.03%`
- model entries beat **random** entries (same rate), net: `False`
- model entries beat **buy & hold**, net: `False`
- model entries beat **always-long** (stop only), net: `True`
- gross-best trailing width: `v35_model_trail_1pct` · net-best: `v35_model_trail_none`
- entry edge positive: `True`
- no-same-day-reentry is a no-op on daily bars: `True`

