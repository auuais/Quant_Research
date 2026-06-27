# V3-5 Live Paper Trading Strategy

## Objective

Run the `V3-5` stock sleeve in Alpaca paper trading for live forward validation, with broker-native protection and intraday monitoring.

This is a paper-only validation plan. `V3-5` remains a high-risk research branch because historical returns are dominated by frequent risk exits and a loose entry mapper.

## Symbols And Capital

- symbols: `TSLA`, `AVGO`, `NVDA`
- target paper sleeve: `$60,000`
- allocation: equal weight, one third per symbol
- sizing: whole shares only

Whole-share sizing is required so Alpaca can hold native `GTC` trailing-stop protection.

## Signal Policy

- daily model: existing `V3-5` directional signal
- entry gate:
  - daily `V3-5` prediction must be bullish
  - prediction strength must be `medium` or `high`
  - latest intraday news sentiment must be bullish
  - no same-day re-entry after an exit
- exit gate:
  - daily `V3-5` model no longer supports long, or
  - fresh intraday news sentiment turns bearish, or
  - risk stop is breached

Lack of fresh news blocks new entries but does not force an exit by itself.

## Risk Policy

- fixed stop loss: `5%`
- trailing stop: `1%`
- primary protection: Alpaca broker-native trailing stop
- strategy-side protection: minute risk checks during regular market hours
- stale protection check: every cycle verifies that the existing protective order still matches the expected trailing-stop setup

Alpaca trailing stops trigger only during regular market hours and become market orders when triggered. That means gap/slippage risk still exists.

## Cadence

- risk check interval: `60` seconds
- market/news refresh interval: `600` seconds
- maximum news staleness: `1800` seconds
- weekly report cadence: once per week, outside market hours

Reasoning:

- risk checks are lightweight and should run minute by minute
- DeepSeek news scoring is expensive enough that rescoring every minute is not useful
- ten-minute news checks are a practical balance for the current local model and three-symbol sleeve

## Operational Commands

Dry run one cycle:

```powershell
.\.venv\Scripts\python -m algoding.cli v35-intraday-run --max-cycles 1 --sleep-seconds 60 --news-check-seconds 600
```

Run a full regular-hours paper session:

```powershell
.\.venv\Scripts\python -m algoding.cli v35-intraday-run --submit --target-capital 60000 --max-cycles 390 --sleep-seconds 60 --news-check-seconds 600
```

Generate weekly report:

```powershell
.\.venv\Scripts\python -m algoding.cli v35-paper-report
```

## Tightening From The Prior Plan

- bypasses date-level historical news cache for intraday news checks
- avoids exiting just because no fresh news exists
- records news freshness metadata in each cycle output
- keeps entries stricter than exits: stale/no news blocks entry, but only fresh bearish news exits
- keeps broker-native trailing-stop maintenance in the minute loop
- exposes news cadence through CLI instead of hard-coding it

## Validation Criteria

Track weekly:

- realized and unrealized P/L
- number of entries and exits by symbol
- trailing-stop order presence and expiry status
- number of signal exits versus risk exits
- slippage from intended stop level where available
- whether paper results resemble the historical V3-5 replay

Pause the sleeve if:

- protective trailing stops are missing on any open position
- the runner misses more than one regular-hours cycle unexpectedly
- realized drawdown exceeds `5%` of sleeve capital
- Alpaca rejects protective orders repeatedly
- model/news scoring becomes stale for more than `30` minutes during an open position
