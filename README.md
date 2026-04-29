# Algoding

Systematic trading research and execution scaffold focused on:

- research and backtesting
- live market-data ingestion
- shadow execution
- paper trading

## Current scope

This repository is set up for a proof of concept using:

- Python `3.11+`
- QuestDB for runtime market-event storage
- PostgreSQL for metadata and audit records
- Alpaca as the first paper-trading broker target

## Repository layout

```text
config/                  Static strategy and risk configuration
infra/docker/            Local infrastructure definitions
src/algoding/            Application package
```

## Quick start

1. Copy `.env.example` to `.env`.
2. Start infrastructure:

```powershell
docker compose -f infra/docker/docker-compose.yml up -d
```

3. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

4. Install the project:

```powershell
pip install -e .
```

5. Run the scaffold checks:

```powershell
python -m algoding.cli healthcheck
python -m algoding.cli show-config
```

6. Seed and inspect the live-event path:

```powershell
python -m algoding.cli seed-sample --symbol SPY --price 501.25
python -m algoding.cli questdb-count
```

7. Run live bar ingestion or shadow mode after setting Alpaca credentials in `.env`:

```powershell
python -m algoding.cli ingest-bars --symbols SPY,QQQ --max-events 10
python -m algoding.cli shadow-run --symbols SPY --max-events 10
```

8. Build and optionally submit a paper-trading rebalance:

```powershell
python -m algoding.cli paper-account
python -m algoding.cli paper-rebalance --symbols SPY,QQQ,IWM,TLT,GLD,XLF,XLE,XLV,SMH
python -m algoding.cli paper-rebalance --symbols SPY,QQQ,IWM,TLT,GLD,XLF,XLE,XLV,SMH --submit
python -m algoding.cli basket-daily-cycle
python -m algoding.cli basket-daily-cycle --submit
python -m algoding.cli basket-report
```

9. Compare XLE strategy variants and optionally submit the current best one:

```powershell
python -m algoding.cli xle-compare
python -m algoding.cli xle-report
python -m algoding.cli xle-run-best
python -m algoding.cli xle-run-best --submit
python -m algoding.cli xle-daily-cycle
python -m algoding.cli xle-daily-cycle --submit
```

10. Register promoted research winners for broker paper trading and run them:

```powershell
python -m algoding.cli promote-candidate --market gold --symbol GLD --strategy-name gld_momentum_aggressive --notional 1000
python -m algoding.cli promote-candidate --market stocks --symbol META --strategy-name meta_mean_reversion_daily --notional 1000
python -m algoding.cli promoted-report
python -m algoding.cli promoted-run
python -m algoding.cli promoted-run --submit
```

11. Run cross-market historical research and inspect the leaderboard:

```powershell
python -m algoding.cli historical-research-run --markets etfs,commodities,stocks,fx --windows 6m,1y
python -m algoding.cli historical-research-report --limit 20
python -m algoding.cli historical-chart --market commodities --symbol SLV --strategy-name slv_momentum_daily --window 3y
python -m algoding.cli historical-chart-interactive --market commodities --symbol SLV --strategy-name slv_momentum_daily --window 3y
```

Historical research summaries are persisted in:

- `logs/strategy_runs.jsonl`
- PostgreSQL table: `strategy_runs`

## Hosted LLM models

The news sentiment engine supports:

- local Hugging Face models via `LLM_NEWS_MODEL_NAME`
- hosted OpenAI models when `LLM_NEWS_MODEL_NAME` is set to an OpenAI model name such as `gpt-5`
- optional GPU-quantized local runs via `LLM_NEWS_QUANTIZATION=4bit|8bit`
- hard GPU-only enforcement via `LLM_NEWS_FORCE_GPU=true`

Required environment for hosted OpenAI scoring:

```env
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_NEWS_MODEL_NAME=gpt-5
```

Example local GPU-only setup for a larger open model:

```env
LLM_NEWS_MODEL_NAME=U:\models\Qwen3-4B
LLM_NEWS_CACHE_DIR=cache/llm_news_qwen3_4b_gpu
LLM_NEWS_QUANTIZATION=4bit
LLM_NEWS_FORCE_GPU=true
```

## Next implementation targets

- historical data downloader
- richer strategy signal generation
- replay mode from QuestDB
- portfolio and risk engine
