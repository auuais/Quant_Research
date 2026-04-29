# Algorithmic Trading Strategy

Last updated: 2026-03-27

## 1. Executive view

The right version of this project is not "find one bot and go live fast." It is:

1. Build a reliable research and execution platform.
2. Start with low-frequency, liquid instruments and a research-first workflow.
3. Separate internal strategy evaluation from broker-facing paper execution.
4. Use paper trading first, then tiny live size only after a full research conclusion.
5. Optimize for survival, process quality, and controlled drawdown.

That is the realistic path for a solo or small-team retail algorithmic trading project.

## 2. Operating model

Our operating model is now:

1. Keep real broker paper trading for the chosen live candidate strategy.
2. Keep parallel strategy evaluation in an internal paper ledger.
3. Treat research as a formal program with tracked goals, experiments, and performance logs.
4. Do not move to real-money trading until the research program is concluded and reviewed.

### Why internal strategy books matter

For example:

- Strategy A buys $500 of XLE
- Strategy B buys $300 of XLE
- Strategy C sells $200 of XLE later

Alpaca may show only one net XLE position, but our internal system must still say:

- which strategy owns what notional exposure
- which strategy produced what P&L
- which strategy is currently active, inactive, or promoted to broker execution

So broker paper trading and internal evaluation are not the same thing.

## 3. What the internet evidence suggests

The current evidence does not support aggressive retail expectations.

- S&P Dow Jones' SPIVA U.S. Year-End 2024 scorecard says 65% of active large-cap U.S. equity funds underperformed the S&P 500 in 2024, and over the 15-year period ending December 2024 there were no categories in which a majority of active managers outperformed. Source: https://www.spglobal.com/spdji/en/documents/spiva/spiva-us-year-end-2024.pdf
- S&P Dow Jones' U.S. Persistence Scorecard Year-End 2024 says persistent outperformance is hard to find, and among top-quartile active domestic equity funds as of December 2020, not a single fund remained in the top quartile over the next four years. Source: https://www.spglobal.com/spdji/en/spiva/article/us-persistence-scorecard
- Investor.gov says day trading is extremely risky and can result in substantial financial losses in a very short period of time. Source: https://www.investor.gov/additional-resources/general-resources/glossary/day-trading
- FINRA's current public day-trading page still states that pattern day traders must maintain at least $25,000 in margin-account equity. Source: https://www.finra.org/investors/investing/investment-products/stocks/day-trading
- FINRA announced on January 7, 2026 that it filed a proposed rule change that would eliminate the current pattern day trader framework and the $25,000 minimum, but that is a proposal, not a final effective rule. Source: https://www.finra.org/compliance-tools/weekly-archive/01072026
- OANDA's developer portal currently warns that CFDs are complex instruments and says 76.6% of retail investor accounts lose money trading CFDs with that provider. Source: https://developer.oanda.com/
- The CFTC's forex advisory says forex trading is risky and you should thoroughly investigate claims that downplay those risks. Source: https://www.cftc.gov/LearnAndProtect/AdvisoriesAndArticles/CustomerAdvisory_MustKnowForex.html

Conclusion: a credible retail trading business plan should assume modest returns, long testing cycles, and a serious probability that many candidate strategies will fail.

## 4. Recommended market and broker strategy

### Start here

Use U.S. equities and ETFs first, with daily or multi-hour holding periods.

Why:

- Lowest implementation complexity relative to intraday or multi-asset execution.
- Avoids immediate dependence on ultra-low latency.
- Easier to backtest with cleaner data.
- Easier to control slippage and operational risk.
- Easier to stay below the practical complexity of futures, forex, or CFDs early on.

### Broker recommendation

Use a two-broker path:

1. Alpaca for fast paper-trading iteration and API-first development.
2. Interactive Brokers for broader live-trading capability once the system is stable.

Evidence:

- Alpaca's docs state that paper trading is free, uses real-time simulation, and is available globally as a paper-only account. Sources:
  - https://docs.alpaca.markets/
  - https://docs.alpaca.markets/docs/trading/paper-trading/
- Interactive Brokers states that its APIs cover stocks, options, futures, currencies, bonds, and more across 170 markets in 40 countries, and IBKR paper accounts support the Web API and TWS API with some simulator limitations. Sources:
  - https://www.interactivebrokers.com/en/trading/ib-api.php
  - https://www.interactivebrokers.com/campus/glossary-terms/paper-trading-account/

### Avoid at the beginning

- High-frequency intraday equities
- Leveraged forex
- CFDs
- Options-selling income systems
- Any strategy that needs perfect fills to look profitable in backtests

## 5. What we need to start

The first milestone is not "profit." It is a working research-to-execution proof of concept.

We need these seven things:

1. A local development environment
   - Python 3.11+
   - Git
   - Docker Desktop
   - PostgreSQL
2. A market-data path
   - Historical data for research
   - Live or delayed live stream for shadow and paper execution
3. A time-series storage layer
   - QuestDB is the best fit for this phase
   - InfluxDB is acceptable if you already know it
4. A research engine
   - Backtesting, analytics, parameter sweeps, walk-forward tests
5. A shadow execution engine
   - Live signals on live data, but orders go to a null sink
6. A paper-trading engine
   - Same signal path, same risk checks, real broker API session, paper account destination
7. Monitoring and controls
   - logs, alerts, heartbeats, reconciliation, and kill switches

The practical build order should be:

1. Historical research
2. Live-data ingestion
3. Shadow execution
4. Paper trading
5. Tiny live deployment

## 6. Recommended technical stack

### Phase 1 stack

- Language: Python
- Research: pandas, polars, numpy, scipy, statsmodels, vectorbt or backtrader
- Data storage:
  - QuestDB for tick/bar/event time-series ingestion during shadow and paper phases
  - PostgreSQL for metadata, strategy config, orders, fills, and audit logs
  - Parquet for offline research snapshots, reproducible datasets, and archive exports
- Execution service: Python async service with strict risk checks
- Shadow service: same signal and risk pipeline as live, but broker orders route to a null sink with full timestamp logging
- Scheduling: cron / Windows Task Scheduler initially
- Monitoring: logs + Telegram/Slack/email alerts

### Why QuestDB first

Your point is correct: Parquet is excellent for offline research, but it hides ingestion behavior that matters in production.

QuestDB is a better fit for the proof-of-concept phase because it lets us:

- ingest streaming ticks or bars with timestamps close to real operating conditions
- measure data lag, event arrival gaps, and heartbeat failures
- replay recent history into shadow or paper environments
- keep SQL-style analytics simple

Parquet should stay in the stack, but as a research and archive format, not as the primary paper-trading runtime store.

### Why Python first

- Fastest path to research, backtesting, broker APIs, and ops.
- Alpaca and IBKR both support Python workflows.
- Large ecosystem for statistics, optimization, and data engineering.

### When to add another language

Add Rust or Go only if one of these becomes true:

- You need lower-latency streaming and order routing.
- Python process stability becomes a bottleneck.
- You are running several independent live services and want stronger deployment ergonomics.

For a first profitable or first reliable system, Python is usually the correct choice.

## 7. Research-first strategy roadmap

The best initial production strategy is not prediction-heavy machine learning. It is a simple, testable, diversified systematic process.

### Production track

Start with:

- rule-based daily/weekly ETF basket
- low turnover
- fully tracked paper execution
- strict broker sync and performance logging

### Research track 1

Parallel internal evaluation of multiple strategies on the same instrument or basket, including:

- XLE parallel strategy books
- basket momentum variants
- basket mean-reversion variants
- aggressive and conservative versions of the same signal family

### Research track 2

After the rule-based research track is stable:

- ML/DL prototype on the same ETF basket
- compare to classical rule-based baselines
- no promotion to live candidate without out-of-sample validation

### Research track 3

After ML/DL baseline work:

- LLM sentiment overlay
- use sentiment as an auxiliary signal, not as the first production engine

### Recommended first production strategy

Medium-term trend and momentum on liquid U.S. ETFs.

Universe example:

- SPY
- QQQ
- IWM
- TLT
- GLD
- XLF
- XLE
- XLV
- SMH

Core logic:

- Trade only highly liquid instruments.
- Use daily bars at first.
- Rank instruments by 3- to 12-month momentum.
- Hold the strongest 2 to 4 names or go to cash/T-bill proxy when market regime is weak.
- Use volatility targeting and position caps.
- Rebalance weekly, not continuously.

Why this is a good first system:

- Much less microstructure sensitivity than intraday strategies.
- Backtests are easier to audit.
- Easier to execute consistently at retail size.
- Lower operational complexity.
- Less exposed to pattern day trading constraints.

### Secondary strategy after that

Mean reversion only after the trend system is stable, and only on very liquid ETFs or large-cap equities with conservative assumptions about slippage and gaps.

## 8. Market expansion roadmap

After the current ETF basket research is mature, expand in this order:

### Phase A

- Gold
  - first via liquid ETF exposure

### Phase B

- Large-cap stocks
  - highly liquid names only
  - focus on robust basket-style research before single-name complexity explodes

### Phase C

- Major FX pairs
  - research first
  - only after risk, session handling, and leverage assumptions are fully documented

### Later phases

- ML/DL on the same basket
- LLM sentiment overlay
- only after that consider more complex markets such as crypto, futures, and options

## 9. System design

Build the platform as seven modules:

1. Data ingestion
   - Broker market data plus separate historical dataset where needed
   - Persist streaming events into QuestDB
2. Research and backtesting
   - Event-driven backtest with fees, slippage, delistings if using single stocks
3. Signal generation
   - Deterministic signals with versioned parameters
4. Portfolio construction
   - Risk limits, volatility targeting, exposure caps
5. Shadow execution
   - Run the live signal stack against live data
   - Emit synthetic orders to a null sink
   - Measure latency, dropped events, stale-data behavior, and decision timing
6. Execution
   - Broker adapter layer for Alpaca and IBKR
7. Monitoring and controls
   - Order audit log, PnL checks, kill switch, stale-data checks

Required controls:

- Max risk per position
- Max daily loss
- Max gross exposure
- Max turnover
- Trading halt on stale data or repeated API errors
- Broker/account reconciliation at least daily
- Heartbeat alarms for data-feed and broker-session disconnects

### Why shadow execution is required

Paper trading is necessary, but it is not enough.

Shadow execution is the stage that proves:

- your live market-data session remains healthy
- your code handles real message timing and real network conditions
- your risk engine behaves correctly under live clock conditions
- your strategy does not depend on simulator-only assumptions

The correct progression is:

1. Backtest
2. Shadow
3. Paper
4. Tiny live

## 10. Research governance

Before real-money trading, all research tracks must produce:

- hypothesis statement
- dataset description
- backtest assumptions
- strategy definition
- performance metrics
- drawdown analysis
- failure cases
- promotion decision: reject, keep in research, paper candidate, or live candidate

Research work must be tracked in `TRACKER.md`, and performance logs must be stored in the system of record.

## 11. Performance logging policy

We are not only tracking goals. We are also logging performance.

Required logs:

- internal strategy-book performance by strategy
- broker paper orders
- broker fills
- broker positions
- account snapshots
- comparison reports for candidate strategies

These logs must support:

- per-strategy attribution
- portfolio attribution
- drawdown review
- promotion/rejection decisions
- eventual publication of research summaries

## 12. Publication goal

This project may produce publishable research later, but publication happens only after:

- methodology is stable
- data assumptions are documented
- performance claims are properly caveated
- hypothetical and paper results are clearly labeled

The purpose of publication is:

- document research credibility
- attract technically serious users or partners later
- show methodology, not marketing hype

## 13. Realistic goals

These are realistic goals inferred from the sources above and from how systematic retail strategies usually fail in practice.

They are not guarantees and should be treated as operating targets.

### First 90 days

- Goal: no live trading
- Output: clean data pipeline, QuestDB ingestion, baseline backtester, shadow engine, broker integration
- Success metric: no silent failures, full event logs, backtest assumptions documented, stable live-data heartbeat

### Months 4 to 6

- Goal: shadow-trade first, then paper-trade one strategy continuously
- Output: at least 2 to 4 weeks of uninterrupted shadow execution plus 8 to 12 weeks of uninterrupted paper execution
- Success metric: shadow timing is stable, paper and expected fills are directionally consistent, risk rules behave correctly

### Months 7 to 12

- Goal: tiny live deployment
- Capital: only risk capital you can afford to lose
- Success metric: operational stability matters more than return

Reasonable first-year live targets:

- Net return: 4% to 10%
- Max drawdown: keep under 8% to 12%
- Monthly turnover: moderate, not hyperactive
- Sharpe target: 0.7 to 1.2 is respectable for a first retail systematic strategy after costs

If you are targeting 25% to 50% annual returns from the start with controlled drawdowns, that is not a realistic base plan.

### Years 2 to 3

Only scale if:

- Live execution matches research expectations closely enough
- Slippage is stable
- The strategy survives different market regimes
- You have at least one additional uncorrelated strategy

Reasonable medium-term targets after real validation:

- Net return: 8% to 15%
- Max drawdown: 10% to 15%
- Focus: capital preservation first, then scaling

Anything materially above that may happen in a good year, but it should not be your base-case business plan.

## 14. Risk framework

Use these hard rules:

- Risk no more than 25 to 50 basis points of account equity per position at entry in the first live version.
- Cap portfolio gross exposure until the live system proves stable.
- Do not average down automatically.
- Do not trade illiquid small caps.
- Do not let one strategy control the whole account.
- Stop live trading after any unexplained execution bug.

Operational risks matter as much as market risk:

- API disconnects
- wrong-symbol orders
- stale prices
- split/dividend handling
- timezone/session bugs
- duplicate orders after retries
- hidden lag between signal calculation and order submission

## 15. Business strategy for the project

If this project is meant to become a serious product or company, use a staged model:

### Stage 1

Personal trading research platform.

### Stage 2

Internal production trading stack with paper and micro live accounts.

### Stage 3

Performance reporting, analytics dashboards, and broker abstraction.

### Stage 4

Only after legal/compliance review:

- managed accounts
- signal platform
- portfolio analytics SaaS
- educational product

Do not start by marketing returns. Start by building infrastructure, controls, and honest reporting.

## 16. 12-month implementation roadmap

### Month 1

- Define target market: U.S. equities and ETFs
- Open Alpaca paper account
- Open IBKR account or paper environment
- Build PostgreSQL and QuestDB locally
- Build data schema and market-data ingestion

### Month 2

- Build event-driven backtester
- Build live-data ingestion into QuestDB
- Model commissions, spread, and slippage
- Add benchmark comparison against SPY and 3-month T-bill proxy

### Month 3

- Build first momentum/trend strategy
- Add portfolio construction and volatility targeting
- Run walk-forward tests
- Build the shadow execution path with null-order sink

### Month 4

- Run shadow continuously against live data
- Connect paper execution to Alpaca
- Add order state machine and reconciliation
- Add alerts and kill switch

### Month 5

- Compare shadow timing and signal logs against expectations
- Paper trade continuously
- Compare backtest assumptions to paper behavior
- Build internal parallel strategy books for XLE and basket variants
- Tighten risk controls

### Month 6

- Review paper-trading logs
- Reject or revise the strategy if live-like behavior diverges materially
- Formalize `research candidate` vs `paper candidate` vs `live candidate`

### Months 7 to 9

- Expand research to gold and large-cap stocks
- Build comparative reports for rule-based strategies
- Start ML/DL prototype research on the same basket only after rule-based baselines are stable

### Months 10 to 12

- Expand research to major FX pairs
- Build LLM sentiment overlay research after ML baseline work exists
- Decide which strategy families are worthy of later production promotion
- Do not move to real-money trading before a formal research review is complete

## 17. Practical recommendation

If the goal is to build something that actually has a chance of surviving:

- Use Python first.
- Use QuestDB as the runtime time-series store for the proof-of-concept phase.
- Start with Alpaca paper trading.
- Insert a shadow-execution stage before relying on paper results.
- Plan to migrate serious live trading to Interactive Brokers.
- Trade liquid U.S. ETFs on daily or weekly rebalancing.
- Keep broker paper trading for the active candidate only.
- Keep parallel strategy research in internal ledgers.
- Set modest expectations.
- Treat strategy research and execution reliability as equally important.

## 18. Immediate build checklist

If we were starting today, the concrete setup list would be:

- Python project with dependency management and environment file support
- Docker Compose for QuestDB and PostgreSQL
- `src/` layout with `data`, `research`, `signals`, `portfolio`, `execution`, `shadow`, and `monitoring` packages
- broker adapters for Alpaca first, IBKR second
- historical data downloader
- live market-data ingestor into QuestDB
- null-order sink for shadow mode
- paper-order router
- internal virtual ledgers for strategy-level attribution
- strategy config and risk config files
- structured logging and alert hooks
- research tracker and performance log registry

That is the minimum setup needed to do research, implementation, and a believable proof of concept.

## 19. My bottom-line recommendation

The strongest realistic version of this project is:

"Build a Python-based systematic trading platform for liquid U.S. ETFs, evaluate multiple strategies internally with clean attribution, keep only the chosen candidate in broker paper trading, expand research later to gold, major FX pairs, and large-cap stocks, and only then consider ML/DL and LLM overlays before any real-money deployment."

That is not the most exciting story, but it is the one most consistent with the current evidence and with realistic retail constraints.

## 20. Source links

- SPIVA U.S. Year-End 2024: https://www.spglobal.com/spdji/en/documents/spiva/spiva-us-year-end-2024.pdf
- SPIVA U.S. Persistence Scorecard Year-End 2024: https://www.spglobal.com/spdji/en/spiva/article/us-persistence-scorecard
- Investor.gov day-trading glossary: https://www.investor.gov/additional-resources/general-resources/glossary/day-trading
- FINRA day-trading page: https://www.finra.org/investors/investing/investment-products/stocks/day-trading
- FINRA January 7, 2026 update on proposed PDT overhaul: https://www.finra.org/compliance-tools/weekly-archive/01072026
- Alpaca docs home: https://docs.alpaca.markets/
- Alpaca paper-trading docs: https://docs.alpaca.markets/docs/trading/paper-trading/
- Interactive Brokers API overview: https://www.interactivebrokers.com/en/trading/ib-api.php
- IBKR paper-trading account notes: https://www.interactivebrokers.com/campus/glossary-terms/paper-trading-account/
- OANDA developer portal and CFD risk disclosure: https://developer.oanda.com/
- CFTC forex advisory: https://www.cftc.gov/LearnAndProtect/AdvisoriesAndArticles/CustomerAdvisory_MustKnowForex.html
- SEC market structure and algorithmic trading report page: https://www.sec.gov/file/market-structure-and-algorithmic-trading
- SEC staff report on algorithmic trading in U.S. capital markets: https://www.sec.gov/files/marketstructure/research/algo_trading_report_2020.pdf
