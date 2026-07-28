from __future__ import annotations

import argparse
import json

from algoding.app import build_app_context
from algoding.data.alpaca import AlpacaBarIngestor
from algoding.data.models import MarketEvent
from algoding.execution.deepseek_news_research import DeepseekNewsResearchLab
from algoding.execution.deepseek_directional_crypto_research import DeepseekDirectionalCryptoResearchLab
from algoding.execution.deepseek_directional_research import DeepseekDirectionalResearchLab
from algoding.execution.deepseek_directional_v3_hourly import DeepseekDirectionalV3HourlyLab
from algoding.execution.deepseek_directional_v3_variants import run_v3_top5_trailing_no_same_day_reentry
from algoding.execution.deepseek_directional_v4_research import DeepseekDirectionalV4ResearchLab
from algoding.execution.deepseek_strict_compare import DeepseekStrictComparisonLab
from algoding.execution.qwen_embedding_directional_research import QwenEmbeddingDirectionalResearchLab
from algoding.execution.v35_execution_controls import run_v35_execution_controls
from algoding.execution.long_short_research import LongShortResearchLab
from algoding.execution.cross_sectional_retrain import CrossSectionalRetrainLab
from algoding.execution.long_short_corrected import LongShortCorrectedLab
from algoding.execution.hybrid_alpha_research import HybridAlphaResearchLab
from algoding.execution.factor_ml_research import FactorMlResearchLab
from algoding.execution.factor_validation_research import FactorValidationResearchLab
from algoding.execution.factor_news_overlay_research import FactorNewsOverlayResearchLab
from algoding.execution.sentiment_balanced_research import SentimentBalancedResearchLab
from algoding.execution.embedding_sentiment_research import EmbeddingSentimentResearchLab
from algoding.execution.portfolio_controls_research import PortfolioControlsResearchLab
from algoding.execution.price_momentum_validation_research import PriceMomentumValidationResearchLab
from algoding.execution.cross_asset_momentum_research import CrossAssetMomentumResearchLab
from algoding.execution.regime_switch_research import RegimeSwitchResearchLab
from algoding.execution.vol_carry_research import VolCarryResearchLab
from algoding.execution.event_premia_research import EventPremiaResearchLab
from algoding.execution.alpha_factory_research import AlphaFactoryResearchLab
from algoding.execution.book_size_sensitivity_research import BookSizeSensitivityResearchLab
from algoding.execution.dl_research import DlResearchLab
from algoding.execution.historical_research import HistoricalResearchLab
from algoding.execution.llm_research import LlmNewsResearchLab
from algoding.execution.ml_research import MlResearchLab
from algoding.execution.parallel_research import ParallelResearchLedger
from algoding.execution.parallel_xle import ParallelXleLedger
from algoding.execution.promotion import PromotedPaperTrader
from algoding.execution.sync import PaperBrokerSync
from algoding.execution.v35_intraday import V35IntradayTrader
from algoding.execution.v35_paper import V35PaperTrader
from algoding.execution.paper_workflow import MomentumPaperTrader
from algoding.execution.xle_lab import XleStrategyLab
from algoding.portfolio.risk import RiskLimits
from algoding.reporting.historical_charts import HistoricalChartBuilder
from algoding.reporting.deepseek_strict_compare_pdf import build_deepseek_strict_compare_pdf
from algoding.reporting.llm_overlay_pdf import build_llm_overlay_comparison_pdf
from algoding.reporting.research_comparison_pdf import build_research_comparison_pdf
from algoding.research.cost_aware_backtest import ExecutionAssumptions, run_execution_aware_replay
from algoding.shadow.runner import ShadowBarStrategy
from algoding.portfolio.risk import RiskEngine


def cmd_show_config() -> int:
    context = build_app_context()
    print(json.dumps(context.settings.to_dict(), indent=2))
    return 0


def cmd_healthcheck() -> int:
    context = build_app_context()
    report = {
        "environment": context.settings.app_env,
        "postgres_dsn": context.settings.postgres_dsn,
        "questdb_http_url": context.settings.questdb_http_url,
        "alpaca_base_url": context.settings.alpaca_paper_base_url,
        "shadow_sink_path": str(context.settings.shadow_sink_path),
    }
    print(json.dumps(report, indent=2))
    return 0


def cmd_seed_sample(symbol: str, price: float) -> int:
    context = build_app_context()
    event = MarketEvent.sample(symbol=symbol, price=price)
    context.questdb.ingest_market_event(event)
    print(json.dumps({"status": "ok", "event": event.to_serializable_dict()}, indent=2))
    return 0


def cmd_questdb_count() -> int:
    context = build_app_context()
    count = context.questdb.count_market_events()
    print(json.dumps({"market_events": count}, indent=2))
    return 0


def cmd_ingest_bars(symbols: list[str], max_events: int | None) -> int:
    context = build_app_context()
    ingestor = AlpacaBarIngestor(context.settings, context.questdb)
    ingestor.run(symbols=symbols, max_events=max_events)
    print(json.dumps({"status": "completed", "mode": "ingest", "symbols": symbols}, indent=2))
    return 0


def cmd_shadow_run(symbols: list[str], max_events: int | None) -> int:
    context = build_app_context()
    risk_engine = RiskEngine()
    strategy = ShadowBarStrategy(
        threshold_bps=context.settings.shadow_price_move_threshold_bps,
        order_quantity=context.settings.shadow_order_quantity,
        sink=context.shadow_sink,
        risk_engine=risk_engine,
    )
    ingestor = AlpacaBarIngestor(
        context.settings,
        context.questdb,
        subscribers=[strategy.on_market_event],
    )
    ingestor.run(symbols=symbols, max_events=max_events)
    print(
        json.dumps(
            {
                "status": "completed",
                "mode": "shadow",
                "symbols": symbols,
                "orders_emitted": strategy.orders_emitted,
            },
            indent=2,
        )
    )
    return 0


def cmd_paper_account() -> int:
    context = build_app_context()
    trader = MomentumPaperTrader(context.settings)
    print(json.dumps(trader.get_account_summary(), indent=2))
    return 0


def cmd_paper_rebalance(symbols: list[str], submit: bool) -> int:
    context = build_app_context()
    trader = MomentumPaperTrader(context.settings)
    plan = trader.build_plan(symbols)
    results = trader.execute_plan(plan, submit=submit)
    print(
        json.dumps(
            {
                "mode": "paper",
                "submitted": submit,
                "targets": plan.targets,
                "current_positions": plan.current_positions,
                "rankings": plan.rankings,
                "orders": results,
            },
            indent=2,
        )
    )
    return 0


def cmd_basket_daily_cycle(symbols: list[str], submit: bool) -> int:
    context = build_app_context()
    trader = MomentumPaperTrader(context.settings)
    result = trader.run_daily_cycle(symbols, submit=submit)
    print(json.dumps(result, indent=2))
    return 0


def cmd_basket_report(limit: int) -> int:
    context = build_app_context()
    trader = MomentumPaperTrader(context.settings)
    print(
        json.dumps(
            {
                "account": trader.get_account_summary(),
                "snapshots": trader.list_account_snapshots(limit),
            },
            indent=2,
        )
    )
    return 0


def cmd_xle_compare(symbol: str, bars: int) -> int:
    context = build_app_context()
    lab = XleStrategyLab(context.settings)
    evaluations = lab.compare(symbol=symbol, bars=bars)
    print(json.dumps({"symbol": symbol, "bars": bars, "evaluations": evaluations}, indent=2))
    return 0


def cmd_xle_report(symbol: str, limit: int) -> int:
    context = build_app_context()
    lab = XleStrategyLab(context.settings)
    report = lab.latest_report(symbol=symbol, limit=limit)
    print(json.dumps({"symbol": symbol, "latest_runs": report}, indent=2))
    return 0


def cmd_xle_run_best(symbol: str, bars: int, submit: bool) -> int:
    context = build_app_context()
    lab = XleStrategyLab(context.settings)
    run = lab.build_best_strategy_run(symbol=symbol, bars=bars)
    results = lab.execute_best_strategy(run, submit=submit)
    print(
        json.dumps(
            {
                "symbol": symbol,
                "submitted": submit,
                "best_strategy": run.best,
                "current_position_qty": run.current_position_qty,
                "orders": results,
                "evaluations": run.evaluations,
            },
            indent=2,
        )
    )
    return 0


def cmd_xle_parallel_run(symbol: str, bars: int) -> int:
    context = build_app_context()
    ledger = ParallelXleLedger(context.settings)
    result = ledger.run(symbol=symbol, bars=bars)
    print(json.dumps(result, indent=2))
    return 0


def cmd_xle_parallel_report(symbol: str) -> int:
    context = build_app_context()
    ledger = ParallelXleLedger(context.settings)
    result = ledger.report(symbol=symbol)
    print(json.dumps(result, indent=2))
    return 0


def cmd_xle_parallel_history(symbol: str, strategy_name: str | None, limit: int) -> int:
    context = build_app_context()
    ledger = ParallelXleLedger(context.settings)
    result = ledger.history(symbol=symbol, strategy_name=strategy_name, limit=limit)
    print(json.dumps(result, indent=2))
    return 0


def cmd_research_parallel_run(market: str, bars: int) -> int:
    context = build_app_context()
    ledger = ParallelResearchLedger(context.settings)
    result = ledger.run_market(market=market, bars=bars)
    print(json.dumps(result, indent=2))
    return 0


def cmd_research_parallel_report(market: str) -> int:
    context = build_app_context()
    ledger = ParallelResearchLedger(context.settings)
    result = ledger.report_market(market=market)
    print(json.dumps(result, indent=2))
    return 0


def cmd_research_parallel_history(symbol: str, strategy_name: str | None, limit: int) -> int:
    context = build_app_context()
    ledger = ParallelResearchLedger(context.settings)
    result = ledger.history(symbol=symbol, strategy_name=strategy_name, limit=limit)
    print(json.dumps(result, indent=2))
    return 0


def cmd_promote_candidate(market: str, symbol: str, strategy_name: str, notional: float) -> int:
    context = build_app_context()
    trader = PromotedPaperTrader(context.settings)
    result = trader.promote(market=market, symbol=symbol, strategy_name=strategy_name, notional=notional)
    print(json.dumps(result, indent=2))
    return 0


def cmd_promoted_report() -> int:
    context = build_app_context()
    trader = PromotedPaperTrader(context.settings)
    print(json.dumps({"promoted_candidates": trader.list_promoted()}, indent=2))
    return 0


def cmd_promoted_run(submit: bool) -> int:
    context = build_app_context()
    trader = PromotedPaperTrader(context.settings)
    result = trader.run_promoted(submit=submit)
    print(json.dumps(result, indent=2))
    return 0


def cmd_historical_research_run(
    markets: list[str],
    windows: list[str],
    timeframe: str,
    regular_hours_only: bool,
    execution_aware: bool,
    no_same_day_reentry: bool,
    stop_loss_pct: float | None,
    trailing_stop_pct: float | None,
) -> int:
    context = build_app_context()
    lab = HistoricalResearchLab(context.settings)
    result = lab.run(
        markets=markets,
        windows=windows,
        timeframe=timeframe,
        regular_hours_only=regular_hours_only,
        execution_aware=execution_aware,
        no_same_day_reentry=no_same_day_reentry,
        stop_loss_pct=stop_loss_pct,
        trailing_stop_pct=trailing_stop_pct,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_historical_research_report(limit: int, execution_aware: bool) -> int:
    context = build_app_context()
    lab = HistoricalResearchLab(context.settings)
    result = lab.report(limit=limit, execution_aware=execution_aware)
    print(json.dumps(result, indent=2))
    return 0


def cmd_ml_research_run(
    markets: list[str],
    windows: list[str],
    models: list[str],
    timeframe: str,
    regular_hours_only: bool,
    no_same_day_reentry: bool,
    stop_loss_pct: float | None,
    trailing_stop_pct: float | None,
) -> int:
    context = build_app_context()
    lab = MlResearchLab(context.settings)
    result = lab.run(
        markets=markets,
        windows=windows,
        models=models,
        timeframe=timeframe,
        regular_hours_only=regular_hours_only,
        no_same_day_reentry=no_same_day_reentry,
        stop_loss_pct=stop_loss_pct,
        trailing_stop_pct=trailing_stop_pct,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_ml_research_report(limit: int) -> int:
    context = build_app_context()
    lab = MlResearchLab(context.settings)
    result = lab.report(limit=limit)
    print(json.dumps(result, indent=2))
    return 0


def cmd_dl_research_run(
    markets: list[str],
    windows: list[str],
    models: list[str],
    timeframe: str,
    regular_hours_only: bool,
    no_same_day_reentry: bool,
    stop_loss_pct: float | None,
    trailing_stop_pct: float | None,
) -> int:
    context = build_app_context()
    lab = DlResearchLab(context.settings)
    result = lab.run(
        markets=markets,
        windows=windows,
        models=models,
        timeframe=timeframe,
        regular_hours_only=regular_hours_only,
        no_same_day_reentry=no_same_day_reentry,
        stop_loss_pct=stop_loss_pct,
        trailing_stop_pct=trailing_stop_pct,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_dl_research_report(limit: int) -> int:
    context = build_app_context()
    lab = DlResearchLab(context.settings)
    result = lab.report(limit=limit)
    print(json.dumps(result, indent=2))
    return 0


def cmd_llm_research_run(
    windows: list[str],
    markets: list[str],
    commodity_symbols: list[str],
    stock_symbols: list[str],
    fx_symbols: list[str],
    stocks_only: bool,
    fx_only: bool,
    model_name: str | None,
    timeframe: str,
    regular_hours_only: bool,
    stop_loss_pct: float | None,
    trailing_stop_pct: float | None,
) -> int:
    context = build_app_context()
    if model_name:
        context.settings.llm_news_model_name = model_name
    lab = LlmNewsResearchLab(context.settings)
    result = lab.run(
        windows=windows,
        markets=markets,
        commodity_symbols=commodity_symbols,
        stock_symbols=stock_symbols,
        fx_symbols=fx_symbols,
        stocks_only=stocks_only,
        fx_only=fx_only,
        timeframe=timeframe,
        regular_hours_only=regular_hours_only,
        stop_loss_pct=stop_loss_pct,
        trailing_stop_pct=trailing_stop_pct,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_llm_research_report(limit: int) -> int:
    context = build_app_context()
    lab = LlmNewsResearchLab(context.settings)
    result = lab.report(limit=limit)
    print(json.dumps(result, indent=2))
    return 0


def cmd_deepseek_news_meta_run(
    training_symbols: list[str],
    evaluation_symbols: list[str],
    base_model_path: str,
    output_root: str,
    max_bars: int,
) -> int:
    context = build_app_context()
    lab = DeepseekNewsResearchLab(context.settings)
    result = lab.run(
        training_symbols=training_symbols or None,
        evaluation_symbols=evaluation_symbols or None,
        base_model_path=base_model_path,
        output_root=output_root,
        max_bars=max_bars,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_deepseek_directional_run(
    training_symbols: list[str],
    evaluation_symbols: list[str],
    base_model_path: str,
    output_root: str,
    max_bars: int,
    num_train_epochs: float,
    max_train_samples: int,
    max_eval_samples: int,
    include_base_model: bool,
    scoring_batch_size: int,
) -> int:
    context = build_app_context()
    lab = DeepseekDirectionalResearchLab(context.settings)
    result = lab.run(
        training_symbols=training_symbols or None,
        evaluation_symbols=evaluation_symbols or None,
        base_model_path=base_model_path,
        output_root=output_root,
        max_bars=max_bars,
        num_train_epochs=num_train_epochs,
        max_train_samples=max_train_samples if max_train_samples > 0 else None,
        max_eval_samples=max_eval_samples if max_eval_samples > 0 else None,
        include_base_model=include_base_model,
        scoring_batch_size=scoring_batch_size,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_deepseek_directional_crypto_run(
    training_symbols: list[str],
    evaluation_symbols: list[str],
    base_model_path: str,
    output_root: str,
    max_bars: int,
    num_train_epochs: float,
    max_train_samples: int | None,
    max_eval_samples: int | None,
    scoring_batch_size: int,
) -> int:
    context = build_app_context()
    lab = DeepseekDirectionalCryptoResearchLab(context.settings)
    result = lab.run(
        training_symbols=training_symbols,
        evaluation_symbols=evaluation_symbols,
        base_model_path=base_model_path,
        output_root=output_root,
        max_bars=max_bars,
        num_train_epochs=num_train_epochs,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        scoring_batch_size=scoring_batch_size,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_deepseek_directional_v4_run(
    output_root: str,
    base_model_path: str,
    max_bars: int,
    neutral_band: float,
) -> int:
    context = build_app_context()
    lab = DeepseekDirectionalV4ResearchLab(context.settings)
    result = lab.run(
        output_root=output_root,
        base_model_path=base_model_path,
        max_bars=max_bars,
        neutral_band=neutral_band,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_deepseek_directional_v35_variant_run() -> int:
    context = build_app_context()
    result = run_v3_top5_trailing_no_same_day_reentry(settings=context.settings)
    print(json.dumps(result, indent=2))
    return 0


def cmd_qwen_embedding_v3_q3e_run(
    output_root: str,
    model_name: str,
    max_bars: int,
    max_train_samples: int,
    max_length: int,
    batch_size: int,
    quantization: str,
    embargo_days: int,
    v35_scores_path: str,
    end_date: str,
    training_symbols: list[str],
    evaluation_symbols: list[str],
) -> int:
    context = build_app_context()
    lab = QwenEmbeddingDirectionalResearchLab(context.settings)
    result = lab.run(
        output_root=output_root,
        model_name=model_name,
        max_bars=max_bars,
        max_train_samples=max_train_samples,
        max_length=max_length,
        batch_size=batch_size,
        quantization=quantization,
        embargo_days=embargo_days,
        v35_scores_path=v35_scores_path,
        end_date=end_date or None,
        training_symbols=training_symbols or None,
        evaluation_symbols=evaluation_symbols or None,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_hybrid_alpha_run(scores_path: str, output_root: str, max_bars: int, eval_days_count: int, retrain_every: int) -> int:
    context = build_app_context()
    lab = HybridAlphaResearchLab(context.settings)
    result = lab.run(
        scores_path=scores_path,
        output_root=output_root,
        max_bars=max_bars,
        eval_days_count=eval_days_count,
        retrain_every=retrain_every,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_factor_ml_run(output_root: str, max_bars: int, eval_days_count: int, retrain_every: int, forward_horizon: int) -> int:
    context = build_app_context()
    lab = FactorMlResearchLab(context.settings)
    result = lab.run(
        output_root=output_root,
        max_bars=max_bars,
        eval_days_count=eval_days_count,
        retrain_every=retrain_every,
        forward_horizon=forward_horizon,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_factor_validation_run(
    output_root: str,
    max_bars: int,
    window_days: int,
    window_count: int,
    forward_horizon: int,
) -> int:
    context = build_app_context()
    lab = FactorValidationResearchLab(context.settings)
    result = lab.run(
        output_root=output_root,
        max_bars=max_bars,
        window_days=window_days,
        window_count=window_count,
        forward_horizon=forward_horizon,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_factor_news_overlay_run(
    scores_path: str,
    output_root: str,
    max_bars: int,
    window_days: int,
    window_count: int,
    forward_horizon: int,
    news_fresh_days: int,
) -> int:
    context = build_app_context()
    lab = FactorNewsOverlayResearchLab(context.settings)
    result = lab.run(
        scores_path=scores_path,
        output_root=output_root,
        max_bars=max_bars,
        window_days=window_days,
        window_count=window_count,
        forward_horizon=forward_horizon,
        news_fresh_days=news_fresh_days,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_sentiment_balanced_run(
    scores_path: str,
    output_root: str,
    max_bars: int,
    window_days: int,
    window_count: int,
    news_fresh_days: int,
    retrain_every: int,
) -> int:
    context = build_app_context()
    lab = SentimentBalancedResearchLab(context.settings)
    result = lab.run(
        scores_path=scores_path,
        output_root=output_root,
        max_bars=max_bars,
        window_days=window_days,
        window_count=window_count,
        news_fresh_days=news_fresh_days,
        retrain_every=retrain_every,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_embedding_sentiment_run(
    scores_path: str,
    embedding_cache_path: str,
    output_root: str,
    max_bars: int,
    window_days: int,
    window_count: int,
    news_fresh_days: int,
    retrain_every: int,
    svd_components: int,
    book_size: int,
) -> int:
    context = build_app_context()
    lab = EmbeddingSentimentResearchLab(context.settings)
    result = lab.run(
        scores_path=scores_path,
        embedding_cache_path=embedding_cache_path,
        output_root=output_root,
        max_bars=max_bars,
        window_days=window_days,
        window_count=window_count,
        news_fresh_days=news_fresh_days,
        retrain_every=retrain_every,
        svd_components=svd_components,
        book_size=book_size,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_portfolio_controls_run(
    output_root: str,
    max_bars: int,
    window_days: int,
    window_count: int,
    forward_horizon: int,
    book_size: int,
    rebalance_days: int,
) -> int:
    context = build_app_context()
    lab = PortfolioControlsResearchLab(context.settings)
    result = lab.run(
        output_root=output_root,
        max_bars=max_bars,
        window_days=window_days,
        window_count=window_count,
        forward_horizon=forward_horizon,
        book_size=book_size,
        rebalance_days=rebalance_days,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_regime_switch_run(
    output_root: str,
    max_bars: int,
    window_days: int,
    window_count: int,
    forward_horizon: int,
    book_size: int,
    rebalance_days: int,
    train_min_samples: int,
    embargo_days: int,
    label_drawdown_penalty: float,
) -> int:
    context = build_app_context()
    lab = RegimeSwitchResearchLab(context.settings)
    result = lab.run(
        output_root=output_root,
        max_bars=max_bars,
        window_days=window_days,
        window_count=window_count,
        forward_horizon=forward_horizon,
        book_size=book_size,
        rebalance_days=rebalance_days,
        train_min_samples=train_min_samples,
        embargo_days=embargo_days,
        label_drawdown_penalty=label_drawdown_penalty,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_book_size_sensitivity_run(
    output_root: str,
    max_bars: int,
    window_days: int,
    window_count: int,
    forward_horizon: int,
    book_sizes: list[int],
    rebalance_days: int,
) -> int:
    context = build_app_context()
    lab = BookSizeSensitivityResearchLab(context.settings)
    result = lab.run(
        output_root=output_root,
        max_bars=max_bars,
        window_days=window_days,
        window_count=window_count,
        forward_horizon=forward_horizon,
        book_sizes=tuple(book_sizes),
        rebalance_days=rebalance_days,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_price_momentum_validation_run(
    output_root: str,
    max_bars: int,
    window_days: int,
    window_count: int,
    forward_horizon: int,
    book_size: int,
    rebalance_days: int,
) -> int:
    context = build_app_context()
    lab = PriceMomentumValidationResearchLab(context.settings)
    result = lab.run(
        output_root=output_root,
        max_bars=max_bars,
        window_days=window_days,
        window_count=window_count,
        forward_horizon=forward_horizon,
        book_size=book_size,
        rebalance_days=rebalance_days,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_cross_asset_momentum_run(
    output_root: str,
    max_bars: int,
    lookback_days: int,
    rebalance_days: int,
    window_days: int,
    window_count: int,
    book_sizes: list[int],
) -> int:
    context = build_app_context()
    lab = CrossAssetMomentumResearchLab(context.settings)
    result = lab.run(
        output_root=output_root,
        max_bars=max_bars,
        lookback_days=lookback_days,
        rebalance_days=rebalance_days,
        window_days=window_days,
        window_count=window_count,
        book_sizes=tuple(book_sizes),
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_vol_carry_research(output_root: str, window_days: int, window_count: int, portfolio_cap: float) -> int:
    lab = VolCarryResearchLab()
    result = lab.run(
        output_root=output_root,
        window_days=window_days,
        window_count=window_count,
        portfolio_cap=portfolio_cap,
    )
    decision = result["decision"]
    print(
        json.dumps(
            {
                "version": result["version"],
                "status": decision["status"],
                "best_variant": decision.get("best_variant"),
                "failed_checks": decision.get("failed_checks"),
                "best_variant_full_history_metrics": decision.get("best_variant_full_history_metrics"),
                "output_root": output_root,
            },
            indent=2,
            default=str,
        )
    )
    return 0


def cmd_event_premia_research(output_root: str, start: str, panel_symbols: int, include_panel: bool) -> int:
    lab = EventPremiaResearchLab()
    result = lab.run(
        output_root=output_root,
        start=start,
        panel_symbols=panel_symbols,
        include_panel=include_panel,
    )
    decision = result["decision"]
    print(
        json.dumps(
            {
                "version": result["version"],
                "status": decision["status"],
                "survived_confirmation": decision["survived_confirmation"],
                "dropped_permanently": decision["dropped_permanently"],
                "sleeves_meeting_standalone_gate": decision["sleeves_meeting_standalone_gate"],
                "output_root": output_root,
            },
            indent=2,
            default=str,
        )
    )
    return 0


def cmd_alpha_factory_run(
    output_root: str,
    start: str,
    search_budget: int,
    seed: int,
    llm_model_path: str | None,
    llm_proposals: int,
) -> int:
    lab = AlphaFactoryResearchLab()
    result = lab.run(
        output_root=output_root,
        start=start,
        search_budget=search_budget,
        seed=seed,
        llm_model_path=llm_model_path,
        llm_proposals=llm_proposals,
    )
    decision = result["decision"]
    decay = result["decay_analysis"]
    print(
        json.dumps(
            {
                "version": result["version"],
                "status": decision["status"],
                "evaluations": result["search"]["evaluations"],
                "accepted_factors": decision.get("accepted_factor_count"),
                "validation_to_test_decay": decay.get("validation_to_test_decay"),
                "kill_rule_fired": decision.get("kill_rule_fired"),
                "ensemble_test_sharpe": decision.get("ensemble_test_sharpe"),
                "incumbent_resid_mom_60d_test_sharpe": decision.get("incumbent_resid_mom_60d_test_sharpe"),
                "output_root": output_root,
            },
            indent=2,
            default=str,
        )
    )
    return 0


def cmd_long_short_corrected_run(scores_path: str, output_root: str, eval_days_count: int, book_size: int) -> int:
    context = build_app_context()
    lab = LongShortCorrectedLab(context.settings)
    result = lab.run(
        scores_path=scores_path,
        output_root=output_root,
        eval_days_count=eval_days_count,
        book_size=book_size,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_cross_sectional_retrain_run(
    output_root: str,
    base_model_path: str,
    max_bars: int,
    eval_days_count: int,
    num_train_epochs: float,
    max_train_samples: int,
) -> int:
    context = build_app_context()
    lab = CrossSectionalRetrainLab(context.settings)
    result = lab.run(
        output_root=output_root,
        base_model_path=base_model_path,
        max_bars=max_bars,
        eval_days_count=eval_days_count,
        num_train_epochs=num_train_epochs,
        max_train_samples=max_train_samples,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_long_short_research_run(
    output_root: str,
    scores_path: str,
    max_bars: int,
    eval_days_count: int,
    book_size: int,
    rebalance_days: int,
    cost_bps_per_side: float,
    short_borrow_annual: float,
    book_mode: str,
) -> int:
    context = build_app_context()
    lab = LongShortResearchLab(context.settings)
    result = lab.run(
        output_root=output_root,
        scores_path=scores_path,
        max_bars=max_bars,
        eval_days_count=eval_days_count,
        book_size=book_size,
        rebalance_days=rebalance_days,
        cost_bps_per_side=cost_bps_per_side,
        short_borrow_annual=short_borrow_annual,
        book_mode=book_mode,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_v35_execution_controls_run(
    output_root: str,
    v35_scores_path: str,
    end_date: str,
    seeds: int,
    evaluation_symbols: list[str],
) -> int:
    context = build_app_context()
    result = run_v35_execution_controls(
        context.settings,
        output_root=output_root,
        v35_scores_path=v35_scores_path,
        symbols=evaluation_symbols or None,
        end_date=end_date or None,
        seeds=seeds,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_deepseek_directional_v35_hourly_run(
    training_symbols: list[str],
    evaluation_symbols: list[str],
    base_model_path: str,
    output_root: str,
    max_bars: int,
    num_train_epochs: float,
    max_train_samples: int,
    max_eval_samples: int,
    scoring_batch_size: int,
) -> int:
    context = build_app_context()
    lab = DeepseekDirectionalV3HourlyLab(context.settings)
    result = lab.run(
        training_symbols=training_symbols or None,
        evaluation_symbols=evaluation_symbols or None,
        base_model_path=base_model_path,
        output_root=output_root,
        max_bars=max_bars,
        num_train_epochs=num_train_epochs,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        scoring_batch_size=scoring_batch_size,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_v35_paper_run(submit: bool, target_capital: float) -> int:
    context = build_app_context()
    trader = V35PaperTrader(context.settings)
    result = trader.run_cycle(submit=submit, target_capital=target_capital)
    print(json.dumps(result, indent=2))
    return 0


def cmd_v35_paper_report() -> int:
    context = build_app_context()
    trader = V35PaperTrader(context.settings)
    result = trader.weekly_report()
    print(json.dumps(result, indent=2))
    return 0


def cmd_v35_intraday_run(
    submit: bool,
    target_capital: float,
    lookback_bars: int,
    max_cycles: int,
    sleep_seconds: int,
    news_check_seconds: int,
) -> int:
    context = build_app_context()
    trader = V35IntradayTrader(context.settings)
    result = trader.run_active_hours(
        submit=submit,
        target_capital=target_capital,
        lookback_bars=lookback_bars,
        max_cycles=max_cycles,
        sleep_seconds=sleep_seconds,
        news_check_seconds=news_check_seconds,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_deepseek_strict_compare_run(
    previous_report_path: str,
    meta_artifact_path: str,
    output_path: str,
    max_bars: int,
) -> int:
    context = build_app_context()
    lab = DeepseekStrictComparisonLab(context.settings)
    result = lab.run(
        previous_report_path=previous_report_path,
        meta_artifact_path=meta_artifact_path,
        output_path=output_path,
        max_bars=max_bars,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_deepseek_strict_compare_pdf(comparison_artifact: str, output_path: str | None) -> int:
    result = build_deepseek_strict_compare_pdf(
        comparison_artifact=comparison_artifact,
        output_path=output_path,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_llm_overlay_pdf(hourly_artifact: str, output_path: str | None) -> int:
    result = build_llm_overlay_comparison_pdf(hourly_artifact=hourly_artifact, output_path=output_path)
    print(json.dumps(result, indent=2))
    return 0


def cmd_dl_fixed_split_run(
    markets: list[str],
    models: list[str],
    symbols: list[str],
    timeframe: str,
    regular_hours_only: bool,
    no_same_day_reentry: bool,
    train_bars: int,
    test_bars: int,
    stop_loss_pct: float | None,
    trailing_stop_pct: float | None,
) -> int:
    context = build_app_context()
    lab = DlResearchLab(context.settings)
    result = lab.run_fixed_split(
        markets=markets,
        models=models,
        symbols=symbols,
        timeframe=timeframe,
        regular_hours_only=regular_hours_only,
        no_same_day_reentry=no_same_day_reentry,
        train_bars=train_bars,
        test_bars=test_bars,
        stop_loss_pct=stop_loss_pct,
        trailing_stop_pct=trailing_stop_pct,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_research_comparison_pdf(output_path: str | None, dl_no_rule_artifact: str | None) -> int:
    result = build_research_comparison_pdf(
        output_path=output_path,
        dl_no_rule_artifact=dl_no_rule_artifact,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_historical_chart(
    market: str,
    symbol: str,
    strategy_name: str,
    window_name: str,
    output_path: str | None,
    timeframe: str,
    regular_hours_only: bool,
    no_same_day_reentry: bool,
) -> int:
    context = build_app_context()
    builder = HistoricalChartBuilder(context.settings)
    result = builder.build_chart(
        market=market,
        symbol=symbol,
        strategy_name=strategy_name,
        window_name=window_name,
        output_path=output_path,
        timeframe=timeframe,
        regular_hours_only=regular_hours_only,
        no_same_day_reentry=no_same_day_reentry,
    )
    print(
        json.dumps(
            {
                "output_path": result.output_path,
                "market": result.market,
                "symbol": result.symbol,
                "strategy_name": result.strategy_name,
                "window_name": result.window_name,
                "summary": result.summary,
            },
            indent=2,
        )
    )
    return 0


def cmd_historical_chart_interactive(
    market: str,
    symbol: str,
    strategy_name: str,
    window_name: str,
    output_path: str | None,
    timeframe: str,
    regular_hours_only: bool,
    no_same_day_reentry: bool,
) -> int:
    context = build_app_context()
    builder = HistoricalChartBuilder(context.settings)
    result = builder.build_interactive_chart(
        market=market,
        symbol=symbol,
        strategy_name=strategy_name,
        window_name=window_name,
        output_path=output_path,
        timeframe=timeframe,
        regular_hours_only=regular_hours_only,
        no_same_day_reentry=no_same_day_reentry,
    )
    print(
        json.dumps(
            {
                "output_path": result.output_path,
                "market": result.market,
                "symbol": result.symbol,
                "strategy_name": result.strategy_name,
                "window_name": result.window_name,
                "summary": result.summary,
            },
            indent=2,
        )
    )
    return 0


def cmd_historical_cost_aware_run(
    market: str,
    symbol: str,
    strategy_name: str,
    window_name: str,
    timeframe: str,
    stop_loss_pct: float | None,
    trailing_stop_pct: float | None,
    starting_capital: float,
    commission_per_order: float,
    quoted_spread_bps: float,
    market_impact_bps: float,
    stop_extra_slippage_bps: float,
    max_bar_participation_rate: float,
    max_bar_shares: float | None,
    sec_fee_per_million_sell: float,
    finra_taf_per_share_sell: float,
    finra_taf_cap_per_trade: float,
    regular_hours_only: bool,
    no_same_day_reentry: bool,
) -> int:
    context = build_app_context()
    risk_limits = RiskLimits.from_file().with_overrides(
        stop_loss_pct=stop_loss_pct,
        trailing_stop_pct=trailing_stop_pct,
    )
    builder = HistoricalChartBuilder(context.settings, risk_limits=risk_limits)
    _, normalized_symbol, _, bars, trace = builder.build_trace(
        market=market,
        symbol=symbol,
        strategy_name=strategy_name,
        window_name=window_name,
        risk_limits=risk_limits,
        timeframe=timeframe,
        regular_hours_only=regular_hours_only,
        no_same_day_reentry=no_same_day_reentry,
    )
    assumptions = ExecutionAssumptions(
        starting_capital=starting_capital,
        commission_per_order=commission_per_order,
        quoted_spread_bps=quoted_spread_bps,
        market_impact_bps=market_impact_bps,
        stop_extra_slippage_bps=stop_extra_slippage_bps,
        max_bar_participation_rate=max_bar_participation_rate,
        max_bar_shares=max_bar_shares,
        sec_fee_per_million_sell=sec_fee_per_million_sell,
        finra_taf_per_share_sell=finra_taf_per_share_sell,
        finra_taf_cap_per_trade=finra_taf_cap_per_trade,
    )
    result = run_execution_aware_replay(
        bars=bars,
        trace=trace,
        assumptions=assumptions,
    )
    print(
        json.dumps(
            {
                "market": market,
                "symbol": normalized_symbol,
                "strategy_name": strategy_name,
                "window_name": f"{window_name}_{timeframe}",
                "summary": result["summary"],
            },
            indent=2,
        )
    )
    return 0


def cmd_xle_daily_cycle(symbol: str, bars: int, submit: bool) -> int:
    context = build_app_context()
    lab = XleStrategyLab(context.settings)
    run = lab.build_best_strategy_run(symbol=symbol, bars=bars)
    order_results = lab.execute_best_strategy(run, submit=submit)
    sync_result = PaperBrokerSync(context.settings).run(days=14)
    account = MomentumPaperTrader(context.settings).get_account_summary()
    print(
        json.dumps(
            {
                "symbol": symbol,
                "submitted": submit,
                "best_strategy": run.best,
                "current_position_qty": run.current_position_qty,
                "orders": order_results,
                "sync": sync_result,
                "account": account,
            },
            indent=2,
        )
    )
    return 0


def cmd_paper_sync(days: int) -> int:
    context = build_app_context()
    sync_result = PaperBrokerSync(context.settings).run(days=days)
    print(json.dumps(sync_result, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="algoding")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("show-config")
    subparsers.add_parser("healthcheck")
    seed_parser = subparsers.add_parser("seed-sample")
    seed_parser.add_argument("--symbol", default="SPY")
    seed_parser.add_argument("--price", type=float, default=500.0)
    subparsers.add_parser("questdb-count")
    ingest_parser = subparsers.add_parser("ingest-bars")
    ingest_parser.add_argument("--symbols", default="")
    ingest_parser.add_argument("--max-events", type=int, default=None)
    shadow_parser = subparsers.add_parser("shadow-run")
    shadow_parser.add_argument("--symbols", default="")
    shadow_parser.add_argument("--max-events", type=int, default=None)
    subparsers.add_parser("paper-account")
    paper_parser = subparsers.add_parser("paper-rebalance")
    paper_parser.add_argument("--symbols", default="")
    paper_parser.add_argument("--submit", action="store_true")
    basket_parser = subparsers.add_parser("basket-daily-cycle")
    basket_parser.add_argument("--symbols", default="")
    basket_parser.add_argument("--submit", action="store_true")
    basket_report_parser = subparsers.add_parser("basket-report")
    basket_report_parser.add_argument("--limit", type=int, default=10)
    paper_sync_parser = subparsers.add_parser("paper-sync")
    paper_sync_parser.add_argument("--days", type=int, default=14)
    xle_compare_parser = subparsers.add_parser("xle-compare")
    xle_compare_parser.add_argument("--symbol", default="XLE")
    xle_compare_parser.add_argument("--bars", type=int, default=90)
    xle_report_parser = subparsers.add_parser("xle-report")
    xle_report_parser.add_argument("--symbol", default="XLE")
    xle_report_parser.add_argument("--limit", type=int, default=10)
    xle_run_parser = subparsers.add_parser("xle-run-best")
    xle_run_parser.add_argument("--symbol", default="XLE")
    xle_run_parser.add_argument("--bars", type=int, default=90)
    xle_run_parser.add_argument("--submit", action="store_true")
    xle_parallel_run_parser = subparsers.add_parser("xle-parallel-run")
    xle_parallel_run_parser.add_argument("--symbol", default="XLE")
    xle_parallel_run_parser.add_argument("--bars", type=int, default=90)
    xle_parallel_report_parser = subparsers.add_parser("xle-parallel-report")
    xle_parallel_report_parser.add_argument("--symbol", default="XLE")
    xle_parallel_history_parser = subparsers.add_parser("xle-parallel-history")
    xle_parallel_history_parser.add_argument("--symbol", default="XLE")
    xle_parallel_history_parser.add_argument("--strategy-name", default=None)
    xle_parallel_history_parser.add_argument("--limit", type=int, default=50)
    research_parallel_run_parser = subparsers.add_parser("research-parallel-run")
    research_parallel_run_parser.add_argument("--market", default="gold")
    research_parallel_run_parser.add_argument("--bars", type=int, default=90)
    research_parallel_report_parser = subparsers.add_parser("research-parallel-report")
    research_parallel_report_parser.add_argument("--market", default="gold")
    research_parallel_history_parser = subparsers.add_parser("research-parallel-history")
    research_parallel_history_parser.add_argument("--symbol", default="GLD")
    research_parallel_history_parser.add_argument("--strategy-name", default=None)
    research_parallel_history_parser.add_argument("--limit", type=int, default=50)
    promote_candidate_parser = subparsers.add_parser("promote-candidate")
    promote_candidate_parser.add_argument("--market", required=True)
    promote_candidate_parser.add_argument("--symbol", required=True)
    promote_candidate_parser.add_argument("--strategy-name", required=True)
    promote_candidate_parser.add_argument("--notional", type=float, default=1000.0)
    subparsers.add_parser("promoted-report")
    promoted_run_parser = subparsers.add_parser("promoted-run")
    promoted_run_parser.add_argument("--submit", action="store_true")
    historical_research_run_parser = subparsers.add_parser("historical-research-run")
    historical_research_run_parser.add_argument("--markets", default="etfs,commodities,stocks,fx")
    historical_research_run_parser.add_argument("--windows", default="6m,1y")
    historical_research_run_parser.add_argument("--timeframe", default="day")
    historical_research_run_parser.add_argument("--regular-hours-only", action="store_true")
    historical_research_run_parser.add_argument("--execution-aware", action="store_true")
    historical_research_run_parser.add_argument("--no-same-day-reentry", action="store_true")
    historical_research_run_parser.add_argument("--stop-loss-pct", type=float, default=None)
    historical_research_run_parser.add_argument("--trailing-stop-pct", type=float, default=None)
    historical_research_report_parser = subparsers.add_parser("historical-research-report")
    historical_research_report_parser.add_argument("--limit", type=int, default=50)
    historical_research_report_parser.add_argument("--execution-aware", action="store_true")
    ml_research_run_parser = subparsers.add_parser("ml-research-run")
    ml_research_run_parser.add_argument("--markets", default="commodities,etfs,stocks")
    ml_research_run_parser.add_argument("--windows", default="1y,2y,3y")
    ml_research_run_parser.add_argument(
        "--models",
        default="logistic_regression,hist_gradient_boosting,mlp_classifier",
    )
    ml_research_run_parser.add_argument("--timeframe", default="day")
    ml_research_run_parser.add_argument("--regular-hours-only", action="store_true")
    ml_research_run_parser.add_argument("--no-same-day-reentry", action="store_true")
    ml_research_run_parser.add_argument("--stop-loss-pct", type=float, default=None)
    ml_research_run_parser.add_argument("--trailing-stop-pct", type=float, default=None)
    ml_research_report_parser = subparsers.add_parser("ml-research-report")
    ml_research_report_parser.add_argument("--limit", type=int, default=50)
    dl_research_run_parser = subparsers.add_parser("dl-research-run")
    dl_research_run_parser.add_argument("--markets", default="commodities,stocks")
    dl_research_run_parser.add_argument("--windows", default="1y,2y,3y")
    dl_research_run_parser.add_argument("--models", default="torch_mlp,torch_lstm,torch_cnn")
    dl_research_run_parser.add_argument("--timeframe", default="hour")
    dl_research_run_parser.add_argument("--regular-hours-only", action="store_true")
    dl_research_run_parser.add_argument("--no-same-day-reentry", action="store_true")
    dl_research_run_parser.add_argument("--stop-loss-pct", type=float, default=None)
    dl_research_run_parser.add_argument("--trailing-stop-pct", type=float, default=0.015)
    dl_research_report_parser = subparsers.add_parser("dl-research-report")
    dl_research_report_parser.add_argument("--limit", type=int, default=50)
    llm_research_run_parser = subparsers.add_parser("llm-news-research-run")
    llm_research_run_parser.add_argument("--windows", default="1y,2y,3y")
    llm_research_run_parser.add_argument("--markets", default="")
    llm_research_run_parser.add_argument("--commodity-symbols", default="")
    llm_research_run_parser.add_argument("--stock-symbols", default="")
    llm_research_run_parser.add_argument("--fx-symbols", default="")
    llm_research_run_parser.add_argument("--stocks-only", action="store_true")
    llm_research_run_parser.add_argument("--fx-only", action="store_true")
    llm_research_run_parser.add_argument("--model-name", default=None)
    llm_research_run_parser.add_argument("--timeframe", default="day")
    llm_research_run_parser.add_argument("--regular-hours-only", action="store_true")
    llm_research_run_parser.add_argument("--stop-loss-pct", type=float, default=0.05)
    llm_research_run_parser.add_argument("--trailing-stop-pct", type=float, default=0.015)
    llm_research_report_parser = subparsers.add_parser("llm-news-research-report")
    llm_research_report_parser.add_argument("--limit", type=int, default=50)
    deepseek_news_meta_parser = subparsers.add_parser("deepseek-news-meta-run")
    deepseek_news_meta_parser.add_argument("--training-symbols", default="")
    deepseek_news_meta_parser.add_argument("--evaluation-symbols", default="")
    deepseek_news_meta_parser.add_argument("--base-model-path", default=r"U:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B")
    deepseek_news_meta_parser.add_argument("--output-root", default="reports/research/deepseek_news_price_meta")
    deepseek_news_meta_parser.add_argument("--max-bars", type=int, default=800)
    deepseek_directional_parser = subparsers.add_parser("deepseek-directional-run")
    deepseek_directional_parser.add_argument("--training-symbols", default="")
    deepseek_directional_parser.add_argument("--evaluation-symbols", default="")
    deepseek_directional_parser.add_argument("--base-model-path", default=r"E:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B")
    deepseek_directional_parser.add_argument("--output-root", default="reports/research/deepseek_directional")
    deepseek_directional_parser.add_argument("--max-bars", type=int, default=800)
    deepseek_directional_parser.add_argument("--num-train-epochs", type=float, default=1.0)
    deepseek_directional_parser.add_argument("--max-train-samples", type=int, default=1200)
    deepseek_directional_parser.add_argument("--max-eval-samples", type=int, default=300)
    deepseek_directional_parser.add_argument("--include-base-model", action="store_true")
    deepseek_directional_parser.add_argument("--scoring-batch-size", type=int, default=4)
    deepseek_directional_crypto_parser = subparsers.add_parser("deepseek-directional-crypto-run")
    deepseek_directional_crypto_parser.add_argument("--training-symbols", default="BTC/USD,ETH/USD,SOL/USD,DOGE/USD,LTC/USD,BCH/USD,AVAX/USD,LINK/USD,UNI/USD,AAVE/USD,XRP/USD")
    deepseek_directional_crypto_parser.add_argument("--evaluation-symbols", default="BTC/USD,ETH/USD,SOL/USD,DOGE/USD,XRP/USD")
    deepseek_directional_crypto_parser.add_argument("--base-model-path", default=r"E:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B")
    deepseek_directional_crypto_parser.add_argument("--output-root", default="reports/research/deepseek_directional_crypto_cv0")
    deepseek_directional_crypto_parser.add_argument("--max-bars", type=int, default=800)
    deepseek_directional_crypto_parser.add_argument("--num-train-epochs", type=float, default=1.0)
    deepseek_directional_crypto_parser.add_argument("--max-train-samples", type=int, default=1200)
    deepseek_directional_crypto_parser.add_argument("--max-eval-samples", type=int, default=300)
    deepseek_directional_crypto_parser.add_argument("--scoring-batch-size", type=int, default=4)
    deepseek_directional_v4_parser = subparsers.add_parser("deepseek-directional-v4-run")
    deepseek_directional_v4_parser.add_argument("--output-root", default="reports/research/deepseek_directional_v4")
    deepseek_directional_v4_parser.add_argument("--base-model-path", default=r"E:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B")
    deepseek_directional_v4_parser.add_argument("--max-bars", type=int, default=1400)
    deepseek_directional_v4_parser.add_argument("--neutral-band", type=float, default=0.3)
    deepseek_directional_v35_hourly_parser = subparsers.add_parser("deepseek-directional-v35-hourly-run")
    deepseek_directional_v35_hourly_parser.add_argument("--training-symbols", default="")
    deepseek_directional_v35_hourly_parser.add_argument("--evaluation-symbols", default="")
    deepseek_directional_v35_hourly_parser.add_argument("--base-model-path", default=r"E:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B")
    deepseek_directional_v35_hourly_parser.add_argument("--output-root", default="reports/research/deepseek_directional_v3_hourly")
    deepseek_directional_v35_hourly_parser.add_argument("--max-bars", type=int, default=3500)
    deepseek_directional_v35_hourly_parser.add_argument("--num-train-epochs", type=float, default=1.0)
    deepseek_directional_v35_hourly_parser.add_argument("--max-train-samples", type=int, default=1200)
    deepseek_directional_v35_hourly_parser.add_argument("--max-eval-samples", type=int, default=300)
    deepseek_directional_v35_hourly_parser.add_argument("--scoring-batch-size", type=int, default=4)
    subparsers.add_parser("deepseek-directional-v35-variant-run")
    qwen_q3e_parser = subparsers.add_parser("qwen-embedding-v3-q3e-run")
    qwen_q3e_parser.add_argument("--output-root", default="reports/research/qwen_embedding_v3_q3e")
    qwen_q3e_parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-4B")
    qwen_q3e_parser.add_argument("--max-bars", type=int, default=1400)
    qwen_q3e_parser.add_argument("--max-train-samples", type=int, default=10000)
    qwen_q3e_parser.add_argument("--max-length", type=int, default=768)
    qwen_q3e_parser.add_argument("--batch-size", type=int, default=4)
    qwen_q3e_parser.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    qwen_q3e_parser.add_argument("--embargo-days", type=int, default=5)
    qwen_q3e_parser.add_argument(
        "--v35-scores-path",
        default="reports/research/deepseek_directional_v3/finetuned_directional_scores.jsonl",
    )
    qwen_q3e_parser.add_argument(
        "--end-date",
        default="",
        help="Cap the evaluation window (YYYY-MM-DD). Defaults to the last day in the V3-5 score cache.",
    )
    qwen_q3e_parser.add_argument("--training-symbols", default="")
    qwen_q3e_parser.add_argument("--evaluation-symbols", default="")
    hybrid_parser = subparsers.add_parser("hybrid-alpha-run")
    hybrid_parser.add_argument("--scores-path", default="reports/research/long_short_v1/broad_directional_scores.jsonl")
    hybrid_parser.add_argument("--output-root", default="reports/research/hybrid_v1")
    hybrid_parser.add_argument("--max-bars", type=int, default=450)
    hybrid_parser.add_argument("--eval-days-count", type=int, default=200)
    hybrid_parser.add_argument("--retrain-every", type=int, default=20)
    factor_ml_parser = subparsers.add_parser("factor-ml-run")
    factor_ml_parser.add_argument("--output-root", default="reports/research/factor_ml_v1")
    factor_ml_parser.add_argument("--max-bars", type=int, default=900)
    factor_ml_parser.add_argument("--eval-days-count", type=int, default=252)
    factor_ml_parser.add_argument("--retrain-every", type=int, default=20)
    factor_ml_parser.add_argument("--forward-horizon", type=int, default=20)
    factor_validation_parser = subparsers.add_parser("factor-validation-run")
    factor_validation_parser.add_argument("--output-root", default="reports/research/factor_validation_v1")
    factor_validation_parser.add_argument("--max-bars", type=int, default=1200)
    factor_validation_parser.add_argument("--window-days", type=int, default=126)
    factor_validation_parser.add_argument("--window-count", type=int, default=5)
    factor_validation_parser.add_argument("--forward-horizon", type=int, default=20)
    factor_news_parser = subparsers.add_parser("factor-news-overlay-run")
    factor_news_parser.add_argument("--scores-path", default="reports/research/long_short_v1/broad_directional_scores.jsonl")
    factor_news_parser.add_argument("--output-root", default="reports/research/factor_news_overlay_v1")
    factor_news_parser.add_argument("--max-bars", type=int, default=1200)
    factor_news_parser.add_argument("--window-days", type=int, default=126)
    factor_news_parser.add_argument("--window-count", type=int, default=3)
    factor_news_parser.add_argument("--forward-horizon", type=int, default=20)
    factor_news_parser.add_argument("--news-fresh-days", type=int, default=3)
    sentiment_balanced_parser = subparsers.add_parser("sentiment-balanced-run")
    sentiment_balanced_parser.add_argument("--scores-path", default="reports/research/long_short_v1/broad_directional_scores.jsonl")
    sentiment_balanced_parser.add_argument("--output-root", default="reports/research/sentiment_v2_balanced")
    sentiment_balanced_parser.add_argument("--max-bars", type=int, default=1200)
    sentiment_balanced_parser.add_argument("--window-days", type=int, default=126)
    sentiment_balanced_parser.add_argument("--window-count", type=int, default=3)
    sentiment_balanced_parser.add_argument("--news-fresh-days", type=int, default=3)
    sentiment_balanced_parser.add_argument("--retrain-every", type=int, default=20)
    embedding_sentiment_parser = subparsers.add_parser("embedding-sentiment-run")
    embedding_sentiment_parser.add_argument("--scores-path", default="reports/research/long_short_v1/broad_directional_scores.jsonl")
    embedding_sentiment_parser.add_argument("--embedding-cache-path", default="reports/research/qwen_embedding_v3_q3e/qwen3_embedding_cache.jsonl")
    embedding_sentiment_parser.add_argument("--output-root", default="reports/research/sentiment_v3_embedding")
    embedding_sentiment_parser.add_argument("--max-bars", type=int, default=1400)
    embedding_sentiment_parser.add_argument("--window-days", type=int, default=126)
    embedding_sentiment_parser.add_argument("--window-count", type=int, default=3)
    embedding_sentiment_parser.add_argument("--news-fresh-days", type=int, default=3)
    embedding_sentiment_parser.add_argument("--retrain-every", type=int, default=20)
    embedding_sentiment_parser.add_argument("--svd-components", type=int, default=64)
    embedding_sentiment_parser.add_argument("--book-size", type=int, default=10)
    portfolio_controls_parser = subparsers.add_parser("portfolio-controls-run")
    portfolio_controls_parser.add_argument("--output-root", default="reports/research/portfolio_controls_v1")
    portfolio_controls_parser.add_argument("--max-bars", type=int, default=1200)
    portfolio_controls_parser.add_argument("--window-days", type=int, default=126)
    portfolio_controls_parser.add_argument("--window-count", type=int, default=5)
    portfolio_controls_parser.add_argument("--forward-horizon", type=int, default=20)
    portfolio_controls_parser.add_argument("--book-size", type=int, default=20)
    portfolio_controls_parser.add_argument("--rebalance-days", type=int, default=5)
    regime_switch_parser = subparsers.add_parser("regime-switch-run")
    regime_switch_parser.add_argument("--output-root", default="reports/research/regime_switch_v1")
    regime_switch_parser.add_argument("--max-bars", type=int, default=1200)
    regime_switch_parser.add_argument("--window-days", type=int, default=126)
    regime_switch_parser.add_argument("--window-count", type=int, default=5)
    regime_switch_parser.add_argument("--forward-horizon", type=int, default=20)
    regime_switch_parser.add_argument("--book-size", type=int, default=20)
    regime_switch_parser.add_argument("--rebalance-days", type=int, default=5)
    regime_switch_parser.add_argument("--train-min-samples", type=int, default=60)
    regime_switch_parser.add_argument("--embargo-days", type=int, default=5)
    regime_switch_parser.add_argument("--label-drawdown-penalty", type=float, default=0.35)
    book_size_parser = subparsers.add_parser("book-size-sensitivity-run")
    book_size_parser.add_argument("--output-root", default="reports/research/book_size_sensitivity_v1")
    book_size_parser.add_argument("--max-bars", type=int, default=1200)
    book_size_parser.add_argument("--window-days", type=int, default=126)
    book_size_parser.add_argument("--window-count", type=int, default=5)
    book_size_parser.add_argument("--forward-horizon", type=int, default=20)
    book_size_parser.add_argument("--book-sizes", default="5,10,15,20,25,30,40")
    book_size_parser.add_argument("--rebalance-days", type=int, default=5)
    price_momentum_parser = subparsers.add_parser("price-momentum-validation-run")
    price_momentum_parser.add_argument("--output-root", default="reports/research/price_momentum_validation_v1")
    price_momentum_parser.add_argument("--max-bars", type=int, default=1200)
    price_momentum_parser.add_argument("--window-days", type=int, default=126)
    price_momentum_parser.add_argument("--window-count", type=int, default=5)
    price_momentum_parser.add_argument("--forward-horizon", type=int, default=20)
    price_momentum_parser.add_argument("--book-size", type=int, default=20)
    price_momentum_parser.add_argument("--rebalance-days", type=int, default=5)
    cross_asset_momentum_parser = subparsers.add_parser("cross-asset-momentum-run")
    cross_asset_momentum_parser.add_argument("--output-root", default="reports/research/cross_asset_momentum_v1")
    cross_asset_momentum_parser.add_argument("--max-bars", type=int, default=1600)
    cross_asset_momentum_parser.add_argument("--lookback-days", type=int, default=20)
    cross_asset_momentum_parser.add_argument("--rebalance-days", type=int, default=5)
    cross_asset_momentum_parser.add_argument("--window-days", type=int, default=126)
    cross_asset_momentum_parser.add_argument("--window-count", type=int, default=6)
    cross_asset_momentum_parser.add_argument("--book-sizes", default="20,5")
    vol_carry_parser = subparsers.add_parser("vol-carry-research")
    vol_carry_parser.add_argument("--output-root", default="reports/research/vol_carry_v1")
    vol_carry_parser.add_argument("--window-days", type=int, default=126)
    vol_carry_parser.add_argument("--window-count", type=int, default=5)
    vol_carry_parser.add_argument("--portfolio-cap", type=float, default=0.05)
    event_premia_parser = subparsers.add_parser("event-premia-research")
    event_premia_parser.add_argument("--output-root", default="reports/research/event_premia_v1")
    event_premia_parser.add_argument("--start", default="2005-01-01")
    event_premia_parser.add_argument("--panel-symbols", type=int, default=100)
    event_premia_parser.add_argument("--no-panel", action="store_true")
    alpha_factory_parser = subparsers.add_parser("alpha-factory-run")
    alpha_factory_parser.add_argument("--output-root", default="reports/research/alpha_factory_v1")
    alpha_factory_parser.add_argument("--start", default="2005-01-01")
    alpha_factory_parser.add_argument("--search-budget", type=int, default=4000)
    alpha_factory_parser.add_argument("--seed", type=int, default=11)
    alpha_factory_parser.add_argument("--llm-model-path", default=None)
    alpha_factory_parser.add_argument("--llm-proposals", type=int, default=0)
    ls_corrected_parser = subparsers.add_parser("long-short-corrected-run")
    ls_corrected_parser.add_argument("--scores-path", required=True)
    ls_corrected_parser.add_argument("--output-root", default="reports/research/long_short_v1_corrected")
    ls_corrected_parser.add_argument("--eval-days-count", type=int, default=280)
    ls_corrected_parser.add_argument("--book-size", type=int, default=20)
    xs_retrain_parser = subparsers.add_parser("cross-sectional-retrain-run")
    xs_retrain_parser.add_argument("--output-root", default="reports/research/long_short_v1")
    xs_retrain_parser.add_argument("--base-model-path", default=r"E:\Deepseek\models\DeepSeek-R1-Distill-Qwen-1.5B")
    xs_retrain_parser.add_argument("--max-bars", type=int, default=600)
    xs_retrain_parser.add_argument("--eval-days-count", type=int, default=280)
    xs_retrain_parser.add_argument("--num-train-epochs", type=float, default=1.0)
    xs_retrain_parser.add_argument("--max-train-samples", type=int, default=6000)
    long_short_parser = subparsers.add_parser("long-short-research-run")
    long_short_parser.add_argument("--output-root", default="reports/research/long_short_v1")
    long_short_parser.add_argument("--scores-path", default="reports/research/long_short_v1/broad_directional_scores.jsonl")
    long_short_parser.add_argument("--max-bars", type=int, default=450)
    long_short_parser.add_argument("--eval-days-count", type=int, default=280)
    long_short_parser.add_argument("--book-size", type=int, default=20)
    long_short_parser.add_argument("--rebalance-days", type=int, default=5)
    long_short_parser.add_argument("--cost-bps-per-side", type=float, default=1.0)
    long_short_parser.add_argument("--short-borrow-annual", type=float, default=0.01)
    long_short_parser.add_argument("--book-mode", default="sign", choices=["sign", "rank"])
    v35_controls_parser = subparsers.add_parser("v35-execution-controls-run")
    v35_controls_parser.add_argument("--output-root", default="reports/research/qwen_embedding_v3_q3e/controls")
    v35_controls_parser.add_argument(
        "--v35-scores-path",
        default="reports/research/deepseek_directional_v3/finetuned_directional_scores.jsonl",
    )
    v35_controls_parser.add_argument("--end-date", default="")
    v35_controls_parser.add_argument("--seeds", type=int, default=25)
    v35_controls_parser.add_argument("--evaluation-symbols", default="")
    v35_paper_run_parser = subparsers.add_parser("v35-paper-run")
    v35_paper_run_parser.add_argument("--submit", action="store_true")
    v35_paper_run_parser.add_argument("--target-capital", type=float, default=60000.0)
    subparsers.add_parser("v35-paper-report")
    v35_intraday_run_parser = subparsers.add_parser("v35-intraday-run")
    v35_intraday_run_parser.add_argument("--submit", action="store_true")
    v35_intraday_run_parser.add_argument("--target-capital", type=float, default=60000.0)
    v35_intraday_run_parser.add_argument("--lookback-bars", type=int, default=180)
    v35_intraday_run_parser.add_argument("--max-cycles", type=int, default=1)
    v35_intraday_run_parser.add_argument("--sleep-seconds", type=int, default=60)
    v35_intraday_run_parser.add_argument("--news-check-seconds", type=int, default=600)
    deepseek_strict_compare_parser = subparsers.add_parser("deepseek-strict-compare-run")
    deepseek_strict_compare_parser.add_argument("--previous-report-path", default="reports/research/llm_news_sentiment_deepseek_stocks_report_full.json")
    deepseek_strict_compare_parser.add_argument("--meta-artifact-path", default="reports/research/deepseek_news_price_meta_full/deepseek_news_price_meta_run.json")
    deepseek_strict_compare_parser.add_argument("--output-path", default="reports/research/deepseek_top5_strict_exact_vs_meta.json")
    deepseek_strict_compare_parser.add_argument("--max-bars", type=int, default=800)
    deepseek_strict_compare_pdf_parser = subparsers.add_parser("deepseek-strict-compare-pdf")
    deepseek_strict_compare_pdf_parser.add_argument("--comparison-artifact", default="reports/research/deepseek_top5_strict_exact_vs_meta.json")
    deepseek_strict_compare_pdf_parser.add_argument("--output-path", default=None)
    llm_overlay_pdf_parser = subparsers.add_parser("llm-overlay-pdf")
    llm_overlay_pdf_parser.add_argument("--hourly-artifact", required=True)
    llm_overlay_pdf_parser.add_argument("--output-path", default=None)
    dl_fixed_split_parser = subparsers.add_parser("dl-fixed-split-run")
    dl_fixed_split_parser.add_argument("--markets", default="commodities,etfs,stocks")
    dl_fixed_split_parser.add_argument("--models", default="torch_cnn,torch_gru,torch_transformer,torch_transformer_gru")
    dl_fixed_split_parser.add_argument("--symbols", default="")
    dl_fixed_split_parser.add_argument("--timeframe", default="day")
    dl_fixed_split_parser.add_argument("--regular-hours-only", action="store_true")
    dl_fixed_split_parser.add_argument("--no-same-day-reentry", action="store_true")
    dl_fixed_split_parser.add_argument("--train-bars", type=int, default=3528)
    dl_fixed_split_parser.add_argument("--test-bars", type=int, default=252)
    dl_fixed_split_parser.add_argument("--stop-loss-pct", type=float, default=None)
    dl_fixed_split_parser.add_argument("--trailing-stop-pct", type=float, default=0.015)
    research_comparison_pdf_parser = subparsers.add_parser("research-comparison-pdf")
    research_comparison_pdf_parser.add_argument("--output-path", default=None)
    research_comparison_pdf_parser.add_argument("--dl-no-rule-artifact", default=None)
    historical_chart_parser = subparsers.add_parser("historical-chart")
    historical_chart_parser.add_argument("--market", required=True)
    historical_chart_parser.add_argument("--symbol", required=True)
    historical_chart_parser.add_argument("--strategy-name", required=True)
    historical_chart_parser.add_argument("--window", default="3y")
    historical_chart_parser.add_argument("--timeframe", default="day")
    historical_chart_parser.add_argument("--regular-hours-only", action="store_true")
    historical_chart_parser.add_argument("--no-same-day-reentry", action="store_true")
    historical_chart_parser.add_argument("--output-path", default=None)
    historical_chart_interactive_parser = subparsers.add_parser("historical-chart-interactive")
    historical_chart_interactive_parser.add_argument("--market", required=True)
    historical_chart_interactive_parser.add_argument("--symbol", required=True)
    historical_chart_interactive_parser.add_argument("--strategy-name", required=True)
    historical_chart_interactive_parser.add_argument("--window", default="3y")
    historical_chart_interactive_parser.add_argument("--timeframe", default="day")
    historical_chart_interactive_parser.add_argument("--regular-hours-only", action="store_true")
    historical_chart_interactive_parser.add_argument("--no-same-day-reentry", action="store_true")
    historical_chart_interactive_parser.add_argument("--output-path", default=None)
    historical_cost_aware_run_parser = subparsers.add_parser("historical-cost-aware-run")
    historical_cost_aware_run_parser.add_argument("--market", required=True)
    historical_cost_aware_run_parser.add_argument("--symbol", required=True)
    historical_cost_aware_run_parser.add_argument("--strategy-name", required=True)
    historical_cost_aware_run_parser.add_argument("--window", default="3y")
    historical_cost_aware_run_parser.add_argument("--timeframe", default="day")
    historical_cost_aware_run_parser.add_argument("--stop-loss-pct", type=float, default=None)
    historical_cost_aware_run_parser.add_argument("--trailing-stop-pct", type=float, default=None)
    historical_cost_aware_run_parser.add_argument("--starting-capital", type=float, default=100000.0)
    historical_cost_aware_run_parser.add_argument("--commission-per-order", type=float, default=0.0)
    historical_cost_aware_run_parser.add_argument("--quoted-spread-bps", type=float, default=1.0)
    historical_cost_aware_run_parser.add_argument("--market-impact-bps", type=float, default=1.0)
    historical_cost_aware_run_parser.add_argument("--stop-extra-slippage-bps", type=float, default=2.0)
    historical_cost_aware_run_parser.add_argument("--max-bar-participation-rate", type=float, default=0.0005)
    historical_cost_aware_run_parser.add_argument("--max-bar-shares", type=float, default=None)
    historical_cost_aware_run_parser.add_argument("--sec-fee-per-million-sell", type=float, default=0.0)
    historical_cost_aware_run_parser.add_argument("--finra-taf-per-share-sell", type=float, default=0.000195)
    historical_cost_aware_run_parser.add_argument("--finra-taf-cap-per-trade", type=float, default=9.79)
    historical_cost_aware_run_parser.add_argument("--regular-hours-only", action="store_true")
    historical_cost_aware_run_parser.add_argument("--no-same-day-reentry", action="store_true")
    xle_daily_parser = subparsers.add_parser("xle-daily-cycle")
    xle_daily_parser.add_argument("--symbol", default="XLE")
    xle_daily_parser.add_argument("--bars", type=int, default=90)
    xle_daily_parser.add_argument("--submit", action="store_true")

    args = parser.parse_args()
    if args.command == "show-config":
        return cmd_show_config()
    if args.command == "healthcheck":
        return cmd_healthcheck()
    if args.command == "seed-sample":
        return cmd_seed_sample(args.symbol, args.price)
    if args.command == "questdb-count":
        return cmd_questdb_count()
    if args.command == "ingest-bars":
        context = build_app_context()
        symbols = context.settings.resolve_symbols(args.symbols)
        return cmd_ingest_bars(symbols, args.max_events)
    if args.command == "shadow-run":
        context = build_app_context()
        symbols = context.settings.resolve_symbols(args.symbols)
        return cmd_shadow_run(symbols, args.max_events)
    if args.command == "paper-account":
        return cmd_paper_account()
    if args.command == "paper-rebalance":
        context = build_app_context()
        symbols = context.settings.resolve_symbols(args.symbols)
        return cmd_paper_rebalance(symbols, args.submit)
    if args.command == "basket-daily-cycle":
        context = build_app_context()
        symbols = context.settings.resolve_symbols(args.symbols)
        return cmd_basket_daily_cycle(symbols, args.submit)
    if args.command == "basket-report":
        return cmd_basket_report(args.limit)
    if args.command == "paper-sync":
        return cmd_paper_sync(args.days)
    if args.command == "xle-compare":
        return cmd_xle_compare(args.symbol, args.bars)
    if args.command == "xle-report":
        return cmd_xle_report(args.symbol, args.limit)
    if args.command == "xle-run-best":
        return cmd_xle_run_best(args.symbol, args.bars, args.submit)
    if args.command == "xle-parallel-run":
        return cmd_xle_parallel_run(args.symbol, args.bars)
    if args.command == "xle-parallel-report":
        return cmd_xle_parallel_report(args.symbol)
    if args.command == "xle-parallel-history":
        return cmd_xle_parallel_history(args.symbol, args.strategy_name, args.limit)
    if args.command == "research-parallel-run":
        return cmd_research_parallel_run(args.market, args.bars)
    if args.command == "research-parallel-report":
        return cmd_research_parallel_report(args.market)
    if args.command == "research-parallel-history":
        return cmd_research_parallel_history(args.symbol, args.strategy_name, args.limit)
    if args.command == "promote-candidate":
        return cmd_promote_candidate(args.market, args.symbol, args.strategy_name, args.notional)
    if args.command == "promoted-report":
        return cmd_promoted_report()
    if args.command == "promoted-run":
        return cmd_promoted_run(args.submit)
    if args.command == "historical-research-run":
        return cmd_historical_research_run(
            [value for value in args.markets.split(",") if value.strip()],
            [value for value in args.windows.split(",") if value.strip()],
            args.timeframe,
            args.regular_hours_only,
            args.execution_aware,
            args.no_same_day_reentry,
            args.stop_loss_pct,
            args.trailing_stop_pct,
        )
    if args.command == "historical-research-report":
        return cmd_historical_research_report(args.limit, args.execution_aware)
    if args.command == "ml-research-run":
        return cmd_ml_research_run(
            [value for value in args.markets.split(",") if value.strip()],
            [value for value in args.windows.split(",") if value.strip()],
            [value for value in args.models.split(",") if value.strip()],
            args.timeframe,
            args.regular_hours_only,
            args.no_same_day_reentry,
            args.stop_loss_pct,
            args.trailing_stop_pct,
        )
    if args.command == "ml-research-report":
        return cmd_ml_research_report(args.limit)
    if args.command == "dl-research-run":
        return cmd_dl_research_run(
            [value for value in args.markets.split(",") if value.strip()],
            [value for value in args.windows.split(",") if value.strip()],
            [value for value in args.models.split(",") if value.strip()],
            args.timeframe,
            args.regular_hours_only,
            args.no_same_day_reentry,
            args.stop_loss_pct,
            args.trailing_stop_pct,
        )
    if args.command == "dl-research-report":
        return cmd_dl_research_report(args.limit)
    if args.command == "llm-news-research-run":
        return cmd_llm_research_run(
            [value for value in args.windows.split(",") if value.strip()],
            [value.strip().lower() for value in args.markets.split(",") if value.strip()],
            [value.strip().upper() for value in args.commodity_symbols.split(",") if value.strip()],
            [value.strip().upper() for value in args.stock_symbols.split(",") if value.strip()],
            [value.strip().upper() for value in args.fx_symbols.split(",") if value.strip()],
            args.stocks_only,
            args.fx_only,
            args.model_name,
            args.timeframe,
            args.regular_hours_only,
            args.stop_loss_pct,
            args.trailing_stop_pct,
        )
    if args.command == "llm-news-research-report":
        return cmd_llm_research_report(args.limit)
    if args.command == "deepseek-news-meta-run":
        return cmd_deepseek_news_meta_run(
            [value.strip().upper() for value in args.training_symbols.split(",") if value.strip()],
            [value.strip().upper() for value in args.evaluation_symbols.split(",") if value.strip()],
            args.base_model_path,
            args.output_root,
            args.max_bars,
        )
    if args.command == "deepseek-directional-run":
        return cmd_deepseek_directional_run(
            [value.strip().upper() for value in args.training_symbols.split(",") if value.strip()],
            [value.strip().upper() for value in args.evaluation_symbols.split(",") if value.strip()],
            args.base_model_path,
            args.output_root,
            args.max_bars,
            args.num_train_epochs,
            args.max_train_samples,
            args.max_eval_samples,
            args.include_base_model,
            args.scoring_batch_size,
        )
    if args.command == "deepseek-directional-crypto-run":
        return cmd_deepseek_directional_crypto_run(
            [value.strip().upper() for value in args.training_symbols.split(",") if value.strip()],
            [value.strip().upper() for value in args.evaluation_symbols.split(",") if value.strip()],
            args.base_model_path,
            args.output_root,
            args.max_bars,
            args.num_train_epochs,
            args.max_train_samples,
            args.max_eval_samples,
            args.scoring_batch_size,
        )
    if args.command == "deepseek-directional-v4-run":
        return cmd_deepseek_directional_v4_run(
            args.output_root,
            args.base_model_path,
            args.max_bars,
            args.neutral_band,
        )
    if args.command == "deepseek-directional-v35-hourly-run":
        return cmd_deepseek_directional_v35_hourly_run(
            [value.strip().upper() for value in args.training_symbols.split(",") if value.strip()],
            [value.strip().upper() for value in args.evaluation_symbols.split(",") if value.strip()],
            args.base_model_path,
            args.output_root,
            args.max_bars,
            args.num_train_epochs,
            args.max_train_samples,
            args.max_eval_samples,
            args.scoring_batch_size,
        )
    if args.command == "deepseek-directional-v35-variant-run":
        return cmd_deepseek_directional_v35_variant_run()
    if args.command == "qwen-embedding-v3-q3e-run":
        return cmd_qwen_embedding_v3_q3e_run(
            args.output_root,
            args.model_name,
            args.max_bars,
            args.max_train_samples,
            args.max_length,
            args.batch_size,
            args.quantization,
            args.embargo_days,
            args.v35_scores_path,
            args.end_date,
            [value.strip().upper() for value in args.training_symbols.split(",") if value.strip()],
            [value.strip().upper() for value in args.evaluation_symbols.split(",") if value.strip()],
        )
    if args.command == "hybrid-alpha-run":
        return cmd_hybrid_alpha_run(args.scores_path, args.output_root, args.max_bars, args.eval_days_count, args.retrain_every)
    if args.command == "factor-ml-run":
        return cmd_factor_ml_run(args.output_root, args.max_bars, args.eval_days_count, args.retrain_every, args.forward_horizon)
    if args.command == "factor-validation-run":
        return cmd_factor_validation_run(
            args.output_root,
            args.max_bars,
            args.window_days,
            args.window_count,
            args.forward_horizon,
        )
    if args.command == "factor-news-overlay-run":
        return cmd_factor_news_overlay_run(
            args.scores_path,
            args.output_root,
            args.max_bars,
            args.window_days,
            args.window_count,
            args.forward_horizon,
            args.news_fresh_days,
        )
    if args.command == "sentiment-balanced-run":
        return cmd_sentiment_balanced_run(
            args.scores_path,
            args.output_root,
            args.max_bars,
            args.window_days,
            args.window_count,
            args.news_fresh_days,
            args.retrain_every,
        )
    if args.command == "embedding-sentiment-run":
        return cmd_embedding_sentiment_run(
            args.scores_path,
            args.embedding_cache_path,
            args.output_root,
            args.max_bars,
            args.window_days,
            args.window_count,
            args.news_fresh_days,
            args.retrain_every,
            args.svd_components,
            args.book_size,
        )
    if args.command == "portfolio-controls-run":
        return cmd_portfolio_controls_run(
            args.output_root,
            args.max_bars,
            args.window_days,
            args.window_count,
            args.forward_horizon,
            args.book_size,
            args.rebalance_days,
        )
    if args.command == "regime-switch-run":
        return cmd_regime_switch_run(
            args.output_root,
            args.max_bars,
            args.window_days,
            args.window_count,
            args.forward_horizon,
            args.book_size,
            args.rebalance_days,
            args.train_min_samples,
            args.embargo_days,
            args.label_drawdown_penalty,
        )
    if args.command == "book-size-sensitivity-run":
        return cmd_book_size_sensitivity_run(
            args.output_root,
            args.max_bars,
            args.window_days,
            args.window_count,
            args.forward_horizon,
            [int(value.strip()) for value in args.book_sizes.split(",") if value.strip()],
            args.rebalance_days,
        )
    if args.command == "price-momentum-validation-run":
        return cmd_price_momentum_validation_run(
            args.output_root,
            args.max_bars,
            args.window_days,
            args.window_count,
            args.forward_horizon,
            args.book_size,
            args.rebalance_days,
        )
    if args.command == "cross-asset-momentum-run":
        return cmd_cross_asset_momentum_run(
            args.output_root,
            args.max_bars,
            args.lookback_days,
            args.rebalance_days,
            args.window_days,
            args.window_count,
            [int(value.strip()) for value in args.book_sizes.split(",") if value.strip()],
        )
    if args.command == "vol-carry-research":
        return cmd_vol_carry_research(
            args.output_root,
            args.window_days,
            args.window_count,
            args.portfolio_cap,
        )
    if args.command == "event-premia-research":
        return cmd_event_premia_research(
            args.output_root,
            args.start,
            args.panel_symbols,
            not args.no_panel,
        )
    if args.command == "alpha-factory-run":
        return cmd_alpha_factory_run(
            args.output_root,
            args.start,
            args.search_budget,
            args.seed,
            args.llm_model_path,
            args.llm_proposals,
        )
    if args.command == "long-short-corrected-run":
        return cmd_long_short_corrected_run(args.scores_path, args.output_root, args.eval_days_count, args.book_size)
    if args.command == "cross-sectional-retrain-run":
        return cmd_cross_sectional_retrain_run(
            args.output_root,
            args.base_model_path,
            args.max_bars,
            args.eval_days_count,
            args.num_train_epochs,
            args.max_train_samples,
        )
    if args.command == "long-short-research-run":
        return cmd_long_short_research_run(
            args.output_root,
            args.scores_path,
            args.max_bars,
            args.eval_days_count,
            args.book_size,
            args.rebalance_days,
            args.cost_bps_per_side,
            args.short_borrow_annual,
            args.book_mode,
        )
    if args.command == "v35-execution-controls-run":
        return cmd_v35_execution_controls_run(
            args.output_root,
            args.v35_scores_path,
            args.end_date,
            args.seeds,
            [value.strip().upper() for value in args.evaluation_symbols.split(",") if value.strip()],
        )
    if args.command == "v35-paper-run":
        return cmd_v35_paper_run(args.submit, args.target_capital)
    if args.command == "v35-paper-report":
        return cmd_v35_paper_report()
    if args.command == "v35-intraday-run":
        return cmd_v35_intraday_run(
            args.submit,
            args.target_capital,
            args.lookback_bars,
            args.max_cycles,
            args.sleep_seconds,
            args.news_check_seconds,
        )
    if args.command == "deepseek-strict-compare-run":
        return cmd_deepseek_strict_compare_run(
            args.previous_report_path,
            args.meta_artifact_path,
            args.output_path,
            args.max_bars,
        )
    if args.command == "deepseek-strict-compare-pdf":
        return cmd_deepseek_strict_compare_pdf(args.comparison_artifact, args.output_path)
    if args.command == "llm-overlay-pdf":
        return cmd_llm_overlay_pdf(args.hourly_artifact, args.output_path)
    if args.command == "dl-fixed-split-run":
        return cmd_dl_fixed_split_run(
            [value for value in args.markets.split(",") if value.strip()],
            [value for value in args.models.split(",") if value.strip()],
            [value.strip().upper() for value in args.symbols.split(",") if value.strip()],
            args.timeframe,
            args.regular_hours_only,
            args.no_same_day_reentry,
            args.train_bars,
            args.test_bars,
            args.stop_loss_pct,
            args.trailing_stop_pct,
        )
    if args.command == "research-comparison-pdf":
        return cmd_research_comparison_pdf(args.output_path, args.dl_no_rule_artifact)
    if args.command == "historical-chart":
        return cmd_historical_chart(
            args.market,
            args.symbol,
            args.strategy_name,
            args.window,
            args.output_path,
            args.timeframe,
            args.regular_hours_only,
            args.no_same_day_reentry,
        )
    if args.command == "historical-chart-interactive":
        return cmd_historical_chart_interactive(
            args.market,
            args.symbol,
            args.strategy_name,
            args.window,
            args.output_path,
            args.timeframe,
            args.regular_hours_only,
            args.no_same_day_reentry,
        )
    if args.command == "historical-cost-aware-run":
        return cmd_historical_cost_aware_run(
            args.market,
            args.symbol,
            args.strategy_name,
            args.window,
            args.timeframe,
            args.stop_loss_pct,
            args.trailing_stop_pct,
            args.starting_capital,
            args.commission_per_order,
            args.quoted_spread_bps,
            args.market_impact_bps,
            args.stop_extra_slippage_bps,
            args.max_bar_participation_rate,
            args.max_bar_shares,
            args.sec_fee_per_million_sell,
            args.finra_taf_per_share_sell,
            args.finra_taf_cap_per_trade,
            args.regular_hours_only,
            args.no_same_day_reentry,
        )
    if args.command == "xle-daily-cycle":
        return cmd_xle_daily_cycle(args.symbol, args.bars, args.submit)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
