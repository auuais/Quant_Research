from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from statistics import pstdev
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from algoding.data.historical import AlpacaHistoricalClient
from algoding.data.news import AlpacaNewsHistoricalClient
from algoding.execution.deepseek_directional_research import DeepseekDirectionalResearchLab
from algoding.execution.llm_research import LlmNewsResearchLab
from algoding.portfolio.risk import RiskLimits
from algoding.research.cost_aware_backtest import run_execution_aware_replay
from algoding.research.deepseek_directional_model import (
    DirectionalPrediction,
    DirectionalTradeDefinition,
)
from algoding.research.historical_backtest import BacktestBar
from algoding.research.trade_execution import NO_TRAIL, build_close_execution_trace
from algoding.research.news_directional_labeler import DirectionalNewsExample, build_directional_examples
from algoding.research.news_labeler import BundleMetadata, build_bundle_metadata
from algoding.settings import Settings


TRAINING_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "AMD", "NFLX",
    "JPM", "XOM", "ORCL", "CRM", "WMT", "COST", "UNH", "V", "MA", "HD",
    "ABBV", "MRK", "PG", "PEP", "KO", "CVX", "BAC", "CSCO", "IBM", "QCOM",
    "GE", "PLTR", "ADBE", "INTC",
]

TOP5_SYMBOLS = ["AVGO", "NVDA", "TSLA", "GOOGL", "XOM"]
CLASS_ORDER = ["bearish", "neutral", "bullish"]
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_ORDER)}
INDEX_TO_CLASS = {index: label for label, index in CLASS_TO_INDEX.items()}

VARIANTS = ["price_only_baseline", "q3e_text_only", "q3e_fused", "q3e_sector"]
PRIMARY_THRESHOLD = "medium"
SIGNAL_THRESHOLDS = ["medium", "high"]
LABEL_HORIZON_BARS = 5

EVENT_TYPES = ["earnings", "guidance", "analyst", "mna", "regulatory", "litigation", "product", "macro", "other"]
SESSION_TYPES = ["pre_market", "intraday", "post_close", "mixed"]

SECTOR_UNIVERSES = {
    "semis": ["AVGO", "NVDA", "AMD", "QCOM", "INTC"],
    "growth": ["GOOGL", "TSLA", "META", "AMZN", "NFLX", "AAPL", "MSFT"],
    "energy": ["XOM", "CVX"],
    "financials": ["JPM", "BAC", "V", "MA"],
    "defensive": ["WMT", "COST", "PG", "PEP", "KO", "UNH", "MRK", "ABBV"],
    "software": ["ORCL", "CRM", "ADBE", "PLTR", "IBM", "CSCO"],
    "industrial": ["GE"],
}
SYMBOL_TO_SECTOR = {symbol: sector for sector, symbols in SECTOR_UNIVERSES.items() for symbol in symbols}


@dataclass(frozen=True)
class Q3ERow:
    symbol: str
    trading_day: str
    split: str
    sector: str
    text: str
    direction: str
    strength: str
    relative_to_spy: str
    article_count: int
    event_type: str
    session_label: str
    forward_return_5d: float
    price_features: tuple
    sentiment_features: tuple


class Qwen3EmbeddingEncoder:
    """Frozen Qwen3-Embedding encoder with a stable on-disk embedding cache.

    The cache key is a SHA256 over the fully formatted input text plus the
    identifying metadata, so it is stable across processes (unlike Python ``hash``).
    """

    def __init__(
        self,
        *,
        model_name: str,
        cache_path: Path,
        max_length: int,
        batch_size: int,
        quantization: str,
    ) -> None:
        self._model_name = model_name
        self._cache_path = cache_path
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_length = max_length
        self._batch_size = batch_size
        self._quantization = quantization
        self._cache = self._load_cache()
        self._embedding_dim: Optional[int] = None
        if not torch.cuda.is_available():
            raise RuntimeError("V3-Q3E requires CUDA for Qwen3-Embedding-4B inference.")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True,
        )
        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True, "device_map": {"": 0}}
        if quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["torch_dtype"] = torch.float16
        self._model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self._model.eval()

    @property
    def settings(self) -> dict:
        return {
            "model_name": self._model_name,
            "max_length": self._max_length,
            "batch_size": self._batch_size,
            "quantization": self._quantization,
        }

    def relax_for_oom(self) -> None:
        """Drop to the conservative memory profile requested for GPU OOM fallback."""
        self._batch_size = 1
        self._max_length = min(self._max_length, 512)

    def encode_rows(self, rows: list) -> np.ndarray:
        missing = [row for row in rows if self._cache_key(row) not in self._cache]
        index = 0
        while index < len(missing):
            batch = missing[index : index + self._batch_size]
            try:
                encoded = self._encode_texts([self._format_text(row) for row in batch])
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:  # pragma: no cover - GPU dependent
                at_floor = self._batch_size == 1 and self._max_length <= 512
                if "out of memory" not in str(exc).lower() or at_floor:
                    raise
                torch.cuda.empty_cache()
                self.relax_for_oom()
                continue
            for row, vector in zip(batch, encoded):
                key = self._cache_key(row)
                self._cache[key] = vector.astype(np.float32)
                self._append_cache(key, row, vector)
            index += len(batch)
        if rows and self._embedding_dim is None:
            self._embedding_dim = int(self._cache[self._cache_key(rows[0])].shape[0])
        vectors = [self._cache[self._cache_key(row)] for row in rows]
        return np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, self.embedding_dim), dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim or 0

    def _encode_texts(self, texts: list) -> np.ndarray:
        batch = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        ).to(self._model.device)
        with torch.inference_mode():
            outputs = self._model(**batch)
            embeddings = _last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().float().numpy()

    def _format_text(self, row) -> str:
        task = (
            "Given stock-specific news headlines, classify the stock's next five trading-day "
            "direction as bullish, neutral, or bearish."
        )
        return (
            f"Instruct: {task}\n"
            f"Query: Symbol: {row.symbol}\n"
            f"Trading day: {row.trading_day}\n"
            f"Sector: {row.sector}\n"
            f"Session: {row.session_label}\n"
            f"Event type: {row.event_type}\n"
            f"Headline count: {row.article_count}\n"
            f"Headlines:\n{row.text}"
        )

    def _load_cache(self) -> dict:
        if not self._cache_path.exists():
            return {}
        cache = {}
        for line in self._cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            cache[str(payload["key"])] = np.array(payload["embedding"], dtype=np.float32)
        return cache

    def _append_cache(self, key: str, row, vector: np.ndarray) -> None:
        payload = {
            "key": key,
            "symbol": row.symbol,
            "trading_day": row.trading_day,
            "embedding": [round(float(value), 8) for value in vector.tolist()],
        }
        with self._cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _cache_key(self, row) -> str:
        digest = sha256(self._format_text(row).encode("utf-8")).hexdigest()
        return f"{self._model_name}|{self._max_length}|{row.symbol}|{row.trading_day}|{row.article_count}|{digest}"


class QwenEmbeddingDirectionalResearchLab:
    def __init__(self, settings: Settings) -> None:
        if not settings.has_alpaca_credentials:
            raise RuntimeError("Alpaca credentials are required in .env for V3-Q3E research.")
        self._settings = settings
        self._bars_client = AlpacaHistoricalClient(settings)
        self._news_client = AlpacaNewsHistoricalClient(settings)
        # New default policy (Item 1): loose exit (5% fixed stop, NO trailing) executed market-on-close.
        # This is the only Alpaca-executable, net-positive configuration we measured.
        self._risk_limits = RiskLimits.from_file().with_overrides(stop_loss_pct=0.05, trailing_stop_pct=NO_TRAIL)
        self._assumptions = replace(
            LlmNewsResearchLab._execution_assumptions(market="stocks", timeframe="day", regular_hours_only=False),
            quoted_spread_bps=0.5,
            market_impact_bps=0.5,
            stop_extra_slippage_bps=0.0,
            sec_fee_per_million_sell=27.8,
        )
        self._conviction_sizing = True

    def run(
        self,
        *,
        output_root: str = "reports/research/qwen_embedding_v3_q3e",
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        max_bars: int = 1400,
        max_train_samples: int = 10000,
        max_length: int = 768,
        batch_size: int = 4,
        quantization: str = "4bit",
        embargo_days: int = LABEL_HORIZON_BARS,
        v35_scores_path: str = "reports/research/deepseek_directional_v3/finetuned_directional_scores.jsonl",
        end_date: Optional[str] = None,
        volume_scale: float = 30.0,
        conviction_sizing: bool = True,
        # calibrate=False by default: sigmoid calibration compressed confidence below the strength
        # threshold so the model stopped entering (all-neutral). Kept as an opt-in.
        calibrate: bool = False,
        # triple_barrier with tight +/- vol barriers collapsed "neutral" and degraded macro F1
        # (classifiers went near-degenerate); band is the known-good default. triple_barrier stays
        # available for future tuning (wider barriers to preserve neutral).
        label_mode: str = "band",
        training_symbols: Optional[list] = None,
        evaluation_symbols: Optional[list] = None,
    ) -> dict:
        output_dir = Path(output_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._conviction_sizing = conviction_sizing
        train_universe = training_symbols or TRAINING_SYMBOLS
        eval_universe = evaluation_symbols or TOP5_SYMBOLS
        symbols = sorted({*train_universe, *eval_universe, "SPY"})

        # Load the V3-5 score cache first: its last day caps the evaluation window so the
        # baseline and V3-Q3E share the EXACT same test window (no mismatched/stale windows).
        sentiment_cache = _load_sentiment_cache(Path(v35_scores_path))
        if end_date is None:
            end_date = _cache_max_day(sentiment_cache)

        bars_by_symbol = self._load_bars(symbols=symbols, max_bars=max_bars, volume_scale=volume_scale)
        if end_date is not None:
            bars_by_symbol = {
                symbol: [bar for bar in bars if _bar_day(bar) <= end_date]
                for symbol, bars in bars_by_symbol.items()
            }
        base_split = DeepseekDirectionalResearchLab._build_date_splits(bars_by_symbol["SPY"])
        split_by_day = _apply_embargo(base_split, embargo_days=embargo_days)

        rows = self._build_rows(
            symbols=[symbol for symbol in symbols if symbol != "SPY"],
            bars_by_symbol=bars_by_symbol,
            benchmark_bars=bars_by_symbol["SPY"],
            split_by_day=split_by_day,
            sentiment_cache=sentiment_cache,
            label_mode=label_mode,
        )

        train_rows_all = [row for row in rows if row.split == "train"]
        train_rows = _balanced_limit(train_rows_all, limit=max_train_samples)
        sentiment_coverage = _sentiment_coverage(rows)
        use_sentiment = sentiment_coverage["train"] >= 0.2

        encoder = self._build_encoder(
            output_dir=output_dir,
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            quantization=quantization,
        )

        # Precompute embeddings once for every row group that needs them.
        eval_rows_by_symbol = {
            symbol: {
                "validation": [r for r in rows if r.symbol == symbol and r.split == "validation"],
                "test": [r for r in rows if r.symbol == symbol and r.split == "test"],
            }
            for symbol in eval_universe
        }
        train_embeddings = encoder.encode_rows(train_rows)
        embeddings_by_symbol = {
            symbol: {
                "validation": encoder.encode_rows(groups["validation"]),
                "test": encoder.encode_rows(groups["test"]),
            }
            for symbol, groups in eval_rows_by_symbol.items()
        }

        train_targets = np.array([CLASS_TO_INDEX[row.direction] for row in train_rows], dtype=np.int64)
        models = self._fit_all_models(train_rows, train_embeddings, train_targets, use_sentiment=use_sentiment, calibrate=calibrate)

        variant_results = {}
        for variant in VARIANTS:
            variant_results[variant] = self._evaluate_variant(
                variant=variant,
                models=models,
                eval_rows_by_symbol=eval_rows_by_symbol,
                embeddings_by_symbol=embeddings_by_symbol,
                bars_by_symbol=bars_by_symbol,
                split_by_day=split_by_day,
                use_sentiment=use_sentiment,
            )

        v35_baseline = self._evaluate_v35_baseline(
            eval_rows_by_symbol=eval_rows_by_symbol,
            bars_by_symbol=bars_by_symbol,
            split_by_day=split_by_day,
            sentiment_cache=sentiment_cache,
        )

        comparison = _build_comparison(variant_results, v35_baseline)
        best_variant = _select_best_variant(variant_results, v35_baseline)
        charts = _write_charts(output_dir, variant_results, v35_baseline)

        result = {
            "version": "V3-Q3E",
            "model_name": model_name,
            "model_role": "frozen Qwen3 embedding encoder + downstream logistic directional classifiers",
            "cuda_used": True,
            "encoder_settings": encoder.settings,
            "training_symbols": train_universe,
            "evaluation_symbols": eval_universe,
            "max_bars": max_bars,
            "evaluation_window_end": end_date,
            "evaluation_window_note": (
                "Window capped at the last day present in the V3-5 score cache so the baseline "
                "and V3-Q3E are scored on the identical test window."
            ),
            "label": {
                "horizon_bars": LABEL_HORIZON_BARS,
                "classes": CLASS_ORDER,
                "label_mode": label_mode,
                "definition": (
                    "triple-barrier: first of +/- vol-scaled barrier touched within the horizon (else neutral)"
                    if label_mode == "triple_barrier"
                    else "5-day forward close-to-close return vs volatility-scaled neutral band"
                ),
            },
            "leakage_controls": {
                "embargo_days": embargo_days,
                "purge_applied": "last <embargo_days> train and validation days dropped (label horizon overlap)",
                "no_same_day_price_leakage": "features and entry both use close[T]; no close[T+k] in features",
                "news_timing": "bundle[T] uses articles before the 16:00 ET close of day T (news_labeler)",
            },
            "date_splits": _split_summary(split_by_day),
            "dataset_sizes": {
                "rows_total": len(rows),
                "train_rows_available": len(train_rows_all),
                "train_rows_used": len(train_rows),
                "train_class_distribution": dict(Counter(row.direction for row in train_rows)),
                "validation_rows_eval": sum(len(g["validation"]) for g in eval_rows_by_symbol.values()),
                "test_rows_eval": sum(len(g["test"]) for g in eval_rows_by_symbol.values()),
            },
            "training_config": {
                "max_train_samples": max_train_samples,
                "balanced_sampling": True,
                "classifier": "StandardScaler + LogisticRegression(class_weight=balanced)",
                "probability_calibration": calibrate,
                "sentiment_features_used": use_sentiment,
                "sentiment_coverage": sentiment_coverage,
            },
            "trade_policy": {
                "exit": "loose: 5% fixed stop, no trailing, exit when model no longer bullish",
                "stop_loss_pct": 0.05,
                "trailing_stop_pct": None,
                "execution": "market_on_close (decisions and fills at the close; no intraday stop)",
                "require_relative_confirmation": False,
                "require_momentum_confirmation": False,
                "signal_thresholds": SIGNAL_THRESHOLDS,
                "primary_threshold": PRIMARY_THRESHOLD,
                "conviction_sizing": conviction_sizing,
                "execution_assumptions": _assumptions_dict(self._assumptions),
                "note": "Item 1 re-baseline: replaced the 1% intraday trailing stop (stop-market) "
                        "with a loose MOC-executed exit — the only Alpaca-executable net-positive config.",
            },
            "variant_results": variant_results,
            "v35_baseline": v35_baseline,
            "comparison_vs_v35": comparison,
            "best_variant": best_variant,
            "charts": charts,
        }

        result = _jsonable(result)
        (output_dir / "v3_q3e_run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (output_dir / "v3_q3e_summary.md").write_text(_summary_markdown(result), encoding="utf-8")
        return result

    # ------------------------------------------------------------------ data

    def _build_encoder(
        self, *, output_dir, model_name, max_length, batch_size, quantization
    ) -> Qwen3EmbeddingEncoder:
        # Model load happens in the constructor (fails fast on load issues); per-batch GPU OOM
        # is handled inside ``encode_rows`` which relaxes to batch_size=1, max_length=512.
        return Qwen3EmbeddingEncoder(
            model_name=model_name,
            cache_path=output_dir / "qwen3_embedding_cache.jsonl",
            max_length=max_length,
            batch_size=batch_size,
            quantization=quantization,
        )

    def _load_bars(self, *, symbols, max_bars, volume_scale=1.0) -> dict:
        # volume_scale undoes the IEX feed's ~30x volume under-reporting so the cost model's
        # participation cap reflects real liquidity. Volume features are z-scored (scale-invariant)
        # and the close-execution gross ignores volume, so scaling only affects the net replay cap.
        stock_bars = self._bars_client.get_recent_bars(symbols, max_bars, timeframe="day")
        return {
            symbol: [
                BacktestBar(
                    timestamp=bar.timestamp.isoformat(),
                    open=float(bar.open),
                    close=float(bar.close),
                    high=float(bar.high),
                    low=float(bar.low),
                    volume=float(getattr(bar, "volume", 0.0) or 0.0) * volume_scale,
                )
                for bar in values
            ]
            for symbol, values in stock_bars.items()
        }

    def _load_news(self, *, symbol, bars) -> list:
        start = datetime.fromisoformat(bars[0].timestamp).astimezone(timezone.utc) - timedelta(days=3)
        end = datetime.fromisoformat(bars[-1].timestamp).astimezone(timezone.utc) + timedelta(days=1)
        return self._news_client.get_news_articles(symbol=symbol, start=start, end=end, include_content=False)

    def _build_rows(self, *, symbols, bars_by_symbol, benchmark_bars, split_by_day, sentiment_cache, label_mode="band") -> list:
        rows = []
        for symbol in symbols:
            bars = bars_by_symbol[symbol]
            articles = self._load_news(symbol=symbol, bars=bars)
            trading_days = [datetime.fromisoformat(bar.timestamp).date() for bar in bars]
            bundles = build_bundle_metadata(symbol=symbol, articles=articles, trading_days=trading_days)
            examples = build_directional_examples(
                symbol=symbol, bundles=bundles, bars=bars, benchmark_bars=benchmark_bars,
                horizon_bars=LABEL_HORIZON_BARS, label_mode=label_mode,
            )
            example_by_day = {example.trading_day: example for example in examples}
            bundle_by_day = {bundle.trading_day: bundle for bundle in bundles}

            days = [datetime.fromisoformat(bar.timestamp).date().isoformat() for bar in bars]
            closes = [float(bar.close) for bar in bars]
            volumes = [float(bar.volume or 0.0) for bar in bars]
            sector = SYMBOL_TO_SECTOR.get(symbol, "unknown")
            for index in range(20, len(bars) - LABEL_HORIZON_BARS):
                trading_day = days[index]
                split = split_by_day.get(trading_day)
                example = example_by_day.get(trading_day)
                bundle = bundle_by_day.get(trading_day)
                if split is None or split == "embargo" or example is None or bundle is None:
                    continue
                rows.append(
                    Q3ERow(
                        symbol=symbol,
                        trading_day=trading_day,
                        split=split,
                        sector=sector,
                        text=bundle.text,
                        direction=example.direction,
                        strength=example.strength,
                        relative_to_spy=example.relative_to_spy,
                        article_count=example.article_count,
                        event_type=example.event_type,
                        session_label=example.session_label,
                        forward_return_5d=float(example.forward_return_5d),
                        price_features=tuple(_price_features(closes=closes, volumes=volumes, index=index)),
                        sentiment_features=tuple(
                            _sentiment_features(sentiment_cache.get(symbol, {}).get(trading_day))
                        ),
                    )
                )
        return rows

    # --------------------------------------------------------------- modeling

    def _fit_all_models(self, train_rows, train_embeddings, train_targets, *, use_sentiment, calibrate=False) -> dict:
        numeric = _numeric_matrix(train_rows, use_sentiment=use_sentiment)
        pooled = {
            "price_only_baseline": _fit_classifier(numeric, train_targets, calibrate=calibrate),
            "q3e_text_only": _fit_classifier(train_embeddings, train_targets, calibrate=calibrate),
            "q3e_fused": _fit_classifier(np.hstack([train_embeddings, numeric]), train_targets, calibrate=calibrate),
        }
        sector_models = {}
        fused = np.hstack([train_embeddings, numeric])
        for sector in sorted({row.sector for row in train_rows}):
            mask = np.array([row.sector == sector for row in train_rows])
            if int(mask.sum()) < 60 or len({train_targets[i] for i in np.where(mask)[0]}) < 2:
                continue
            sector_models[sector] = _fit_classifier(fused[mask], train_targets[mask], calibrate=calibrate)
        pooled["q3e_sector"] = {"pooled": pooled["q3e_fused"], "sector_models": sector_models}
        return pooled

    def _variant_features(self, variant, rows, embeddings, *, use_sentiment) -> np.ndarray:
        if not rows:
            return np.zeros((0, 1), dtype=np.float32)
        numeric = _numeric_matrix(rows, use_sentiment=use_sentiment)
        if variant == "price_only_baseline":
            return numeric
        if variant == "q3e_text_only":
            return embeddings
        return np.hstack([embeddings, numeric])

    def _predict(self, variant, models, rows, embeddings, *, use_sentiment) -> dict:
        if not rows:
            return {}
        features = self._variant_features(variant, rows, embeddings, use_sentiment=use_sentiment)
        if variant == "q3e_sector":
            sector = rows[0].sector
            model = models["q3e_sector"]["sector_models"].get(sector, models["q3e_sector"]["pooled"])
        else:
            model = models[variant]
        labels = model.predict(features)
        probabilities = model.predict_proba(features)
        return _predictions_from_model(rows, labels, probabilities)

    def _evaluate_variant(
        self, *, variant, models, eval_rows_by_symbol, embeddings_by_symbol, bars_by_symbol, split_by_day, use_sentiment
    ) -> dict:
        per_symbol = []
        for symbol, groups in eval_rows_by_symbol.items():
            val_pred = self._predict(variant, models, groups["validation"], embeddings_by_symbol[symbol]["validation"], use_sentiment=use_sentiment)
            test_pred = self._predict(variant, models, groups["test"], embeddings_by_symbol[symbol]["test"], use_sentiment=use_sentiment)
            per_symbol.append(
                _per_symbol_record(
                    symbol=symbol,
                    sector=SYMBOL_TO_SECTOR.get(symbol, "unknown"),
                    validation_rows=groups["validation"],
                    test_rows=groups["test"],
                    validation_predictions=val_pred,
                    test_predictions=test_pred,
                    bars=bars_by_symbol[symbol],
                    split_by_day=split_by_day,
                    risk_limits=self._risk_limits,
                    assumptions=self._assumptions,
                    conviction_sizing=self._conviction_sizing,
                )
            )
        return {"per_symbol": per_symbol, "aggregate": _aggregate(per_symbol)}

    def _evaluate_v35_baseline(self, *, eval_rows_by_symbol, bars_by_symbol, split_by_day, sentiment_cache) -> dict:
        per_symbol = []
        for symbol, groups in eval_rows_by_symbol.items():
            cache = sentiment_cache.get(symbol, {})
            val_pred = {row.trading_day: cache[row.trading_day] for row in groups["validation"] if row.trading_day in cache}
            test_pred = {row.trading_day: cache[row.trading_day] for row in groups["test"] if row.trading_day in cache}
            record = _per_symbol_record(
                symbol=symbol,
                sector=SYMBOL_TO_SECTOR.get(symbol, "unknown"),
                validation_rows=groups["validation"],
                test_rows=groups["test"],
                validation_predictions=val_pred,
                test_predictions=test_pred,
                bars=bars_by_symbol[symbol],
                split_by_day=split_by_day,
                risk_limits=self._risk_limits,
                assumptions=self._assumptions,
                conviction_sizing=self._conviction_sizing,
            )
            test_days = len(groups["test"])
            record["score_coverage"] = round(len(test_pred) / test_days, 4) if test_days else 0.0
            per_symbol.append(record)
        return {
            "source": "DeepSeek V3 finetuned directional cache replayed on the identical test window",
            "per_symbol": per_symbol,
            "aggregate": _aggregate(per_symbol),
        }


# ---------------------------------------------------------------------- pooling


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


# --------------------------------------------------------------- feature utils


def _price_features(*, closes, volumes, index) -> list:
    close = closes[index]
    features = []
    for lookback in (1, 3, 5, 10, 20):
        previous = closes[index - lookback]
        features.append((close / previous) - 1.0 if previous > 0 else 0.0)
    returns = [
        (closes[idx] / closes[idx - 1]) - 1.0 if closes[idx - 1] > 0 else 0.0
        for idx in range(max(1, index - 19), index + 1)
    ]
    features.append(pstdev(returns) if len(returns) > 1 else 0.0)
    ma20 = sum(closes[index - 19 : index + 1]) / 20
    features.append((close / ma20) - 1.0 if ma20 > 0 else 0.0)
    high20 = max(closes[index - 19 : index + 1])
    features.append((close / high20) - 1.0 if high20 > 0 else 0.0)
    volume_window = volumes[index - 19 : index + 1]
    volume_mean = sum(volume_window) / len(volume_window)
    volume_std = pstdev(volume_window) if len(volume_window) > 1 else 0.0
    features.append((volumes[index] - volume_mean) / volume_std if volume_std > 0 else 0.0)
    return [float(value) if math.isfinite(value) else 0.0 for value in features]


def _sentiment_features(prediction) -> list:
    if prediction is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    direction = prediction.direction
    relative = prediction.relative_to_spy
    return [
        1.0 if direction == "bullish" else 0.0,
        1.0 if direction == "bearish" else 0.0,
        1.0 if direction == "neutral" else 0.0,
        float(prediction.confidence),
        1.0 if relative == "outperform" else 0.0,
        1.0 if relative == "underperform" else 0.0,
    ]


def _numeric_matrix(rows, *, use_sentiment) -> np.ndarray:
    matrix = []
    for row in rows:
        features = list(row.price_features)
        features.append(float(row.article_count))
        features.extend(1.0 if row.event_type == event else 0.0 for event in EVENT_TYPES)
        features.extend(1.0 if row.session_label == session else 0.0 for session in SESSION_TYPES)
        if use_sentiment:
            features.extend(row.sentiment_features)
        matrix.append(features)
    return np.array(matrix, dtype=np.float32)


def _fit_classifier(features: np.ndarray, targets: np.ndarray, *, calibrate: bool = True):
    base = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000)),
        ]
    )
    counts = np.bincount(targets) if len(targets) else np.array([0])
    min_per_class = counts[counts > 0].min() if (counts > 0).any() else 0
    if calibrate and min_per_class >= 6:
        try:
            model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
            model.fit(features, targets)
            return model
        except Exception:
            pass
    base.fit(features, targets)
    return base


def _predictions_from_model(rows, labels, probabilities) -> dict:
    predictions = {}
    for row, label, probs in zip(rows, labels, probabilities):
        direction = INDEX_TO_CLASS[int(label)]
        confidence = float(max(probs))
        strength = "high" if confidence >= 0.55 else ("medium" if confidence >= 0.40 else "low")
        predictions[row.trading_day] = DirectionalPrediction(
            symbol=row.symbol,
            trading_day=row.trading_day,
            direction=direction,
            strength=strength,
            relative_to_spy=_relative_from_direction(direction),
            event_type=row.event_type,
            session=row.session_label,
            raw_completion=json.dumps(
                {
                    "model": "V3-Q3E",
                    "direction": direction,
                    "confidence": round(confidence, 6),
                    "probabilities": {INDEX_TO_CLASS[i]: round(float(v), 6) for i, v in enumerate(probs)},
                }
            ),
        )
    return predictions


def _relative_from_direction(direction: str) -> str:
    if direction == "bullish":
        return "outperform"
    if direction == "bearish":
        return "underperform"
    return "inline"


# ------------------------------------------------------------------- splitting


def _apply_embargo(split_by_day: dict, *, embargo_days: int) -> dict:
    if embargo_days <= 0:
        return dict(split_by_day)
    ordered = sorted(split_by_day.items())
    train_days = [day for day, split in ordered if split == "train"]
    validation_days = [day for day, split in ordered if split == "validation"]
    embargoed = set(train_days[-embargo_days:]) | set(validation_days[-embargo_days:])
    return {day: ("embargo" if day in embargoed else split) for day, split in split_by_day.items()}


def _split_summary(split_by_day: dict) -> dict:
    by_split = defaultdict(list)
    for trading_day, split in sorted(split_by_day.items()):
        by_split[split].append(trading_day)
    return {
        split: f"{days[0]} -> {days[-1]} ({len(days)} days)"
        for split, days in by_split.items()
        if days
    }


def _balanced_limit(rows, *, limit: int) -> list:
    if len(rows) <= limit:
        return list(rows)
    by_class = defaultdict(list)
    for row in rows:
        by_class[row.direction].append(row)
    per_class = max(1, limit // max(1, len(by_class)))
    selected = []
    for label in CLASS_ORDER:
        selected.extend(by_class.get(label, [])[:per_class])
    if len(selected) < limit:
        chosen = {(row.symbol, row.trading_day) for row in selected}
        for row in rows:
            if (row.symbol, row.trading_day) not in chosen:
                selected.append(row)
            if len(selected) >= limit:
                break
    return sorted(selected[:limit], key=lambda row: (row.trading_day, row.symbol))


def _sentiment_coverage(rows) -> dict:
    coverage = {}
    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row.split == split]
        if not split_rows:
            coverage[split] = 0.0
            continue
        covered = sum(1 for row in split_rows if any(value != 0.0 for value in row.sentiment_features))
        coverage[split] = round(covered / len(split_rows), 4)
    return coverage


# -------------------------------------------------------------------- scoring


def _bar_day(bar) -> str:
    return datetime.fromisoformat(bar.timestamp).date().isoformat()


def _cache_max_day(sentiment_cache: dict) -> Optional[str]:
    days = [day for per_symbol in sentiment_cache.values() for day in per_symbol]
    return max(days) if days else None


def _load_sentiment_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    cache = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        symbol = str(payload.get("symbol", "")).upper()
        trading_day = str(payload.get("trading_day", ""))
        if not symbol or not trading_day:
            continue
        cache[symbol][trading_day] = DirectionalPrediction(
            symbol=symbol,
            trading_day=trading_day,
            direction=str(payload.get("direction", "neutral")),
            strength=str(payload.get("strength", "low")),
            relative_to_spy=str(payload.get("relative_to_spy", "inline")),
            event_type=str(payload.get("event_type", "other")),
            session=str(payload.get("session", "mixed")),
            raw_completion=str(payload.get("raw_completion", "")),
        )
    return dict(cache)


# ------------------------------------------------------------------- metrics


def _classification_metrics(rows, predictions) -> dict:
    actual, predicted = [], []
    for row in rows:
        prediction = predictions.get(row.trading_day)
        if prediction is None:
            continue
        actual.append(row.direction)
        predicted.append(prediction.direction)
    if not actual:
        return {
            "samples": 0,
            "direction_accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_f1": 0.0,
            "predicted_distribution": {},
            "actual_distribution": {},
            "confusion_matrix": [],
            "confusion_matrix_labels": CLASS_ORDER,
        }
    return {
        "samples": len(actual),
        "direction_accuracy": round(accuracy_score(actual, predicted), 6),
        "balanced_accuracy": round(balanced_accuracy_score(actual, predicted), 6),
        "macro_f1": round(f1_score(actual, predicted, labels=CLASS_ORDER, average="macro", zero_division=0), 6),
        "predicted_distribution": dict(Counter(predicted)),
        "actual_distribution": dict(Counter(actual)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=CLASS_ORDER).tolist(),
        "confusion_matrix_labels": CLASS_ORDER,
    }


def _trade_summaries(*, symbol, bars, split_by_day, predictions, risk_limits, assumptions, conviction_sizing=False) -> dict:
    test_bars = [
        bar for bar in bars
        if split_by_day.get(datetime.fromisoformat(bar.timestamp).date().isoformat()) == "test"
    ]
    summaries = {}
    for threshold in SIGNAL_THRESHOLDS:
        trace = build_close_execution_trace(
            bars=test_bars,
            predictions_by_day=predictions,
            definition=DirectionalTradeDefinition(
                name=f"{symbol.lower()}_v3_q3e_{threshold}",
                minimum_strength=threshold,
                require_relative_confirmation=False,
                require_momentum_confirmation=False,
            ),
            stop_loss_pct=risk_limits.stop_loss_pct,
            trailing_stop_pct=NO_TRAIL,
            conviction_sizing=conviction_sizing,
        )
        gross = dict(trace["summary"])
        net = run_execution_aware_replay(test_bars, trace, assumptions)["summary"]
        summaries[threshold] = {
            "gross_return": float(gross["total_return"]),
            "gross_max_drawdown": float(gross["max_drawdown"]),
            "net_return": float(net["total_return"]),
            "net_max_drawdown": float(net["max_drawdown"]),
            "trades": int(net["trades"]),
            "win_rate": float(net["win_rate"]),
            "total_fees_paid": float(net.get("total_fees_paid", 0.0)),
            "total_slippage_cost": float(net.get("total_slippage_cost", 0.0)),
            "latest_signal": str(gross["latest_signal"]),
            "exit_reasons": dict(gross.get("exit_reasons", {})),
        }
    summaries["test_period"] = (
        f"{datetime.fromisoformat(test_bars[0].timestamp).date().isoformat()} -> "
        f"{datetime.fromisoformat(test_bars[-1].timestamp).date().isoformat()} ({len(test_bars)} days)"
        if test_bars
        else ""
    )
    return summaries


def _per_symbol_record(
    *, symbol, sector, validation_rows, test_rows, validation_predictions, test_predictions,
    bars, split_by_day, risk_limits, assumptions, conviction_sizing=False
) -> dict:
    return {
        "symbol": symbol,
        "sector": sector,
        "validation_samples": len(validation_rows),
        "test_samples": len(test_rows),
        "validation_metrics": _classification_metrics(validation_rows, validation_predictions),
        "test_metrics": _classification_metrics(test_rows, test_predictions),
        "trade_summary": _trade_summaries(
            symbol=symbol, bars=bars, split_by_day=split_by_day, predictions=test_predictions,
            risk_limits=risk_limits, assumptions=assumptions, conviction_sizing=conviction_sizing,
        ),
    }


def _aggregate(per_symbol) -> dict:
    if not per_symbol:
        return {}
    accuracy = [row["test_metrics"]["direction_accuracy"] for row in per_symbol]
    balanced = [row["test_metrics"]["balanced_accuracy"] for row in per_symbol]
    f1 = [row["test_metrics"]["macro_f1"] for row in per_symbol]
    aggregate = {
        "mean_direction_accuracy": round(sum(accuracy) / len(accuracy), 6),
        "mean_balanced_accuracy": round(sum(balanced) / len(balanced), 6),
        "mean_macro_f1": round(sum(f1) / len(f1), 6),
        "symbols_at_or_above_50pct_accuracy": sum(1 for value in accuracy if value >= 0.5),
    }
    for threshold in SIGNAL_THRESHOLDS:
        gross_returns = [row["trade_summary"][threshold]["gross_return"] for row in per_symbol]
        net_returns = [row["trade_summary"][threshold]["net_return"] for row in per_symbol]
        gross_dd = [row["trade_summary"][threshold]["gross_max_drawdown"] for row in per_symbol]
        net_dd = [row["trade_summary"][threshold]["net_max_drawdown"] for row in per_symbol]
        trades = [row["trade_summary"][threshold]["trades"] for row in per_symbol]
        aggregate[threshold] = {
            "mean_gross_return": round(sum(gross_returns) / len(gross_returns), 6),
            "mean_net_return": round(sum(net_returns) / len(net_returns), 6),
            "mean_gross_max_drawdown": round(sum(gross_dd) / len(gross_dd), 6),
            "mean_net_max_drawdown": round(sum(net_dd) / len(net_dd), 6),
            "mean_trades": round(sum(trades) / len(trades), 2),
        }
    return aggregate


def _build_comparison(variant_results, v35_baseline) -> dict:
    base = v35_baseline["aggregate"]
    rows = []
    for variant in VARIANTS:
        aggregate = variant_results[variant]["aggregate"]
        rows.append(
            {
                "variant": variant,
                "mean_direction_accuracy": aggregate.get("mean_direction_accuracy", 0.0),
                "mean_balanced_accuracy": aggregate.get("mean_balanced_accuracy", 0.0),
                "mean_macro_f1": aggregate.get("mean_macro_f1", 0.0),
                "mean_net_return": aggregate.get(PRIMARY_THRESHOLD, {}).get("mean_net_return", 0.0),
                "mean_net_max_drawdown": aggregate.get(PRIMARY_THRESHOLD, {}).get("mean_net_max_drawdown", 0.0),
                "delta_balanced_accuracy_vs_v35": round(
                    aggregate.get("mean_balanced_accuracy", 0.0) - base.get("mean_balanced_accuracy", 0.0), 6
                ),
                "delta_net_return_vs_v35": round(
                    aggregate.get(PRIMARY_THRESHOLD, {}).get("mean_net_return", 0.0)
                    - base.get(PRIMARY_THRESHOLD, {}).get("mean_net_return", 0.0),
                    6,
                ),
            }
        )
    return {
        "primary_threshold": PRIMARY_THRESHOLD,
        "v35": {
            "mean_direction_accuracy": base.get("mean_direction_accuracy", 0.0),
            "mean_balanced_accuracy": base.get("mean_balanced_accuracy", 0.0),
            "mean_macro_f1": base.get("mean_macro_f1", 0.0),
            "mean_net_return": base.get(PRIMARY_THRESHOLD, {}).get("mean_net_return", 0.0),
            "mean_net_max_drawdown": base.get(PRIMARY_THRESHOLD, {}).get("mean_net_max_drawdown", 0.0),
        },
        "variants": rows,
    }


def _select_best_variant(variant_results, v35_baseline) -> dict:
    base_balanced = v35_baseline["aggregate"].get("mean_balanced_accuracy", 0.0)
    ranked = sorted(
        VARIANTS,
        key=lambda variant: (
            variant_results[variant]["aggregate"].get("mean_balanced_accuracy", 0.0),
            variant_results[variant]["aggregate"].get(PRIMARY_THRESHOLD, {}).get("mean_net_return", 0.0),
        ),
        reverse=True,
    )
    best = ranked[0]
    aggregate = variant_results[best]["aggregate"]
    return {
        "name": best,
        "selection_rule": "max mean balanced accuracy, then mean net return at the primary (medium) threshold",
        "mean_balanced_accuracy": aggregate.get("mean_balanced_accuracy", 0.0),
        "mean_net_return": aggregate.get(PRIMARY_THRESHOLD, {}).get("mean_net_return", 0.0),
        "beats_v35_balanced_accuracy": aggregate.get("mean_balanced_accuracy", 0.0) > base_balanced,
    }


# --------------------------------------------------------------------- charts


def _write_charts(output_dir: Path, variant_results, v35_baseline) -> dict:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting optional
        return {"enabled": False, "reason": repr(exc)[:200]}

    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    labels = VARIANTS + ["V3-5"]
    returns = [
        variant_results[v]["aggregate"].get(PRIMARY_THRESHOLD, {}).get("mean_net_return", 0.0) for v in VARIANTS
    ] + [v35_baseline["aggregate"].get(PRIMARY_THRESHOLD, {}).get("mean_net_return", 0.0)]
    drawdowns = [
        variant_results[v]["aggregate"].get(PRIMARY_THRESHOLD, {}).get("mean_net_max_drawdown", 0.0) for v in VARIANTS
    ] + [v35_baseline["aggregate"].get(PRIMARY_THRESHOLD, {}).get("mean_net_max_drawdown", 0.0)]

    paths = {}
    for name, values, title in (
        ("returns_by_variant.png", returns, "Mean net return (test, medium threshold)"),
        ("drawdown_by_variant.png", drawdowns, "Mean net max drawdown (test, medium threshold)"),
    ):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        colors = ["#4C72B0"] * len(VARIANTS) + ["#C44E52"]
        axis.bar(labels, [value * 100 for value in values], color=colors)
        axis.set_ylabel("percent")
        axis.set_title(title)
        axis.axhline(0, color="black", linewidth=0.8)
        for tick in axis.get_xticklabels():
            tick.set_rotation(20)
            tick.set_horizontalalignment("right")
        figure.tight_layout()
        path = charts_dir / name
        figure.savefig(path, dpi=130)
        plt.close(figure)
        paths[name] = str(path)
    return {"enabled": True, "paths": paths}


# -------------------------------------------------------------------- helpers


def _jsonable(obj):
    """Recursively convert numpy scalars/arrays (and tuples) into JSON-native types."""
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _assumptions_dict(assumptions) -> dict:
    return {
        "starting_capital": assumptions.starting_capital,
        "quoted_spread_bps": assumptions.quoted_spread_bps,
        "market_impact_bps": assumptions.market_impact_bps,
        "stop_extra_slippage_bps": assumptions.stop_extra_slippage_bps,
        "max_bar_participation_rate": assumptions.max_bar_participation_rate,
        "finra_taf_per_share_sell": assumptions.finra_taf_per_share_sell,
        "finra_taf_cap_per_trade": assumptions.finra_taf_cap_per_trade,
    }


def _summary_markdown(result: dict) -> str:
    lines = [
        "# V3-Q3E Qwen3-Embedding Directional Run",
        "",
        f"- model: `{result['model_name']}`",
        f"- role: `{result['model_role']}`",
        f"- CUDA used: `{result['cuda_used']}`",
        f"- encoder: `{json.dumps(result['encoder_settings'])}`",
        f"- train rows used: `{result['dataset_sizes']['train_rows_used']}` "
        f"(class balance `{result['dataset_sizes']['train_class_distribution']}`)",
        f"- sentiment features used: `{result['training_config']['sentiment_features_used']}` "
        f"(coverage `{result['training_config']['sentiment_coverage']}`)",
        "",
        "## Date splits (with purge/embargo)",
        "",
    ]
    for split, span in result["date_splits"].items():
        lines.append(f"- {split}: `{span}`")
    lines.extend(["", "## Comparison vs V3-5 (medium threshold, net of costs; loose 5% exit, market-on-close)", "",
                  "| Branch | Mean dir. acc | Balanced acc | Macro F1 | Mean net return | Mean net max DD |",
                  "|---|---:|---:|---:|---:|---:|"])
    v35 = result["comparison_vs_v35"]["v35"]
    lines.append(
        f"| `V3-5` | `{v35['mean_direction_accuracy']:.2%}` | `{v35['mean_balanced_accuracy']:.2%}` | "
        f"`{v35['mean_macro_f1']:.4f}` | `{v35['mean_net_return']:.2%}` | `{v35['mean_net_max_drawdown']:.2%}` |"
    )
    for row in result["comparison_vs_v35"]["variants"]:
        lines.append(
            f"| `{row['variant']}` | `{row['mean_direction_accuracy']:.2%}` | `{row['mean_balanced_accuracy']:.2%}` | "
            f"`{row['mean_macro_f1']:.4f}` | `{row['mean_net_return']:.2%}` | `{row['mean_net_max_drawdown']:.2%}` |"
        )
    best = result["best_variant"]
    lines.extend(["", f"Best variant: `{best['name']}` — {best['selection_rule']} "
                  f"(beats V3-5 balanced accuracy: `{best['beats_v35_balanced_accuracy']}`).", ""])

    for variant in VARIANTS:
        lines.extend([f"## {variant}", "",
                      "| Symbol | Dir. acc | Balanced acc | Macro F1 | Net return (med) | Net max DD (med) | Trades |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for row in result["variant_results"][variant]["per_symbol"]:
            metrics = row["test_metrics"]
            trade = row["trade_summary"][PRIMARY_THRESHOLD]
            lines.append(
                f"| `{row['symbol']}` | `{metrics['direction_accuracy']:.2%}` | `{metrics['balanced_accuracy']:.2%}` | "
                f"`{metrics['macro_f1']:.4f}` | `{trade['net_return']:.2%}` | `{trade['net_max_drawdown']:.2%}` | "
                f"`{trade['trades']}` |"
            )
        lines.append("")
    if result.get("charts", {}).get("enabled"):
        lines.extend(["## Charts", ""] + [f"- `{path}`" for path in result["charts"]["paths"].values()] + [""])
    return "\n".join(lines) + "\n"
