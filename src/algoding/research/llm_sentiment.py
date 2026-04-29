from __future__ import annotations

import hashlib
import json
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import httpx
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

from algoding.data.news import NewsArticle
from algoding.settings import Settings


@dataclass(frozen=True)
class DailyNewsBundle:
    symbol: str
    trading_day: str
    article_ids: tuple[int, ...]
    headline_count: int
    text: str


@dataclass(frozen=True)
class DailySentimentScore:
    symbol: str
    trading_day: str
    label: str
    score: float
    article_ids: tuple[int, ...]
    prompt_hash: str
    model_name: str


class LocalLlmNewsSentimentEngine:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache_dir = settings.llm_news_cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path = self._cache_dir / "daily_bundle_scores.jsonl"
        self._offload_dir = self._cache_dir / "offload"
        self._offload_dir.mkdir(parents=True, exist_ok=True)
        self._model_name = settings.llm_news_model_name
        self._cache = self._load_cache()
        self._candidate_labels = ["bullish", "neutral", "bearish"]
        self._version = "likelihood_v3"
        self._backend = "openai" if self._is_openai_model(self._model_name) else "local"
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._quantization = settings.llm_news_quantization.strip().lower()
        self._force_gpu = settings.llm_news_force_gpu
        self._http_client: httpx.Client | None = None

        if self._force_gpu and self._device != "cuda":
            raise RuntimeError("LLM_NEWS_FORCE_GPU is enabled, but CUDA is not available.")

        if self._backend == "openai":
            if not settings.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is required when LLM_NEWS_MODEL_NAME targets a hosted OpenAI model."
                )
            self._model_type = "remote"
            self._config = None
            self._tokenizer = None
            self._model = None
            self._http_client = httpx.Client(
                base_url=settings.openai_base_url.rstrip("/"),
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
        else:
            self._config = AutoConfig.from_pretrained(self._model_name, trust_remote_code=True)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
            if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            quantization_config = self._build_quantization_config()
            if bool(getattr(self._config, "is_encoder_decoder", False)):
                self._model_type = "seq2seq"
                model_kwargs = {
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                }
                if quantization_config is not None:
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = {"": 0} if self._force_gpu else "auto"
                else:
                    model_kwargs["dtype"] = torch.float16 if self._device == "cuda" else torch.float32
                    model_kwargs["device_map"] = {"": 0} if self._force_gpu and self._device == "cuda" else ("auto" if self._device == "cuda" else None)
                    model_kwargs["offload_folder"] = (
                        None
                        if self._force_gpu
                        else (str(self._offload_dir) if self._device == "cuda" else None)
                    )
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name, **model_kwargs)
            else:
                self._model_type = "causal"
                model_kwargs = {
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                }
                if quantization_config is not None:
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = {"": 0} if self._force_gpu else "auto"
                else:
                    model_kwargs["dtype"] = torch.float16 if self._device == "cuda" else torch.float32
                    model_kwargs["device_map"] = {"": 0} if self._force_gpu and self._device == "cuda" else ("auto" if self._device == "cuda" else None)
                    model_kwargs["offload_folder"] = (
                        None
                        if self._force_gpu
                        else (str(self._offload_dir) if self._device == "cuda" else None)
                    )
                self._model = AutoModelForCausalLM.from_pretrained(self._model_name, **model_kwargs)
            self._validate_runtime_device()
            self._model.eval()

    def score_bundles(self, bundles: list[DailyNewsBundle], *, batch_size: int = 8) -> dict[str, DailySentimentScore]:
        results: dict[str, DailySentimentScore] = {}
        missing: list[tuple[str, DailyNewsBundle, str]] = []

        for bundle in bundles:
            prompt_hash = self._prompt_hash(bundle)
            cache_key = self._cache_key(bundle.symbol, bundle.trading_day, prompt_hash)
            cached = self._cache.get(cache_key)
            if cached is not None:
                results[bundle.trading_day] = cached
                continue
            missing.append((cache_key, bundle, prompt_hash))

        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            prompts = [self._build_prompt(bundle) for _, bundle, _ in batch]
            scores = self._score_prompts(prompts)
            for (cache_key, bundle, prompt_hash), (label, score) in zip(batch, scores):
                sentiment = DailySentimentScore(
                    symbol=bundle.symbol,
                    trading_day=bundle.trading_day,
                    label=label,
                    score=score,
                    article_ids=bundle.article_ids,
                    prompt_hash=prompt_hash,
                    model_name=self._model_name,
                )
                self._cache[cache_key] = sentiment
                self._append_cache(sentiment)
                results[bundle.trading_day] = sentiment

        return results

    def build_trading_day_bundles(
        self,
        symbol: str,
        articles: Iterable[NewsArticle],
        *,
        trading_days: list[date],
        max_headlines_per_day: int = 5,
        timezone_name: str = "America/New_York",
        market_close_hour: int = 16,
    ) -> list[DailyNewsBundle]:
        grouped: dict[str, list[NewsArticle]] = {}
        tz = ZoneInfo(timezone_name)
        if not trading_days:
            return []
        for article in sorted(articles, key=lambda item: item.created_at):
            created = datetime.fromisoformat(article.created_at).astimezone(tz)
            effective_day = self._effective_trading_day(
                created=created,
                trading_days=trading_days,
                market_close_hour=market_close_hour,
            )
            if effective_day is None:
                continue
            grouped.setdefault(effective_day.isoformat(), []).append(article)

        bundles: list[DailyNewsBundle] = []
        for trading_day, items in sorted(grouped.items()):
            selected = items[:max_headlines_per_day]
            segments: list[str] = []
            for index, item in enumerate(selected, start=1):
                headline = item.headline.strip().replace("\n", " ")
                summary = item.summary.strip().replace("\n", " ")
                if summary:
                    segments.append(f"{index}. {headline} Summary: {summary}")
                else:
                    segments.append(f"{index}. {headline}")
            text = "\n".join(segments)
            bundles.append(
                DailyNewsBundle(
                    symbol=symbol,
                    trading_day=trading_day,
                    article_ids=tuple(item.article_id for item in selected),
                    headline_count=len(selected),
                    text=text,
                )
            )
        return bundles

    def fill_missing_days(
        self,
        bars: list[date],
        scores: dict[str, DailySentimentScore],
        symbol: str,
        *,
        carry_days: int = 3,
        decay: float = 0.6,
    ) -> list[DailySentimentScore]:
        resolved: list[DailySentimentScore] = []
        recent_scores: list[float] = []
        for trading_day in bars:
            key = trading_day.isoformat()
            sentiment = scores.get(key)
            if sentiment is None:
                carried = 0.0
                if recent_scores:
                    carried = sum(value * (decay ** idx) for idx, value in enumerate(reversed(recent_scores[-carry_days:])))
                sentiment = DailySentimentScore(
                    symbol=symbol,
                    trading_day=key,
                    label="neutral" if abs(carried) < 0.25 else ("bullish" if carried > 0 else "bearish"),
                    score=round(max(-1.0, min(1.0, carried)), 6),
                    article_ids=(),
                    prompt_hash="",
                    model_name=self._model_name,
                )
            if sentiment.score != 0.0:
                recent_scores.append(float(sentiment.score))
            resolved.append(sentiment)
        return resolved

    def _load_cache(self) -> dict[str, DailySentimentScore]:
        if not self._cache_path.exists():
            return {}
        cache: dict[str, DailySentimentScore] = {}
        for line in self._cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            score = DailySentimentScore(
                symbol=raw["symbol"],
                trading_day=raw["trading_day"],
                label=raw["label"],
                score=float(raw["score"]),
                article_ids=tuple(int(value) for value in raw.get("article_ids", [])),
                prompt_hash=raw.get("prompt_hash", ""),
                model_name=raw.get("model_name", self._model_name),
            )
            cache[self._cache_key(score.symbol, score.trading_day, score.prompt_hash)] = score
        return cache

    def _append_cache(self, score: DailySentimentScore) -> None:
        payload = {
            "symbol": score.symbol,
            "trading_day": score.trading_day,
            "label": score.label,
            "score": score.score,
            "article_ids": list(score.article_ids),
            "prompt_hash": score.prompt_hash,
            "model_name": score.model_name,
        }
        with self._cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _build_prompt(self, bundle: DailyNewsBundle) -> str:
        return (
            f"Classify the likely 1-5 trading day impact on {bundle.symbol} from these news headlines. "
            "Choose the strongest market-relevant label. Do not default to neutral unless the headlines are mixed or insignificant. "
            "Respond with exactly one word: bullish, bearish, or neutral.\n\n"
            f"Date: {bundle.trading_day}\n"
            f"Headlines:\n{bundle.text}\n\n"
            "Answer:"
        )

    def _prompt_hash(self, bundle: DailyNewsBundle) -> str:
        digest = hashlib.sha256()
        digest.update(self._build_prompt(bundle).encode("utf-8"))
        digest.update(self._model_name.encode("utf-8"))
        digest.update(self._version.encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _cache_key(symbol: str, trading_day: str, prompt_hash: str) -> str:
        return f"{symbol}|{trading_day}|{prompt_hash}"

    def _score_prompts(self, prompts: list[str]) -> list[tuple[str, float]]:
        if self._backend == "openai":
            return self._score_prompts_openai(prompts)
        if self._model_type == "causal":
            return self._score_prompts_causal(prompts)
        encoded = self._tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._runtime_device()) for key, value in encoded.items()}
        return self._score_prompts_seq2seq(encoded, len(prompts))

    def _score_prompts_seq2seq(self, encoded: dict[str, torch.Tensor], batch_size: int) -> list[tuple[str, float]]:
        label_scores: list[list[float]] = [[] for _ in range(batch_size)]
        for label in self._candidate_labels:
            label_tokens = self._tokenizer(
                [label] * batch_size,
                padding=True,
                return_tensors="pt",
            )
            label_ids = label_tokens["input_ids"].to(self._device)
            label_ids[label_ids == self._tokenizer.pad_token_id] = -100
            with torch.inference_mode():
                output = self._model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    labels=label_ids,
                )
            logits = output.logits[:, :-1, :].contiguous()
            targets = label_ids[:, 1:].contiguous()
            flat_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(batch_size, -1)
            mask = (targets != -100).float()
            losses = (flat_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            for index in range(batch_size):
                label_scores[index].append(float(-losses[index].item()))

        resolved: list[tuple[str, float]] = []
        for values in label_scores:
            logits = torch.tensor(values, dtype=torch.float32)
            probs = torch.softmax(logits, dim=0)
            bullish, neutral, bearish = [float(value) for value in probs]
            score = bullish - bearish
            best_index = int(torch.argmax(probs).item())
            resolved.append((self._candidate_labels[best_index], round(score, 6)))
        return resolved

    def _score_prompts_openai(self, prompts: list[str]) -> list[tuple[str, float]]:
        if self._http_client is None:
            raise RuntimeError("OpenAI HTTP client is not initialized.")
        resolved: list[tuple[str, float]] = []
        for prompt in prompts:
            response = self._http_client.post(
                "/responses",
                json={
                    "model": self._model_name,
                    "instructions": (
                        "You are classifying short-term market news. "
                        "Return compact JSON with keys label and confidence. "
                        "label must be one of bullish, bearish, neutral. "
                        "confidence must be a number between 0 and 1."
                    ),
                    "input": prompt,
                    "max_output_tokens": 32,
                },
            )
            response.raise_for_status()
            payload = response.json()
            text = self._extract_response_text(payload)
            resolved.append(self._parse_openai_sentiment(text))
        return resolved

    def _score_prompts_causal(self, prompts: list[str]) -> list[tuple[str, float]]:
        prompt_tokens = self._tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=480,
            add_special_tokens=False,
        )
        prompt_lengths = [len(ids) for ids in prompt_tokens["input_ids"]]
        label_scores: list[list[float]] = [[] for _ in range(len(prompts))]

        for label in self._candidate_labels:
            suffix = f" {label}"
            label_token_ids = self._tokenizer(suffix, add_special_tokens=False)["input_ids"]
            combined = self._tokenizer(
                [prompt + suffix for prompt in prompts],
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=False,
                return_tensors="pt",
            )
            combined = {key: value.to(self._runtime_device()) for key, value in combined.items()}
            with torch.inference_mode():
                output = self._model(
                    input_ids=combined["input_ids"],
                    attention_mask=combined["attention_mask"],
                )
            log_probs = F.log_softmax(output.logits, dim=-1)
            for index, prompt_len in enumerate(prompt_lengths):
                seq_len = int(combined["attention_mask"][index].sum().item())
                valid_positions = []
                for offset, token_id in enumerate(label_token_ids):
                    token_position = prompt_len + offset
                    if token_position >= seq_len or token_position == 0:
                        continue
                    valid_positions.append(log_probs[index, token_position - 1, token_id].item())
                score = sum(valid_positions) / max(1, len(valid_positions))
                label_scores[index].append(float(score))

        resolved: list[tuple[str, float]] = []
        for values in label_scores:
            logits = torch.tensor(values, dtype=torch.float32)
            probs = torch.softmax(logits, dim=0)
            bullish, neutral, bearish = [float(value) for value in probs]
            score = bullish - bearish
            best_index = int(torch.argmax(probs).item())
            resolved.append((self._candidate_labels[best_index], round(score, 6)))
        return resolved

    @staticmethod
    def _effective_trading_day(
        *,
        created: datetime,
        trading_days: list[date],
        market_close_hour: int,
    ) -> date | None:
        article_day = created.date()
        index = bisect_left(trading_days, article_day)
        if index >= len(trading_days):
            return None
        if index < len(trading_days) and trading_days[index] == article_day and created.hour < market_close_hour:
            return article_day
        next_index = index + 1 if index < len(trading_days) and trading_days[index] == article_day else index
        if next_index >= len(trading_days):
            return None
        return trading_days[next_index]

    @staticmethod
    def _is_openai_model(model_name: str) -> bool:
        normalized = model_name.strip().lower()
        return normalized.startswith("gpt-") or normalized.startswith("o")

    @staticmethod
    def _extract_response_text(payload: dict[str, object]) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        fragments: list[str] = []
        for item in payload.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    fragments.append(text.strip())
        return "\n".join(fragments).strip()

    def _parse_openai_sentiment(self, text: str) -> tuple[str, float]:
        stripped = text.strip()
        if not stripped:
            return "neutral", 0.0
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                label = str(parsed.get("label", "neutral")).strip().lower()
                confidence = float(parsed.get("confidence", 0.0))
                return self._normalize_label_score(label=label, confidence=confidence)
        except (ValueError, TypeError):
            pass
        first_token = stripped.split()[0].strip().lower().strip(".,:;\"'")
        label = first_token if first_token in self._candidate_labels else "neutral"
        confidence = 1.0 if label != "neutral" else 0.0
        return self._normalize_label_score(label=label, confidence=confidence)

    @staticmethod
    def _normalize_label_score(*, label: str, confidence: float) -> tuple[str, float]:
        bounded = max(0.0, min(1.0, float(confidence)))
        if label == "bullish":
            return "bullish", round(bounded, 6)
        if label == "bearish":
            return "bearish", round(-bounded, 6)
        return "neutral", 0.0

    def _build_quantization_config(self) -> BitsAndBytesConfig | None:
        if self._device != "cuda" or not self._quantization:
            return None
        if self._quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        if self._quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        raise RuntimeError(f"Unsupported LLM_NEWS_QUANTIZATION value: {self._quantization}")

    def _runtime_device(self) -> torch.device | str:
        if self._backend == "openai" or self._model is None:
            return self._device
        device_map = getattr(self._model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for target in device_map.values():
                if isinstance(target, int):
                    return torch.device(f"cuda:{target}")
                if isinstance(target, str) and target.startswith("cuda"):
                    return torch.device(target)
        return getattr(self._model, "device", self._device)

    def _validate_runtime_device(self) -> None:
        if not self._force_gpu or self._model is None:
            return
        device_map = getattr(self._model, "hf_device_map", None)
        if isinstance(device_map, dict):
            invalid = [value for value in device_map.values() if value not in (0, "cuda:0", "cuda")]
            if invalid:
                raise RuntimeError(
                    "LLM_NEWS_FORCE_GPU requires all model weights on GPU, "
                    f"but found non-GPU device map entries: {invalid}"
                )
            return
        runtime_device = str(self._runtime_device())
        if "cuda" not in runtime_device:
            raise RuntimeError(
                "LLM_NEWS_FORCE_GPU requires the model to run on GPU, "
                f"but resolved runtime device was {runtime_device}."
            )
