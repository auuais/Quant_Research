from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from algoding.research.news_labeler import BundleMetadata, build_structured_prompt


@dataclass(frozen=True)
class DeepseekLoraTrainingConfig:
    base_model_path: str
    output_dir: str
    max_length: int = 512
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_train_samples: int | None = 1200
    max_eval_samples: int | None = 300


class PromptCompletionCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __call__(self, features: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        input_id_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        for feature in features:
            prompt = str(feature["prompt"])
            completion = str(feature["completion"])
            prompt_ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
            completion_ids = self._tokenizer(completion, add_special_tokens=False)["input_ids"]
            eos = [self._tokenizer.eos_token_id] if self._tokenizer.eos_token_id is not None else []
            input_ids = (prompt_ids + completion_ids + eos)[: self._max_length]
            labels = ([-100] * len(prompt_ids) + completion_ids + eos)[: self._max_length]
            attention = [1] * len(input_ids)
            input_id_rows.append(input_ids)
            label_rows.append(labels)
            attention_rows.append(attention)

        padded_inputs = self._tokenizer.pad(
            {"input_ids": input_id_rows, "attention_mask": attention_rows},
            padding=True,
            return_tensors="pt",
        )
        max_seq = padded_inputs["input_ids"].shape[1]
        padded_labels = []
        for labels in label_rows:
            padded = labels + ([-100] * (max_seq - len(labels)))
            padded_labels.append(padded)
        padded_inputs["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return padded_inputs


class DeepseekLoraTrainer:
    def __init__(self, config: DeepseekLoraTrainingConfig) -> None:
        self._config = config
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device != "cuda":
            raise RuntimeError("DeepseekLoraTrainer requires CUDA for QLoRA training.")

    def fit(self, train_records: list[dict[str, object]], eval_records: list[dict[str, object]]) -> dict[str, object]:
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(self._config.base_model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self._config.base_model_path,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=self._config.lora_r,
            lora_alpha=self._config.lora_alpha,
            lora_dropout=self._config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)

        train_dataset = Dataset.from_list(self._limit_records(train_records, self._config.max_train_samples))
        eval_dataset = Dataset.from_list(self._limit_records(eval_records, self._config.max_eval_samples))
        collator = PromptCompletionCollator(tokenizer=tokenizer, max_length=self._config.max_length)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            learning_rate=self._config.learning_rate,
            num_train_epochs=self._config.num_train_epochs,
            per_device_train_batch_size=self._config.per_device_train_batch_size,
            per_device_eval_batch_size=self._config.per_device_eval_batch_size,
            gradient_accumulation_steps=self._config.gradient_accumulation_steps,
            eval_strategy="epoch" if len(eval_dataset) else "no",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=bool(len(eval_dataset)),
            fp16=True,
            logging_steps=10,
            report_to=[],
            remove_unused_columns=False,
            optim="paged_adamw_8bit",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if len(eval_dataset) else None,
            data_collator=collator,
        )
        result = trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        return {
            "output_dir": str(output_dir),
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset),
            "train_runtime_seconds": round(float(getattr(result, "metrics", {}).get("train_runtime", 0.0)), 3),
            "train_loss": round(float(getattr(result, "metrics", {}).get("train_loss", 0.0)), 6),
        }

    @staticmethod
    def _limit_records(records: list[dict[str, object]], limit: int | None) -> list[dict[str, object]]:
        if limit is None or len(records) <= limit:
            return records
        return records[:limit]


class FinetunedDeepseekStructuredScorer:
    def __init__(self, *, base_model_path: str, adapter_path: str, cache_path: str | Path) -> None:
        self._base_model_path = base_model_path
        self._adapter_path = adapter_path
        self._cache_path = Path(cache_path)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load_cache()
        self._tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        self._model = PeftModel.from_pretrained(base_model, adapter_path)
        self._model.eval()

    def score_bundles(self, bundles: list[BundleMetadata], batch_size: int = 2) -> dict[str, dict[str, object]]:
        results: dict[str, dict[str, object]] = {}
        pending: list[BundleMetadata] = []
        for bundle in bundles:
            cache_key = self._cache_key(bundle)
            cached = self._cache.get(cache_key)
            if cached is not None:
                results[bundle.trading_day] = cached
                continue
            pending.append(bundle)
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            prompts = [build_structured_prompt(bundle) for bundle in batch]
            parsed = self._generate_and_parse(prompts)
            for bundle, payload in zip(batch, parsed):
                cache_key = self._cache_key(bundle)
                record = {
                    "symbol": bundle.symbol,
                    "trading_day": bundle.trading_day,
                    "headline_count": bundle.headline_count,
                    "text_hash": self._text_hash(bundle.text),
                    **payload,
                }
                self._cache[cache_key] = record
                self._append_cache(record)
                results[bundle.trading_day] = record
        return results

    def _generate_and_parse(self, prompts: list[str]) -> list[dict[str, object]]:
        encoded = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=480,
        ).to("cuda")
        with torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=80,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        resolved: list[dict[str, object]] = []
        for row, prompt in zip(generated, prompts):
            full_text = self._tokenizer.decode(row, skip_special_tokens=True)
            completion = full_text[len(prompt) :].strip() if full_text.startswith(prompt) else full_text.strip()
            resolved.append(self._parse_completion(completion))
        return resolved

    def _parse_completion(self, text: str) -> dict[str, object]:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        raw = match.group(0) if match else text
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {}
        label = str(payload.get("label", "neutral")).strip().lower()
        if label not in {"bullish", "bearish", "neutral"}:
            label = "neutral"
        strength = str(payload.get("strength", "low")).strip().lower()
        if strength not in {"low", "medium", "high"}:
            strength = "low"
        event_type = str(payload.get("event_type", "other")).strip().lower()
        session = str(payload.get("session", "mixed")).strip().lower()
        horizon = str(payload.get("horizon", "5d")).strip().lower()
        return {
            "label": label,
            "strength": strength,
            "event_type": event_type,
            "session": session,
            "horizon": horizon,
            "raw_completion": text,
        }

    @staticmethod
    def _cache_key(bundle: BundleMetadata) -> str:
        return f"{bundle.symbol}|{bundle.trading_day}|{bundle.headline_count}|{FinetunedDeepseekStructuredScorer._text_hash(bundle.text)}"

    def _load_cache(self) -> dict[str, dict[str, object]]:
        if not self._cache_path.exists():
            return {}
        cache: dict[str, dict[str, object]] = {}
        for line in self._cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            cache[self._cache_key_from_payload(payload)] = payload
        return cache

    def _append_cache(self, payload: dict[str, object]) -> None:
        with self._cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    @staticmethod
    def _cache_key_from_payload(payload: dict[str, Any]) -> str:
        return (
            f"{payload.get('symbol')}|{payload.get('trading_day')}|"
            f"{payload.get('headline_count', 0)}|{payload.get('text_hash', 0)}"
        )

    @staticmethod
    def _text_hash(text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()
