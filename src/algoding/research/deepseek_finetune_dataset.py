from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from algoding.research.news_labeler import LabeledNewsExample


def export_finetune_jsonl(
    examples: list[LabeledNewsExample],
    *,
    output_path: str | Path,
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            payload = {
                "symbol": example.symbol,
                "trading_day": example.trading_day,
                "prompt": example.prompt,
                "completion": example.completion,
                "label": example.label,
                "strength": example.strength,
                "event_type": example.event_type,
                "session_label": example.session_label,
                "article_count": example.article_count,
                "forward_return_5d": example.forward_return_5d,
                "forward_excess_return_5d": example.forward_excess_return_5d,
                "realized_vol_20d": example.realized_vol_20d,
                "threshold": example.threshold,
            }
            handle.write(json.dumps(payload) + "\n")
    return str(path)


def examples_to_records(examples: list[LabeledNewsExample]) -> list[dict[str, object]]:
    return [asdict(example) for example in examples]
