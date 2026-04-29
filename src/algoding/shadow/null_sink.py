from __future__ import annotations

import json
from pathlib import Path

from algoding.execution.models import OrderIntent


class NullOrderSink:
    def __init__(self, sink_path: Path) -> None:
        self._sink_path = sink_path
        self._sink_path.parent.mkdir(parents=True, exist_ok=True)

    def submit(self, intent: OrderIntent) -> dict[str, object]:
        payload = intent.to_dict()
        payload["mode"] = "shadow"
        with self._sink_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str))
            handle.write("\n")
        return payload
