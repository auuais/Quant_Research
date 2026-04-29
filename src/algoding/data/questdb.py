from __future__ import annotations

import httpx
from urllib.parse import quote

from algoding.data.models import MarketEvent


class QuestDBClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def ingest_market_event(self, event: MarketEvent) -> None:
        response = httpx.post(
            f"{self._base_url}/write",
            content=event.to_line_protocol(),
            headers={"Content-Type": "text/plain"},
            timeout=5.0,
        )
        response.raise_for_status()

    def query(self, sql: str) -> dict[str, object]:
        response = httpx.get(f"{self._base_url}/exec?query={quote(sql)}", timeout=5.0)
        response.raise_for_status()
        return response.json()

    def count_market_events(self) -> int:
        try:
            payload = self.query("select count(*) from market_events")
        except httpx.HTTPStatusError as exc:
            response_text = exc.response.text.lower()
            if "table does not exist" in response_text:
                return 0
            raise
        dataset = payload.get("dataset", [])
        if not dataset:
            return 0
        return int(dataset[0][0])
