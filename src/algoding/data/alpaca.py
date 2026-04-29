from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timezone

from alpaca.data.enums import DataFeed
from alpaca.data.live.stock import StockDataStream

from algoding.data.models import MarketEvent
from algoding.data.questdb import QuestDBClient
from algoding.settings import Settings

logger = logging.getLogger(__name__)


class AlpacaBarIngestor:
    def __init__(
        self,
        settings: Settings,
        questdb: QuestDBClient,
        subscribers: list[Callable[[MarketEvent], None]] | None = None,
    ) -> None:
        self._settings = settings
        self._questdb = questdb
        self._subscribers = subscribers or []
        self._events_seen = 0
        self._max_events: int | None = None
        self._stream: StockDataStream | None = None

    def run(self, symbols: list[str], max_events: int | None = None) -> None:
        if not self._settings.has_alpaca_credentials:
            raise RuntimeError("Alpaca credentials are required in .env for live ingestion.")
        if not symbols:
            raise ValueError("At least one symbol is required.")

        try:
            feed = DataFeed(self._settings.alpaca_data_feed.lower())
        except ValueError as exc:
            raise ValueError(
                f"Unsupported Alpaca data feed: {self._settings.alpaca_data_feed}"
            ) from exc

        self._max_events = max_events
        self._stream = StockDataStream(
            api_key=self._settings.alpaca_api_key,
            secret_key=self._settings.alpaca_secret_key,
            feed=feed,
            raw_data=False,
        )
        self._stream.subscribe_bars(self._handle_bar, *symbols)
        logger.info("Starting Alpaca bar stream", extra={"symbols": symbols, "feed": feed.value})
        self._stream.run()

    async def _handle_bar(self, bar: object) -> None:
        event = self._to_market_event(bar)
        self._questdb.ingest_market_event(event)
        for subscriber in self._subscribers:
            subscriber(event)
        self._events_seen += 1
        logger.info(
            "Ingested live bar",
            extra={"symbol": event.symbol, "price": event.price, "events_seen": self._events_seen},
        )
        if (
            self._stream is not None
            and self._max_events is not None
            and self._events_seen >= self._max_events
        ):
            logger.info("Stopping Alpaca bar stream", extra={"max_events": self._max_events})
            asyncio.create_task(self._stream.stop_ws())

    @staticmethod
    def _to_market_event(bar: object) -> MarketEvent:
        timestamp = getattr(bar, "timestamp", None)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        return MarketEvent(
            symbol=getattr(bar, "symbol"),
            event_type="bar",
            price=float(getattr(bar, "close")),
            volume=float(getattr(bar, "volume")),
            event_ts=timestamp,
            received_ts=datetime.now(timezone.utc),
        )
