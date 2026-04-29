from __future__ import annotations

import logging

from algoding.data.models import MarketEvent
from algoding.execution.models import OrderIntent
from algoding.portfolio.risk import RiskEngine
from algoding.shadow.null_sink import NullOrderSink

logger = logging.getLogger(__name__)


class ShadowBarStrategy:
    def __init__(
        self,
        threshold_bps: int,
        order_quantity: float,
        sink: NullOrderSink,
        risk_engine: RiskEngine,
    ) -> None:
        self._threshold = threshold_bps / 10_000
        self._order_quantity = order_quantity
        self._sink = sink
        self._risk_engine = risk_engine
        self._last_price_by_symbol: dict[str, float] = {}
        self.orders_emitted = 0

    def on_market_event(self, event: MarketEvent) -> None:
        if event.event_type != "bar":
            return

        previous_price = self._last_price_by_symbol.get(event.symbol)
        self._last_price_by_symbol[event.symbol] = event.price
        if previous_price is None or previous_price <= 0:
            return

        upper_bound = previous_price * (1 + self._threshold)
        lower_bound = previous_price * (1 - self._threshold)
        if event.price >= upper_bound:
            self._emit_order(event.symbol, "buy")
        elif event.price <= lower_bound:
            self._emit_order(event.symbol, "sell")

    def _emit_order(self, symbol: str, side: str) -> None:
        intent = OrderIntent.sample(symbol=symbol, side=side)
        intent.quantity = self._order_quantity
        self._risk_engine.validate(intent)
        payload = self._sink.submit(intent)
        self.orders_emitted += 1
        logger.info("Shadow order emitted", extra={"payload": payload})
