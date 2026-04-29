from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, StopOrderRequest, TrailingStopOrderRequest

from algoding.execution.models import OrderIntent
from algoding.settings import Settings

logger = logging.getLogger(__name__)


def _round_broker_price(price: float) -> float:
    return round(price, 2 if price >= 1 else 4)


class PaperOrderRouter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None
        if settings.has_alpaca_credentials:
            self._client = TradingClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
                paper=True,
            )

    def submit_order(self, intent: OrderIntent) -> dict[str, object]:
        payload = {
            "mode": "paper",
            "broker": "alpaca",
            "base_url": self._settings.alpaca_paper_base_url,
            "symbol": intent.symbol,
            "side": intent.side,
            "quantity": intent.quantity,
            "client_order_id": intent.client_order_id,
            "strategy_name": intent.strategy_name,
        }
        if self._client is None:
            logger.info("Prepared paper order payload", extra={"payload": payload})
            return payload

        request = MarketOrderRequest(
            symbol=intent.symbol,
            qty=float(intent.quantity) if intent.quantity else None,
            notional=float(intent.notional) if intent.notional else None,
            side=OrderSide.BUY if intent.side.lower() == "buy" else OrderSide.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            client_order_id=intent.client_order_id,
        )
        order = self._client.submit_order(order_data=request)
        payload["submitted"] = True
        payload["broker_order_id"] = str(getattr(order, "id", ""))
        payload["notional"] = intent.notional
        return payload

    def list_open_orders(self, symbols: list[str] | None = None) -> list[object]:
        if self._client is None:
            return []
        request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=symbols,
            limit=500,
            after=datetime.now(timezone.utc) - timedelta(days=30),
        )
        return list(self._client.get_orders(filter=request))

    def cancel_order(self, order_id: str) -> None:
        if self._client is None:
            return
        self._client.cancel_order_by_id(order_id)

    def submit_trailing_stop_order(
        self,
        *,
        symbol: str,
        quantity: float,
        trail_percent: float,
        strategy_name: str,
        client_order_id: str | None = None,
    ) -> dict[str, object]:
        payload = {
            "mode": "paper",
            "broker": "alpaca",
            "base_url": self._settings.alpaca_paper_base_url,
            "symbol": symbol,
            "side": "sell",
            "quantity": quantity,
            "client_order_id": client_order_id or str(uuid4()),
            "strategy_name": strategy_name,
            "type": "trailing_stop",
            "trail_percent": trail_percent,
        }
        if self._client is None:
            logger.info("Prepared broker-native trailing stop payload", extra={"payload": payload})
            return payload

        fractional = abs(float(quantity) - round(float(quantity))) > 1e-9
        request = TrailingStopOrderRequest(
            symbol=symbol,
            qty=float(quantity),
            side=OrderSide.SELL,
            type=OrderType.TRAILING_STOP,
            time_in_force=TimeInForce.DAY if fractional else TimeInForce.GTC,
            client_order_id=str(payload["client_order_id"]),
            trail_percent=float(trail_percent),
        )
        order = self._client.submit_order(order_data=request)
        payload["submitted"] = True
        payload["broker_order_id"] = str(getattr(order, "id", ""))
        return payload

    def submit_stop_order(
        self,
        *,
        symbol: str,
        quantity: float,
        stop_price: float,
        strategy_name: str,
        client_order_id: str | None = None,
    ) -> dict[str, object]:
        payload = {
            "mode": "paper",
            "broker": "alpaca",
            "base_url": self._settings.alpaca_paper_base_url,
            "symbol": symbol,
            "side": "sell",
            "quantity": quantity,
            "client_order_id": client_order_id or str(uuid4()),
            "strategy_name": strategy_name,
            "type": "stop",
            "stop_price": _round_broker_price(stop_price),
        }
        if self._client is None:
            logger.info("Prepared broker-native stop payload", extra={"payload": payload})
            return payload

        fractional = abs(float(quantity) - round(float(quantity))) > 1e-9
        request = StopOrderRequest(
            symbol=symbol,
            qty=float(quantity),
            side=OrderSide.SELL,
            type=OrderType.STOP,
            time_in_force=TimeInForce.DAY if fractional else TimeInForce.GTC,
            client_order_id=str(payload["client_order_id"]),
            stop_price=float(payload["stop_price"]),
        )
        order = self._client.submit_order(order_data=request)
        payload["submitted"] = True
        payload["broker_order_id"] = str(getattr(order, "id", ""))
        return payload
