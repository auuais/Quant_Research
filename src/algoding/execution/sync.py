from __future__ import annotations

from datetime import datetime, timedelta, timezone

from alpaca.common.enums import Sort
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from algoding.execution.storage import OrderStore
from algoding.settings import Settings


class PaperBrokerSync:
    def __init__(self, settings: Settings) -> None:
        if not settings.has_alpaca_credentials:
            raise RuntimeError("Alpaca credentials are required in .env for paper sync.")
        self._settings = settings
        self._client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=True,
        )
        self._store = OrderStore(settings)

    def run(self, days: int = 7) -> dict[str, object]:
        self._store.ensure_schema()
        orders = self._client.get_orders(
            filter=GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                after=datetime.now(timezone.utc) - timedelta(days=days),
                direction=Sort.DESC,
                limit=100,
            )
        )
        positions = self._client.get_all_positions()

        synced_orders = 0
        for order in orders:
            payload = {
                "client_order_id": order.client_order_id,
                "strategy_name": "alpaca_sync",
                "symbol": order.symbol,
                "side": str(order.side).split(".")[-1].lower() if order.side else "",
                "quantity": float(order.qty) if order.qty not in (None, "None") else 0.0,
                "notional": float(order.notional) if order.notional not in (None, "None") else None,
                "status": str(order.status).split(".")[-1].lower(),
                "broker_order_id": str(order.id),
                "created_at": order.created_at.isoformat(),
                "updated_at": order.updated_at.isoformat() if order.updated_at else None,
                "filled_at": order.filled_at.isoformat() if order.filled_at else None,
                "filled_qty": float(order.filled_qty) if order.filled_qty not in (None, "None") else 0.0,
                "filled_avg_price": float(order.filled_avg_price)
                if order.filled_avg_price not in (None, "None")
                else None,
            }
            self._store.sync_broker_order(payload)
            synced_orders += 1

        synced_positions = 0
        for position in positions:
            payload = {
                "symbol": position.symbol,
                "qty": float(position.qty),
                "avg_entry_price": float(position.avg_entry_price),
                "current_price": float(position.current_price) if position.current_price else None,
                "market_value": float(position.market_value) if position.market_value else None,
                "unrealized_pl": float(position.unrealized_pl) if position.unrealized_pl else None,
                "change_today": float(position.change_today) if position.change_today else None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._store.sync_position(payload)
            synced_positions += 1

        return {"orders_synced": synced_orders, "positions_synced": synced_positions}
