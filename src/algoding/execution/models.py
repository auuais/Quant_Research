from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from uuid import uuid4


@dataclass
class OrderIntent:
    symbol: str
    side: str
    quantity: float
    notional: float | None
    strategy_name: str
    created_at: datetime
    client_order_id: str
    reason: str = "signal_entry"

    @classmethod
    def sample(cls, symbol: str = "SPY", side: str = "buy") -> "OrderIntent":
        return cls(
            symbol=symbol,
            side=side,
            quantity=1,
            notional=None,
            strategy_name="etf_momentum_v1",
            created_at=datetime.now(timezone.utc),
            client_order_id=str(uuid4()),
            reason="signal_entry",
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
