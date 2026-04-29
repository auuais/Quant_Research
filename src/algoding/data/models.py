from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass
class MarketEvent:
    symbol: str
    event_type: str
    price: float
    volume: float
    event_ts: datetime
    received_ts: datetime

    @classmethod
    def sample(cls, symbol: str = "SPY", price: float = 500.0) -> "MarketEvent":
        now = datetime.now(timezone.utc)
        return cls(
            symbol=symbol,
            event_type="bar",
            price=price,
            volume=1000.0,
            event_ts=now,
            received_ts=now,
        )

    def to_line_protocol(self) -> str:
        event_ns = int(self.event_ts.timestamp() * 1_000_000_000)
        return (
            f"market_events,symbol={self.symbol},event_type={self.event_type} "
            f"price={self.price},volume={self.volume},"
            f"received_ts_ns={int(self.received_ts.timestamp() * 1_000_000_000)} {event_ns}"
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_serializable_dict(self) -> dict[str, object]:
        payload = self.to_dict()
        payload["event_ts"] = self.event_ts.isoformat()
        payload["received_ts"] = self.received_ts.isoformat()
        return payload
