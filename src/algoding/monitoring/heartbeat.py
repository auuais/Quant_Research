from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class HeartbeatMonitor:
    timeout_seconds: int
    last_seen: datetime | None = None

    def mark_seen(self) -> None:
        self.last_seen = datetime.now(timezone.utc)

    def is_stale(self) -> bool:
        if self.last_seen is None:
            return True
        return datetime.now(timezone.utc) - self.last_seen > timedelta(seconds=self.timeout_seconds)
