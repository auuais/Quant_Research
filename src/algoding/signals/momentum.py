from __future__ import annotations

from algoding.execution.models import OrderIntent


def generate_sample_signal(symbol: str = "SPY") -> OrderIntent:
    return OrderIntent.sample(symbol=symbol, side="buy")
