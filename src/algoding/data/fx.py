from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import httpx


@dataclass(frozen=True)
class FxBar:
    symbol: str
    close: float
    timestamp: str


class FrankfurterFxHistoricalClient:
    _BASE_URL = "https://api.frankfurter.app"
    _PAIR_CONFIG = {
        "EURUSD": ("EUR", "USD"),
        "USDJPY": ("USD", "JPY"),
        "GBPUSD": ("GBP", "USD"),
    }

    def get_recent_daily_bars(self, symbols: list[str], lookback_bars: int) -> dict[str, list[FxBar]]:
        results: dict[str, list[FxBar]] = {}
        for symbol in symbols:
            normalized = symbol.strip().upper()
            if normalized not in self._PAIR_CONFIG:
                raise ValueError(f"Unsupported FX symbol: {normalized}")
            base_currency, quote_currency = self._PAIR_CONFIG[normalized]
            start_date = date.today() - timedelta(days=max(lookback_bars * 3, 90))
            url = (
                f"{self._BASE_URL}/{start_date.isoformat()}..{date.today().isoformat()}"
                f"?from={base_currency}&to={quote_currency}"
            )
            response = httpx.get(url, timeout=30.0)
            response.raise_for_status()
            payload = response.json()
            rates = payload.get("rates", {})
            bars = [
                FxBar(symbol=normalized, close=float(values[quote_currency]), timestamp=f"{day}T00:00:00+00:00")
                for day, values in sorted(rates.items())
                if quote_currency in values
            ]
            results[normalized] = bars[-lookback_bars:]
        return results
