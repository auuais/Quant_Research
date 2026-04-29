from __future__ import annotations

from datetime import datetime, timedelta, timezone

from alpaca.data.enums import Adjustment, CryptoFeed, DataFeed
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from algoding.settings import Settings


class AlpacaHistoricalClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.has_alpaca_credentials:
            raise RuntimeError("Alpaca credentials are required in .env for historical data.")
        self._settings = settings
        self._client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            raw_data=False,
        )
        self._crypto_client = CryptoHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            raw_data=False,
        )

    def get_recent_daily_closes(
        self, symbols: list[str], lookback_bars: int
    ) -> dict[str, list[float]]:
        bars_by_symbol = self.get_recent_daily_bars(symbols, lookback_bars)
        return {
            symbol: [float(bar.close) for bar in bars]
            for symbol, bars in bars_by_symbol.items()
        }

    def get_recent_daily_bars(self, symbols: list[str], lookback_bars: int) -> dict[str, list[object]]:
        return self.get_recent_bars(symbols, lookback_bars, timeframe="day")

    def get_recent_hourly_bars(self, symbols: list[str], lookback_bars: int) -> dict[str, list[object]]:
        return self.get_recent_bars(symbols, lookback_bars, timeframe="hour")

    def get_recent_minute_bars(self, symbols: list[str], lookback_bars: int) -> dict[str, list[object]]:
        return self.get_recent_bars(symbols, lookback_bars, timeframe="minute")

    def get_recent_bars(
        self,
        symbols: list[str],
        lookback_bars: int,
        timeframe: str = "day",
    ) -> dict[str, list[object]]:
        normalized = timeframe.strip().lower()
        if normalized == "day":
            timeframe_value = TimeFrame.Day
        elif normalized == "hour":
            timeframe_value = TimeFrame.Hour
        elif normalized == "minute":
            timeframe_value = TimeFrame.Minute
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        start = datetime.now(timezone.utc) - timedelta(
            days=self._estimate_calendar_days(lookback_bars=lookback_bars, timeframe=normalized)
        )
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe_value,
            start=start,
            end=datetime.now(timezone.utc),
            adjustment=Adjustment.ALL,
            feed=DataFeed(self._settings.alpaca_data_feed.lower()),
        )
        response = self._client.get_stock_bars(request)
        bars_by_symbol: dict[str, list[object]] = {}
        for symbol, bars in response.data.items():
            bars_by_symbol[symbol] = list(bars)[-lookback_bars:]
        return bars_by_symbol

    def get_recent_crypto_bars(
        self,
        symbols: list[str],
        lookback_bars: int,
        timeframe: str = "day",
    ) -> dict[str, list[object]]:
        normalized = timeframe.strip().lower()
        timeframe_value = TimeFrame.Day if normalized == "day" else TimeFrame.Hour
        start = datetime.now(timezone.utc) - timedelta(
            days=self._estimate_calendar_days(lookback_bars=lookback_bars, timeframe=normalized)
        )
        request = CryptoBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe_value,
            start=start,
            end=datetime.now(timezone.utc),
        )
        response = self._crypto_client.get_crypto_bars(request, feed=CryptoFeed.US)
        bars_by_symbol: dict[str, list[object]] = {}
        for symbol, bars in response.data.items():
            bars_by_symbol[symbol] = list(bars)[-lookback_bars:]
        return bars_by_symbol

    @staticmethod
    def _estimate_calendar_days(lookback_bars: int, timeframe: str) -> int:
        if timeframe == "minute":
            trading_minutes_per_day = 390.0
            trading_days = max(1.0, lookback_bars / trading_minutes_per_day)
            calendar_days = trading_days * (7.0 / 5.0) * 1.25
            return max(int(calendar_days) + 5, 7)
        if timeframe == "hour":
            trading_hours_per_day = 7.0
            trading_days = max(1.0, lookback_bars / trading_hours_per_day)
            calendar_days = trading_days * (7.0 / 5.0) * 1.25
            return max(int(calendar_days) + 10, 14)
        return max(lookback_bars * 3, 30)
