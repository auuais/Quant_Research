from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class YahooBar:
    timestamp: datetime
    open: float
    close: float
    high: float
    low: float
    volume: float | None


class YahooHistoricalClient:
    def get_recent_daily_bars(self, symbols: list[str], lookback_bars: int) -> dict[str, list[YahooBar]]:
        if not symbols:
            return {}
        start = datetime.now(timezone.utc) - timedelta(days=max(365 * 20, lookback_bars * 3))
        frame = yf.download(
            tickers=symbols,
            start=start.date().isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if frame.empty:
            return {}

        bars_by_symbol: dict[str, list[YahooBar]] = {}
        multi_symbol = isinstance(frame.columns, pd.MultiIndex)
        for symbol in symbols:
            if multi_symbol:
                if symbol not in frame.columns.get_level_values(0):
                    continue
                symbol_frame = frame[symbol].dropna(how="any")
            else:
                symbol_frame = frame.dropna(how="any")
            if symbol_frame.empty:
                continue

            trimmed = symbol_frame.tail(lookback_bars)
            bars_by_symbol[symbol] = [
                YahooBar(
                    timestamp=index.to_pydatetime().replace(tzinfo=timezone.utc),
                    open=float(row["Open"]),
                    close=float(row["Close"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    volume=float(row["Volume"]) if "Volume" in row and pd.notna(row["Volume"]) else None,
                )
                for index, row in trimmed.iterrows()
            ]
        return bars_by_symbol
