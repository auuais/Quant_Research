"""Event and calendar data for the D2 premia battery: FOMC announcements, OPEX, month turns.

FOMC dates are scraped from federalreserve.gov rather than hardcoded, so the list is authoritative and
refreshes itself when the Fed publishes a new year.
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf


FOMC_CURRENT_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FOMC_HISTORICAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
STATEMENT_DATE_PATTERN = re.compile(r"monetary(\d{8})a")
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; algoding-research/1.0)"}
SCHEDULED_MEETINGS_PER_YEAR = 8


@dataclass(frozen=True)
class FomcCalendar:
    announcements: pd.DatetimeIndex
    per_year_counts: dict[int, int]
    years_with_unexpected_counts: list[int]
    source_urls: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "announcements": len(self.announcements),
            "first": self.announcements.min().date().isoformat() if len(self.announcements) else None,
            "last": self.announcements.max().date().isoformat() if len(self.announcements) else None,
            "per_year_counts": {str(year): count for year, count in sorted(self.per_year_counts.items())},
            "years_with_unexpected_counts": self.years_with_unexpected_counts,
            "unexpected_count_note": (
                "the Fed also publishes statements for unscheduled actions, so a year with more than "
                f"{SCHEDULED_MEETINGS_PER_YEAR} statement dates includes non-scheduled announcements"
            ),
            "source_urls": self.source_urls,
        }


class EventCalendarClient:
    def __init__(self, *, cache_dir: str = "cache/events", cache_max_age_hours: float = 168.0) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_max_age = timedelta(hours=cache_max_age_hours)

    def load_fomc_calendar(self, *, start_year: int = 2011, end_year: int | None = None) -> FomcCalendar:
        end_year = end_year or datetime.now(timezone.utc).year
        cached = self._read_cache("fomc_announcements")
        dates: list[pd.Timestamp] = []
        sources: list[str] = []
        if cached is not None:
            dates = list(pd.DatetimeIndex(cached["date"]))
            sources = ["cache"]
        else:
            raw: set[str] = set()
            for url in [FOMC_CURRENT_URL] + [
                FOMC_HISTORICAL_URL.format(year=year) for year in range(start_year, end_year + 1)
            ]:
                try:
                    body = _http_get(url)
                except Exception:
                    continue
                found = set(STATEMENT_DATE_PATTERN.findall(body))
                if found:
                    raw |= found
                    sources.append(url)
            dates = sorted(pd.Timestamp(datetime.strptime(value, "%Y%m%d")) for value in raw)
            if dates:
                self._write_cache("fomc_announcements", pd.DataFrame({"date": dates}))

        index = pd.DatetimeIndex([stamp for stamp in dates if start_year <= stamp.year <= end_year]).sort_values()
        counts = {int(year): int((index.year == year).sum()) for year in sorted(set(index.year))}
        unexpected = [
            year
            for year, count in counts.items()
            if count != SCHEDULED_MEETINGS_PER_YEAR and year < datetime.now(timezone.utc).year
        ]
        return FomcCalendar(
            announcements=index,
            per_year_counts=counts,
            years_with_unexpected_counts=unexpected,
            source_urls=sources,
        )

    def load_ohlc(self, symbols: list[str], *, start: str = "2005-01-01") -> dict[str, pd.DataFrame]:
        cache_key = "ohlc_" + "_".join(sorted(symbol.lower().replace("/", "_") for symbol in symbols))
        cached = self._read_cache(cache_key)
        if cached is not None:
            return {
                symbol: cached[cached["symbol"] == symbol].drop(columns=["symbol"]).set_index("date")
                for symbol in cached["symbol"].unique()
            }
        raw = yf.download(
            tickers=symbols,
            start=start,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if raw.empty:
            return {}
        out: dict[str, pd.DataFrame] = {}
        flat: list[pd.DataFrame] = []
        multi = isinstance(raw.columns, pd.MultiIndex)
        for symbol in symbols:
            try:
                frame = raw[symbol] if multi else raw
            except Exception:
                continue
            frame = frame[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "Close"])
            if frame.empty:
                continue
            frame = frame.rename(columns=str.lower)
            frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
            frame = frame[~frame.index.duplicated(keep="last")]
            out[symbol] = frame
            flat.append(frame.assign(symbol=symbol).reset_index(names="date"))
        if flat:
            self._write_cache(cache_key, pd.concat(flat, ignore_index=True))
        return out

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.parquet"

    def _read_cache(self, key: str) -> pd.DataFrame | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if age > self._cache_max_age:
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def _write_cache(self, key: str, frame: pd.DataFrame) -> None:
        try:
            frame.to_parquet(self._cache_path(key))
        except Exception:
            pass


def _http_get(url: str, *, timeout: float = 30.0) -> str:
    request = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def opex_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Monthly equity-option expiration: the third Friday, rolled back to the prior trading day if closed."""
    stamps: list[pd.Timestamp] = []
    for (year, month), group in pd.Series(index, index=index).groupby([index.year, index.month]):
        fridays = [stamp for stamp in group.index if stamp.weekday() == 4]
        if len(fridays) >= 3:
            stamps.append(fridays[2])
            continue
        third_friday = _nth_weekday(year, month, weekday=4, count=3)
        earlier = [stamp for stamp in group.index if stamp <= third_friday]
        if earlier:
            stamps.append(earlier[-1])
    return pd.DatetimeIndex(sorted(stamps))


def opex_week_flags(index: pd.DatetimeIndex) -> pd.Series:
    """True for the trading days from Monday of OPEX week through OPEX day."""
    flags = pd.Series(False, index=index)
    for expiration in opex_dates(index):
        week_start = expiration - pd.Timedelta(days=int(expiration.weekday()))
        flags.loc[(index >= week_start) & (index <= expiration)] = True
    return flags


def turn_of_month_flags(index: pd.DatetimeIndex, *, last_n: int = 4, first_n: int = 3) -> pd.Series:
    """True for the last `last_n` trading days of a month and the first `first_n` of the next."""
    flags = pd.Series(False, index=index)
    frame = pd.DataFrame({"stamp": index}, index=index)
    grouped = frame.groupby([index.year, index.month])["stamp"]
    for _, stamps in grouped:
        ordered = list(stamps)
        for stamp in ordered[:first_n] + ordered[-last_n:]:
            flags.loc[stamp] = True
    return flags


def quarter_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    stamps: list[pd.Timestamp] = []
    frame = pd.DataFrame({"stamp": index}, index=index)
    for (_, month), stamps_in_month in frame.groupby([index.year, index.quarter])["stamp"]:
        del month
        ordered = list(stamps_in_month)
        if ordered:
            stamps.append(ordered[-1])
    return pd.DatetimeIndex(sorted(stamps))


def _nth_weekday(year: int, month: int, *, weekday: int, count: int) -> pd.Timestamp:
    stamp = pd.Timestamp(year=year, month=month, day=1)
    found = 0
    while True:
        if stamp.weekday() == weekday:
            found += 1
            if found == count:
                return stamp
        stamp += pd.Timedelta(days=1)
