"""Volatility-complex data access: CBOE cash indices plus VIX-ETP derived futures-index returns."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import yfinance as yf


CBOE_INDEX_URLS = {
    "vix": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "vix9d": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv",
    "vix3m": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
    "vix6m": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX6M_History.csv",
    "vvix": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv",
}
YAHOO_INDEX_FALLBACK = {"vix": "^VIX", "vix9d": "^VIX9D", "vix3m": "^VIX3M", "vix6m": "^VIX6M", "vvix": "^VVIX"}

# ProShares cut SVXY's target exposure from -1x to -0.5x effective 2018-02-28; UVXY went +2x -> +1.5x the same day.
# Verified empirically in _leverage_diagnostics (rolling beta vs VXX) rather than trusted blindly.
SVXY_LEVERAGE_CHANGE = date(2018, 2, 28)
ETP_SYMBOLS = ["VXX", "SVXY", "UVXY", "SVIX", "BIL", "SPY"]
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; algoding-research/1.0)"}


@dataclass(frozen=True)
class VolatilityPanel:
    frame: pd.DataFrame
    provenance: dict[str, object] = field(default_factory=dict)

    @property
    def start(self) -> str:
        return self.frame.index.min().date().isoformat() if len(self.frame) else ""

    @property
    def end(self) -> str:
        return self.frame.index.max().date().isoformat() if len(self.frame) else ""


class VolatilityDataClient:
    def __init__(self, *, cache_dir: str = "cache/volatility", cache_max_age_hours: float = 18.0) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_max_age = timedelta(hours=cache_max_age_hours)

    def load_cboe_index(self, name: str) -> pd.Series:
        key = name.lower()
        if key not in CBOE_INDEX_URLS:
            raise KeyError(f"unknown CBOE index: {name}")
        cached = self._read_cache(f"cboe_{key}")
        if cached is not None:
            return cached[key]

        series: pd.Series | None = None
        try:
            series = _parse_cboe_csv(_http_get(CBOE_INDEX_URLS[key]), key)
        except Exception:
            series = None
        if series is None or series.empty:
            series = _yahoo_close(YAHOO_INDEX_FALLBACK[key])
        series = series.rename(key).sort_index()
        self._write_cache(f"cboe_{key}", series.to_frame())
        return series

    def load_etp_closes(self, symbols: list[str] | None = None) -> pd.DataFrame:
        wanted = symbols or ETP_SYMBOLS
        cache_key = "etp_" + "_".join(sorted(symbol.lower() for symbol in wanted))
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached
        frame = _yahoo_closes(wanted)
        self._write_cache(cache_key, frame)
        return frame

    def build_panel(self, *, start: str = "2011-01-01") -> VolatilityPanel:
        indices = {}
        index_sources: dict[str, str] = {}
        for name in ("vix", "vix9d", "vix3m", "vix6m", "vvix"):
            try:
                series = self.load_cboe_index(name)
            except Exception:
                continue
            if series is not None and not series.empty:
                indices[name] = series
                index_sources[name] = f"{series.index.min().date()} -> {series.index.max().date()} ({len(series)} rows)"
        etps = self.load_etp_closes()

        frame = pd.DataFrame(indices)
        frame = frame.join(etps, how="outer").sort_index()
        frame = frame.loc[frame.index >= pd.Timestamp(start)]

        index_returns, provenance = _reconstruct_futures_index_returns(etps)
        frame["vix_futures_index_return"] = index_returns
        frame["cash_return"] = etps["BIL"].pct_change() if "BIL" in etps else 0.0
        frame["spy_return"] = etps["SPY"].pct_change() if "SPY" in etps else 0.0

        for symbol in ("SVXY", "SVIX", "VXX", "UVXY"):
            if symbol in etps:
                frame[f"{symbol.lower()}_return"] = etps[symbol].pct_change()
        short_vol, short_vol_instrument, short_vol_notes = _real_short_vol_returns(etps)
        long_vol, long_vol_instrument, long_vol_notes = _real_long_vol_returns(etps)
        frame["short_vol_return"] = short_vol
        frame["short_vol_instrument"] = short_vol_instrument
        frame["long_vol_return"] = long_vol
        frame["long_vol_instrument"] = long_vol_instrument
        provenance["tradeable_legs"] = {
            "short_vol": short_vol_notes,
            "long_vol": long_vol_notes,
            "why_real_not_synthetic": (
                "Synthetic leverage rescaling was tested and rejected: a -1x series rebuilt from VXX loses "
                "48.7% over 2018-02-01..2018-02-12 while the real -1x product (SVXY) lost 90.7%. VIX futures "
                "moved violently between the 16:00 equity close and the 16:15 futures settlement, so ETPs "
                "striking NAV at different times are not interchangeable in exactly the crises that decide "
                "this sleeve. Backtests therefore use the real closes of the product actually holdable."
            ),
        }

        # The cash indices publish with a short lag; forward-fill signal inputs but never returns.
        for column in ("vix", "vix9d", "vix3m", "vix6m", "vvix"):
            if column in frame:
                frame[column] = frame[column].ffill(limit=5)

        frame = frame.dropna(subset=["vix", "vix_futures_index_return"])
        return VolatilityPanel(
            frame=frame,
            provenance={
                "cboe_indices": index_sources,
                "etp_coverage": {
                    symbol: f"{etps[symbol].dropna().index.min().date()} -> {etps[symbol].dropna().index.max().date()}"
                    for symbol in etps.columns
                    if etps[symbol].notna().any()
                },
                "futures_index_reconstruction": provenance,
                "panel_rows": int(len(frame)),
            },
        )

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


def _parse_cboe_csv(body: str, name: str) -> pd.Series:
    frame = pd.read_csv(StringIO(body))
    date_column = frame.columns[0]
    close_column = "CLOSE" if "CLOSE" in frame.columns else frame.columns[-1]
    frame[date_column] = pd.to_datetime(frame[date_column], format="mixed", errors="coerce")
    frame = frame.dropna(subset=[date_column]).set_index(date_column).sort_index()
    series = pd.to_numeric(frame[close_column], errors="coerce").dropna()
    series.index = series.index.tz_localize(None).normalize()
    return series[~series.index.duplicated(keep="last")].rename(name)


def _yahoo_closes(symbols: list[str]) -> pd.DataFrame:
    raw = yf.download(
        tickers=symbols,
        start="2005-01-01",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()
    out: dict[str, pd.Series] = {}
    multi = isinstance(raw.columns, pd.MultiIndex)
    for symbol in symbols:
        try:
            series = raw[symbol]["Close"] if multi else raw["Close"]
        except Exception:
            continue
        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty:
            continue
        series.index = pd.DatetimeIndex(series.index).tz_localize(None).normalize()
        out[symbol] = series[~series.index.duplicated(keep="last")]
    return pd.DataFrame(out).sort_index()


def _yahoo_close(symbol: str) -> pd.Series:
    frame = _yahoo_closes([symbol])
    if frame.empty:
        return pd.Series(dtype=float)
    return frame[symbol]


def _reconstruct_futures_index_returns(etps: pd.DataFrame) -> tuple[pd.Series, dict[str, object]]:
    """Rebuild short-term VIX futures index daily returns from whichever ETP is available.

    VXX is +1x so it needs no leverage assumption and is preferred wherever it exists. Before VXX
    (2018-01) the only source is SVXY, whose target exposure changed from -1x to -0.5x on
    2018-02-28; UVXY provides an independent cross-check.
    """
    returns = {symbol: etps[symbol].pct_change() for symbol in etps.columns if symbol in {"VXX", "SVXY", "UVXY", "SVIX"}}
    svxy_leverage = pd.Series(1.0, index=etps.index)
    svxy_leverage.loc[svxy_leverage.index >= pd.Timestamp(SVXY_LEVERAGE_CHANGE)] = 0.5
    uvxy_leverage = pd.Series(2.0, index=etps.index)
    uvxy_leverage.loc[uvxy_leverage.index >= pd.Timestamp(SVXY_LEVERAGE_CHANGE)] = 1.5

    from_vxx = returns.get("VXX")
    from_svxy = -returns["SVXY"] / svxy_leverage if "SVXY" in returns else None
    from_uvxy = returns["UVXY"] / uvxy_leverage if "UVXY" in returns else None

    candidates = [series for series in (from_vxx, from_svxy, from_uvxy) if series is not None]
    if not candidates:
        raise RuntimeError("no VIX ETP available to reconstruct futures-index returns")

    combined = pd.Series(index=etps.index, dtype=float)
    source = pd.Series(index=etps.index, dtype=object)
    for series, label in ((from_uvxy, "uvxy_derived"), (from_svxy, "svxy_derived"), (from_vxx, "vxx_direct")):
        if series is None:
            continue
        mask = series.notna() & combined.isna() if label != "vxx_direct" else series.notna()
        combined.loc[mask] = series.loc[mask]
        source.loc[mask] = label

    diagnostics = _leverage_diagnostics(from_vxx, from_svxy, from_uvxy)
    provenance = {
        "method": (
            "index return = VXX return where available (+1x, no leverage assumption); "
            "otherwise -SVXY/leverage, else UVXY/leverage. SVXY leverage 1.0 before "
            f"{SVXY_LEVERAGE_CHANGE.isoformat()}, 0.5 after; UVXY 2.0 then 1.5."
        ),
        "source_day_counts": {str(k): int(v) for k, v in source.value_counts().items()},
        "leverage_diagnostics": diagnostics,
        "caveat": (
            "Pre-2018 index returns are derived from a leveraged ETP, so they inherit its tracking error "
            "and fee drag. Daily-rebalanced leverage rescaling is exact for daily returns but not for "
            "multi-day compounding of the source product."
        ),
    }
    return combined, provenance


def _real_short_vol_returns(etps: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict[str, object]]:
    """Daily returns of the short-vol product actually holdable on each date.

    SVXY spans the whole sample but its target exposure fell from -1x to -0.5x on 2018-02-28; SVIX is
    the modern -1x product from 2022-03-30. Returns are the product's real closes, so the -1x-era
    crash losses are genuine rather than rescaled.
    """
    if "SVXY" not in etps:
        raise RuntimeError("SVXY is required for the short-vol leg")
    svxy = etps["SVXY"].pct_change()
    returns = svxy.copy()
    instrument = pd.Series(index=etps.index, dtype=object)
    instrument.loc[svxy.notna()] = "SVXY"
    exposure = pd.Series(index=etps.index, dtype=float)
    exposure.loc[svxy.notna() & (etps.index < pd.Timestamp(SVXY_LEVERAGE_CHANGE))] = -1.0
    exposure.loc[svxy.notna() & (etps.index >= pd.Timestamp(SVXY_LEVERAGE_CHANGE))] = -0.5
    notes = {
        "policy": "hold SVXY throughout (real closes); target exposure -1x before 2018-02-28, -0.5x after",
        "alternative_policy": "svix_when_available (SVIX from 2022-03-30, else SVXY) is reported as a robustness check",
        "svxy_days": int(svxy.notna().sum()),
        "minus_1x_era_days": int((exposure == -1.0).sum()),
        "minus_0p5x_era_days": int((exposure == -0.5).sum()),
    }
    return returns, instrument, notes


def _real_long_vol_returns(etps: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict[str, object]]:
    """Daily returns of the long-vol product actually holdable: VXX where it exists, else UVXY/1.5-2x.

    UVXY is a leveraged product, so the pre-2018 long-vol leg is deleveraged rather than real. The
    long-vol leg is only used by the optional backwardation variants; results depending on it are
    flagged in the report.
    """
    vxx = etps["VXX"].pct_change() if "VXX" in etps else None
    uvxy = etps["UVXY"].pct_change() if "UVXY" in etps else None
    returns = pd.Series(index=etps.index, dtype=float)
    instrument = pd.Series(index=etps.index, dtype=object)
    if uvxy is not None:
        leverage = pd.Series(2.0, index=etps.index)
        leverage.loc[leverage.index >= pd.Timestamp(SVXY_LEVERAGE_CHANGE)] = 1.5
        deleveraged = uvxy / leverage
        mask = deleveraged.notna()
        returns.loc[mask] = deleveraged.loc[mask]
        instrument.loc[mask] = "UVXY_deleveraged"
    if vxx is not None:
        mask = vxx.notna()
        returns.loc[mask] = vxx.loc[mask]
        instrument.loc[mask] = "VXX"
    notes = {
        "policy": "VXX real closes where available (2018-01-25+); before that UVXY deleveraged by its 2x target",
        "caveat": "pre-2018 long-vol returns are synthetic (deleveraged UVXY) and inherit the crisis-timing problem",
        "vxx_days": int(vxx.notna().sum()) if vxx is not None else 0,
    }
    return returns, instrument, notes


def _leverage_diagnostics(
    from_vxx: pd.Series | None,
    from_svxy: pd.Series | None,
    from_uvxy: pd.Series | None,
) -> dict[str, object]:
    """Empirically confirm the assumed ETP leverage by regressing derived series against VXX."""
    if from_vxx is None:
        return {"note": "VXX unavailable; leverage assumptions unverified"}
    out: dict[str, object] = {}
    for label, series in (("svxy_derived", from_svxy), ("uvxy_derived", from_uvxy)):
        if series is None:
            continue
        joined = pd.concat({"vxx": from_vxx, "derived": series}, axis=1).dropna()
        if len(joined) < 60:
            continue
        beta = float(joined["derived"].cov(joined["vxx"]) / joined["vxx"].var())
        out[label] = {
            "overlap_days": int(len(joined)),
            "beta_vs_vxx": round(beta, 4),
            "correlation": round(float(joined["derived"].corr(joined["vxx"])), 4),
            "reads_as_expected": bool(abs(beta - 1.0) < 0.15),
        }
    return out
