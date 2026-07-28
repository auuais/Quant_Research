"""Volatility-sleeve signals: HAR-RV realized-variance forecasting, term structure, and VRP.

Every series here is built so the value stamped on day T uses only information available at T's close,
matching the FACTOR-VALIDATION timing convention (decide at T close, earn T+1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS = 252.0
HAR_LAGS = (1, 5, 22)


@dataclass(frozen=True)
class HarForecast:
    forecast: pd.Series
    diagnostics: dict[str, object]


def realized_variance(returns: pd.Series, window: int) -> pd.Series:
    """Annualized realized variance over a trailing window, known at the window's last close."""
    return (returns.pow(2).rolling(window).mean()) * TRADING_DAYS


def har_rv_forecast(
    returns: pd.Series,
    *,
    horizon: int = 21,
    min_train: int = 504,
    refit_every: int = 21,
) -> HarForecast:
    """Walk-forward HAR-RV forecast of average annualized variance over the next `horizon` days.

    The HAR design regresses forward realized variance on trailing daily/weekly/monthly realized
    variance. Coefficients are refit on an expanding window; a fit made at T only sees targets whose
    forward window closed on or before T, so no future variance leaks into the forecast.
    """
    returns = returns.dropna()
    features = pd.DataFrame(
        {f"rv_{lag}": realized_variance(returns, lag) for lag in HAR_LAGS},
        index=returns.index,
    )
    forward = realized_variance(returns, horizon).shift(-horizon)

    design = features.dropna()
    forecast = pd.Series(index=returns.index, dtype=float)
    fits: list[dict[str, object]] = []
    coefficients: np.ndarray | None = None
    positions = list(range(len(design)))
    for offset, position in enumerate(positions):
        stamp = design.index[position]
        # Only targets whose forward window has fully closed by `stamp` are usable for fitting.
        usable = forward.loc[:stamp].dropna()
        usable = usable.loc[usable.index <= stamp - pd.Timedelta(days=1)]
        train_rows = design.loc[design.index.isin(usable.index)]
        if len(train_rows) >= min_train and (coefficients is None or offset % refit_every == 0):
            target = usable.loc[train_rows.index]
            matrix = np.column_stack([np.ones(len(train_rows)), train_rows.to_numpy()])
            coefficients, *_ = np.linalg.lstsq(matrix, target.to_numpy(), rcond=None)
            fits.append({"as_of": stamp.date().isoformat(), "train_rows": int(len(train_rows))})
        if coefficients is not None:
            row = np.concatenate([[1.0], design.iloc[position].to_numpy()])
            forecast.loc[stamp] = max(float(row @ coefficients), 1e-6)

    realized_check = pd.concat({"forecast": forecast, "actual": forward}, axis=1).dropna()
    diagnostics: dict[str, object] = {
        "horizon_days": horizon,
        "min_train_rows": min_train,
        "refit_every": refit_every,
        "refits": len(fits),
        "first_forecast": forecast.dropna().index.min().date().isoformat() if forecast.notna().any() else None,
        "forecast_rows": int(forecast.notna().sum()),
    }
    if len(realized_check) > 30:
        errors = realized_check["forecast"] - realized_check["actual"]
        baseline = realized_check["actual"].expanding().mean().shift(1)
        baseline_frame = pd.concat({"baseline": baseline, "actual": realized_check["actual"]}, axis=1).dropna()
        diagnostics.update(
            {
                "oos_correlation_forecast_vs_actual": round(
                    float(realized_check["forecast"].corr(realized_check["actual"])), 4
                ),
                "rmse": round(float(np.sqrt((errors**2).mean())), 6),
                "mean_bias": round(float(errors.mean()), 6),
                "beats_expanding_mean_rmse": bool(
                    len(baseline_frame) > 30
                    and np.sqrt((errors.loc[baseline_frame.index] ** 2).mean())
                    < np.sqrt(((baseline_frame["baseline"] - baseline_frame["actual"]) ** 2).mean())
                ),
            }
        )
    return HarForecast(forecast=forecast, diagnostics=diagnostics)


def build_signals(frame: pd.DataFrame, *, percentile_window: int = 504) -> tuple[pd.DataFrame, dict[str, object]]:
    """Term-structure, VRP, and guard features for the vol sleeve."""
    signals = pd.DataFrame(index=frame.index)

    signals["slope_3m"] = frame["vix3m"] / frame["vix"] - 1.0
    if "vix9d" in frame:
        signals["slope_9d"] = frame["vix"] / frame["vix9d"] - 1.0
    if "vix6m" in frame:
        signals["slope_6m"] = frame["vix6m"] / frame["vix"] - 1.0

    har = har_rv_forecast(frame["spy_return"].dropna())
    implied_variance = (frame["vix"] / 100.0) ** 2
    signals["har_forecast_variance"] = har.forecast
    signals["implied_variance"] = implied_variance
    signals["vrp"] = implied_variance - har.forecast
    signals["vrp_positive"] = signals["vrp"] > 0

    signals["vix_percentile"] = _rolling_percentile(frame["vix"], percentile_window)
    if "vvix" in frame:
        signals["vvix_percentile"] = _rolling_percentile(frame["vvix"], percentile_window)
    signals["vix_trend_5d"] = frame["vix"] / frame["vix"].shift(5) - 1.0
    signals["realized_vol_21d"] = np.sqrt(realized_variance(frame["spy_return"], 21))

    diagnostics = {
        "har_rv": har.diagnostics,
        "slope_contango_fraction": round(float((signals["slope_3m"] > 0).mean()), 4),
        "vrp_positive_fraction": round(float(signals["vrp"].dropna().gt(0).mean()), 4),
        "signal_rows": int(len(signals.dropna(subset=["slope_3m"]))),
        "percentile_window": percentile_window,
    }
    return signals, diagnostics


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=max(60, window // 4)).rank(pct=True)
