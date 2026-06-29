"""Forecasting models for the fertilizer price pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ForecastResult:
    """Uniform forecast container shared by every method.

    All series are in **price level (USD/Ton)** space, regardless of whether the
    model worked internally on logs / returns.
    """

    method: str
    product: str
    mean: pd.Series                       # point forecast over the horizon
    lower: pd.Series                      # lower prediction-interval bound
    upper: pd.Series                      # upper prediction-interval bound
    fitted: pd.Series | None = None       # in-sample fitted values (level)
    spec: str = ""                        # human-readable model spec
    diagnostics: dict = field(default_factory=dict)
    alpha: float = 0.05                   # 1 - confidence level


def future_index(history: pd.Series, horizon: int,
                 freq: str = "7D") -> pd.DatetimeIndex:
    """Build the future date index immediately following ``history``."""
    start = history.index[-1] + pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(start=start, periods=horizon, freq=freq)


def shrink_toward_naive(fc: "ForecastResult", last_value: float,
                        s: float) -> "ForecastResult":
    """Pull a forecast's point path toward the random walk (last observed value).

    ``s`` in [0, 1]: 0 = pure model, 1 = pure naive. Backtests on these weekly
    commodity series show the models slightly *over-extrapolate* recent momentum
    that mean-reverts, so a partial shrink toward "next week = this week"
    measurably lowers RMSE/MAE/MAPE/sMAPE. The prediction interval is shifted by
    the same amount, preserving its (volatility-aware) width around the new mean.
    """
    if not s or s <= 0:
        return fc
    s = min(float(s), 1.0)
    new_mean = (1 - s) * fc.mean + s * last_value
    delta = new_mean - fc.mean
    fc.mean = new_mean
    fc.lower = fc.lower + delta
    fc.upper = fc.upper + delta
    fc.diagnostics["shrink"] = s
    return fc
