"""
Naive random-walk benchmark.

Point forecast = last observed value carried forward ("next week = this week").
For weekly commodity prices this is a deliberately strong baseline: in
walk-forward tests it is hard to beat at multi-week horizons, so it is the
honest reference every model should be compared against.

Prediction interval: a random walk in log-price, with per-step variance equal
to the recent (~2 year) log-return variance, so the band widens like sqrt(h).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import ForecastResult, future_index


def fit_forecast(y: pd.Series, horizon: int, alpha: float = 0.05,
                 product: str = "") -> ForecastResult:
    y = y.dropna()
    last = float(y.iloc[-1])
    logp = np.log(y)
    ret = logp.diff().dropna()
    recent = ret.iloc[-min(104, len(ret)):]
    sigma = float(np.std(recent.values, ddof=1)) if len(recent) > 2 else 0.0

    idx = future_index(y, horizon, "7D")
    h = np.arange(1, horizon + 1)
    se = sigma * np.sqrt(h)
    z = stats.norm.ppf(1 - alpha / 2)

    mean = pd.Series(last, index=idx)
    lower = pd.Series(last * np.exp(-z * se), index=idx)
    upper = pd.Series(last * np.exp(z * se), index=idx)
    # In-sample "fitted" of a random walk is just the previous observation.
    fitted = y.shift(1).dropna().rename(product)

    return ForecastResult(
        method="Naive", product=product, mean=mean, lower=lower, upper=upper,
        fitted=fitted, spec="Naive random walk (last value carried forward)",
        diagnostics={"sigma_weekly_logret": sigma}, alpha=alpha,
    )
