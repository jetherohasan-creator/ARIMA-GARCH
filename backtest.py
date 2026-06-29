"""
Walk-forward (rolling-origin) backtesting and accuracy metrics.

For each origin we refit the model on data up to the origin and forecast the
next ``test_h`` steps, then score against the held-out actuals. Metrics are
averaged across origins.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config
from models import arima_garch, linreg, naive, sarima, shrink_toward_naive

log = logging.getLogger("backtest")

SHRINK_METHODS = {"SARIMA", "ARIMA-GARCH"}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rmse(a, f): return float(np.sqrt(np.mean((a - f) ** 2)))
def mae(a, f): return float(np.mean(np.abs(a - f)))


def mape(a, f):
    mask = a != 0
    return float(np.mean(np.abs((a[mask] - f[mask]) / a[mask])) * 100)


def smape(a, f):
    denom = (np.abs(a) + np.abs(f))
    mask = denom != 0
    return float(np.mean(2 * np.abs(f[mask] - a[mask]) / denom[mask]) * 100)


def coverage(a, lo, hi):
    return float(np.mean((a >= lo) & (a <= hi)) * 100)


def _score(actual: pd.Series, fc) -> dict:
    a = actual.values
    f = fc.mean.reindex(actual.index).values
    lo = fc.lower.reindex(actual.index).values
    hi = fc.upper.reindex(actual.index).values
    # isfinite (not just ~isnan) so a divergent inf forecast is excluded, not
    # propagated into an inf metric.
    m = np.isfinite(f) & np.isfinite(a)
    if m.sum() == 0:
        return {}
    a, f, lo, hi = a[m], f[m], lo[m], hi[m]
    return {
        "RMSE": rmse(a, f), "MAE": mae(a, f), "MAPE": mape(a, f),
        "sMAPE": smape(a, f), "coverage": coverage(a, lo, hi),
    }


# --------------------------------------------------------------------------- #
# Model dispatch (fast settings for repeated refits)
# --------------------------------------------------------------------------- #
def _forecast(method: str, train: pd.Series, h: int, product: str,
              args, materials=None):
    shrink = getattr(args, "shrink", 0.0)
    last_value = float(train.dropna().iloc[-1])
    if method == "Naive":
        return naive.fit_forecast(train, h, product=product)
    if method == "SARIMA":
        fc = sarima.fit_forecast(train, h, freq=args.sarima_freq,
                                 product=product)
        return shrink_toward_naive(fc, last_value, shrink) \
            if args.sarima_freq != "monthly" else fc
    if method == "ARIMA-GARCH":
        fc = arima_garch.fit_forecast(train, h, product=product,
                                      vol=args.garch_vol)
        return shrink_toward_naive(fc, last_value, shrink)
    if method == "LinearRegression":
        mat = None
        if args.lr_mode == "materials" and materials:
            mat = {k: v.reindex(train.index) for k, v in materials.items()}
        return linreg.fit_forecast(train, h, mode=args.lr_mode,
                                   product=product, materials=mat)
    raise ValueError(method)


def walk_forward(y: pd.Series, method: str, product: str, args,
                 materials=None, n_origins: int = 6, test_h: int = 13) -> dict:
    """Rolling-origin CV. Returns averaged metrics + per-origin detail."""
    s = y.dropna()
    n = len(s)
    test_h = min(test_h, max(4, args.horizon // 2))
    # Reserve the last (n_origins-1)*step + test_h points for testing.
    min_train = max(60, int(n * 0.55))
    span = n - min_train - test_h
    if span <= 0:
        n_origins = 1
        step = 0
    else:
        step = max(1, span // max(1, n_origins - 1))

    per_origin = []
    for i in range(n_origins):
        cut = min_train + i * step
        if cut + test_h > n:
            break
        train = s.iloc[:cut]
        actual = s.iloc[cut:cut + test_h]
        try:
            fc = _forecast(method, train, test_h, product, args, materials)
            sc = _score(actual, fc)
            if sc:
                per_origin.append(sc)
        except Exception as e:
            log.warning("[%s/%s] origin %d failed: %s", product, method, i, e)

    if not per_origin:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan,
                "sMAPE": np.nan, "coverage": np.nan, "n_origins": 0}
    agg = {k: float(np.mean([o[k] for o in per_origin])) for k in per_origin[0]}
    agg["n_origins"] = len(per_origin)
    return agg


def run_all(clean: dict, methods: list[str], args,
            materials_lookup=None) -> pd.DataFrame:
    """Backtest every (product, method) pair into a tidy metrics frame."""
    rows = []
    for key, res in clean.items():
        y = res.target
        materials = materials_lookup(key) if materials_lookup else None
        for method in methods:
            log.info("Backtesting %s / %s ...", key, method)
            m = walk_forward(y, method, key, args, materials=materials)
            rows.append({"product": res.label, "key": key, "method": method,
                         **m})
    return pd.DataFrame(rows)
