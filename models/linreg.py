"""
Linear-regression baselines (``--lr-mode``).

* ``trend``     : OLS  price ~ t + t^2   (deterministic time trend).
* ``lags``      : autoregressive features price ~ lags{1,2,4,8,52} (+ trend).
* ``materials`` : NPK ~ Urea + ZA + TSP  (economically-motivated blend model).

Prediction intervals come from OLS standard errors of prediction (trend mode,
exact) or an approximate residual-variance interval (lags/materials, where
future regressors must themselves be projected).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import ForecastResult, future_index

log = logging.getLogger("linreg")

LAGS = [1, 2, 4, 8, 52]


# --------------------------------------------------------------------------- #
# trend
# --------------------------------------------------------------------------- #
def _fit_trend(y: pd.Series, horizon: int, alpha: float, product: str):
    s = y.dropna()
    t = np.arange(len(s), dtype=float)
    X = sm.add_constant(np.column_stack([t, t ** 2]))
    model = sm.OLS(s.values, X).fit()

    idx = future_index(s, horizon, "7D")
    tf = np.arange(len(s), len(s) + horizon, dtype=float)
    Xf = sm.add_constant(np.column_stack([tf, tf ** 2]), has_constant="add")
    pred = model.get_prediction(Xf).summary_frame(alpha=alpha)

    mean = pd.Series(pred["mean"].values, index=idx)
    lower = pd.Series(pred["obs_ci_lower"].values, index=idx)
    upper = pd.Series(pred["obs_ci_upper"].values, index=idx)
    fitted = pd.Series(model.fittedvalues, index=s.index)
    spec = f"OLS price ~ t + t^2 (R²={model.rsquared:.3f})"
    return mean, lower, upper, fitted, spec, {"r2": float(model.rsquared)}


# --------------------------------------------------------------------------- #
# lags
# --------------------------------------------------------------------------- #
def _build_lag_frame(s: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": s})
    for L in LAGS:
        df[f"lag{L}"] = s.shift(L)
    df["t"] = np.arange(len(s))
    return df


def _fit_lags(y: pd.Series, horizon: int, alpha: float, product: str):
    s = y.dropna()
    df = _build_lag_frame(s).dropna()
    feats = [f"lag{L}" for L in LAGS] + ["t"]
    X = sm.add_constant(df[feats])
    model = sm.OLS(df["y"], X).fit()
    sigma = np.sqrt(model.scale)
    fitted = pd.Series(model.fittedvalues, index=df.index)

    # Recursive multi-step forecast, feeding predictions back as lags.
    hist = s.copy()
    idx = future_index(s, horizon, "7D")
    preds = []
    for h in range(horizon):
        t_next = len(s) + h
        row = {"const": 1.0, "t": t_next}
        for L in LAGS:
            pos = len(hist) - L
            row[f"lag{L}"] = hist.iloc[pos] if pos >= 0 else hist.iloc[0]
        xf = np.array([row["const"]] + [row[f"lag{L}"] for L in LAGS] + [row["t"]])
        yhat = float(model.params.values @ xf)
        preds.append(yhat)
        hist = pd.concat([hist, pd.Series([yhat], index=[idx[h]])])

    mean = pd.Series(preds, index=idx)
    # Interval widens ~ sqrt(h) with the 1-step residual sigma (approximation).
    z = stats.norm.ppf(1 - alpha / 2)
    widen = z * sigma * np.sqrt(np.arange(1, horizon + 1))
    lower, upper = mean - widen, mean + widen
    spec = f"OLS lags{LAGS}+trend (R²={model.rsquared:.3f})"
    return mean, lower, upper, fitted, spec, {"r2": float(model.rsquared)}


# --------------------------------------------------------------------------- #
# materials (NPK only)
# --------------------------------------------------------------------------- #
def _fit_materials(y: pd.Series, horizon: int, alpha: float, product: str,
                   materials: dict[str, pd.Series]):
    s = y.dropna()
    parts = {"y": s}
    for name, mser in materials.items():
        parts[name] = mser
    df = pd.DataFrame(parts).dropna()
    feats = list(materials.keys())
    X = sm.add_constant(df[feats])
    model = sm.OLS(df["y"], X).fit()
    sigma = np.sqrt(model.scale)
    fitted = pd.Series(model.fittedvalues, index=df.index)

    # Future regressors: hold each material at its last observed value
    # (random-walk assumption — documented). Then apply the fitted relation.
    idx = future_index(s, horizon, "7D")
    last = {name: mser.dropna().iloc[-1] for name, mser in materials.items()}
    xf = np.array([1.0] + [last[name] for name in feats])
    yhat = float(model.params.values @ xf)
    mean = pd.Series([yhat] * horizon, index=idx)
    z = stats.norm.ppf(1 - alpha / 2)
    widen = z * sigma * np.sqrt(np.arange(1, horizon + 1))
    lower, upper = mean - widen, mean + widen

    coefs = {k: round(float(v), 3) for k, v in model.params.items()}
    spec = (f"OLS NPK ~ {'+'.join(feats)} (R²={model.rsquared:.3f}); "
            f"coefs={coefs}")
    log.info("[%s] materials regression coefs=%s R²=%.3f",
             product, coefs, model.rsquared)
    return mean, lower, upper, fitted, spec, {
        "r2": float(model.rsquared), "coefs": coefs}


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def fit_forecast(y: pd.Series, horizon: int, mode: str = "trend",
                 alpha: float = 0.05, product: str = "",
                 materials: dict[str, pd.Series] | None = None) -> ForecastResult:
    if mode == "materials" and materials:
        mean, lower, upper, fitted, spec, diag = _fit_materials(
            y, horizon, alpha, product, materials)
        used = "materials"
    elif mode == "lags":
        mean, lower, upper, fitted, spec, diag = _fit_lags(
            y, horizon, alpha, product)
        used = "lags"
    else:
        if mode == "materials":
            log.warning("[%s] materials mode requested but no materials "
                        "supplied — falling back to trend", product)
        mean, lower, upper, fitted, spec, diag = _fit_trend(
            y, horizon, alpha, product)
        used = "trend"

    diag["lr_mode"] = used
    return ForecastResult(
        method="LinearRegression", product=product, mean=mean, lower=lower,
        upper=upper, fitted=fitted, spec=spec, diagnostics=diag, alpha=alpha,
    )
