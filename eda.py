"""
Exploratory analysis & stationarity testing.

Produces, per product:
  * stationarity tests (ADF + KPSS) on level and first difference,
  * a suggested differencing order ``d``,
  * a seasonality assessment via STL decomposition + seasonal-lag ACF,
  * an ACF/PACF + STL diagnostic plot.
"""
from __future__ import annotations

import logging
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss

import config
import plotting

log = logging.getLogger("eda")


def _adf(series: pd.Series) -> tuple[float, float]:
    s = series.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, *_ = adfuller(s, autolag="AIC")
    return float(stat), float(p)


def _kpss(series: pd.Series) -> tuple[float, float]:
    s = series.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, *_ = kpss(s, regression="c", nlags="auto")
    return float(stat), float(p)


def assess(series: pd.Series, label: str, seasonal_period: int = 52) -> dict:
    """Run stationarity + seasonality diagnostics and return a summary dict."""
    s = series.dropna()
    out: dict = {"label": label, "n": int(s.size)}

    # ADF: H0 = unit root (non-stationary). KPSS: H0 = stationary.
    adf_lvl = _adf(s)
    kpss_lvl = _kpss(s)
    diff1 = s.diff().dropna()
    adf_d1 = _adf(diff1)
    kpss_d1 = _kpss(diff1)

    out.update(
        adf_level_p=adf_lvl[1], kpss_level_p=kpss_lvl[1],
        adf_diff1_p=adf_d1[1], kpss_diff1_p=kpss_d1[1],
    )

    # Suggested d: level stationary if ADF rejects AND KPSS does not.
    level_stationary = (adf_lvl[1] < 0.05) and (kpss_lvl[1] > 0.05)
    diff_stationary = (adf_d1[1] < 0.05) and (kpss_d1[1] > 0.05)
    if level_stationary:
        out["suggested_d"] = 0
    elif diff_stationary or adf_d1[1] < 0.05:
        out["suggested_d"] = 1
    else:
        out["suggested_d"] = 2

    # Seasonality: STL strength + ACF at the seasonal lag.
    out["seasonal_strength"] = np.nan
    out["acf_seasonal_lag"] = np.nan
    if s.size > 2 * seasonal_period:
        try:
            stl = STL(s, period=seasonal_period, robust=True).fit()
            var_resid = np.var(stl.resid)
            var_sr = np.var(stl.seasonal + stl.resid)
            out["seasonal_strength"] = float(max(0.0, 1 - var_resid / var_sr))
        except Exception as e:  # pragma: no cover
            log.warning("[%s] STL failed: %s", label, e)
        try:
            ac = acf(s, nlags=seasonal_period, fft=True)
            out["acf_seasonal_lag"] = float(ac[seasonal_period])
        except Exception:
            pass
    out["has_seasonality"] = bool(
        (out["seasonal_strength"] or 0) > 0.3
        or abs(out["acf_seasonal_lag"] or 0) > 0.3
    )
    return out


def plot_diagnostics(series: pd.Series, key: str, label: str,
                     seasonal_period: int = 52) -> str:
    """Save a 2x2 diagnostic figure (level, ACF, PACF, STL seasonal)."""
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    s = series.dropna()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(s.index, s.values, lw=1)
    axes[0, 0].set_title(f"{label} — level (USD/Ton)")

    nlags = min(60, s.size // 2 - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_acf(s.diff().dropna(), ax=axes[0, 1], lags=nlags)
        plot_pacf(s.diff().dropna(), ax=axes[1, 0], lags=nlags, method="ywm")
    axes[0, 1].set_title("ACF (1st diff)")
    axes[1, 0].set_title("PACF (1st diff)")

    if s.size > 2 * seasonal_period:
        try:
            stl = STL(s, period=seasonal_period, robust=True).fit()
            axes[1, 1].plot(s.index, stl.seasonal, lw=1, color="tab:green")
            axes[1, 1].set_title(f"STL seasonal (m={seasonal_period})")
        except Exception:
            axes[1, 1].set_visible(False)
    else:
        axes[1, 1].text(0.5, 0.5, "series too short for STL",
                        ha="center", va="center")
    fig.suptitle(f"EDA diagnostics — {label}")
    fig.tight_layout()
    path = os.path.join(config.PLOT_DIR, f"{key}_eda.png")
    plotting.safe_savefig(fig, path, dpi=110)
    return path
