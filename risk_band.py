"""
Business risk band (Ceiling / Mid / Floor) + commercial signals.

This integrates the decision-support framing from the reference notebook into our
validated ARIMA-GARCH engine:

* **Mid**     — model's best estimate (forecast / fitted conditional mean).
* **Ceiling** — Mid + Z·σ. If the market price approaches/exceeds it: *waspada*
  — consider hedging or setting a ceiling price (HET).
* **Floor**   — max(Mid − Z·σ, 0). If price approaches it: buying opportunity /
  good time to negotiate cheap contracts.

Unlike the reference (which used a **constant** band width), our band uses the
GARCH **conditional volatility**, so it widens with risk over the horizon. Z is
configurable (default 1σ, matching the reference).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def z_to_alpha(z: float) -> float:
    """Two-sided tail mass for a given sigma multiple (Z=1 -> ~0.317)."""
    return float(2 * (1 - stats.norm.cdf(z)))


def insample_mape(actual: pd.Series, fitted: pd.Series) -> float:
    """In-sample 1-step fit MAPE (goodness-of-fit, NOT forecast skill)."""
    a, f = actual.align(fitted, join="inner")
    m = (a != 0) & np.isfinite(a) & np.isfinite(f)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((a[m] - f[m]) / a[m])) * 100)


def forecast_band(fc, z: float = 1.0) -> pd.DataFrame:
    """Ceiling/Mid/Floor over the forecast horizon at Z·σ (price scale)."""
    mid = fc.mean
    se = fc.diagnostics.get("logret_se")
    if se is not None and len(se) == len(mid):
        se = np.asarray(se, dtype=float)
    else:  # fall back: back out sigma from the stored alpha interval
        q = stats.norm.ppf(1 - fc.alpha / 2)
        se = np.log(fc.upper.values / mid.values) / q
    ceiling = mid * np.exp(z * se)
    floor = (mid * np.exp(-z * se)).clip(lower=0)
    return pd.DataFrame({"floor": floor, "mid": mid, "ceiling": ceiling})


def insample_band(fc, actual: pd.Series, z: float = 1.0) -> pd.DataFrame:
    """Historical Ceiling/Mid/Floor using GARCH conditional volatility."""
    mid = fc.fitted
    sigma = fc.diagnostics.get("insample_sigma_logret")
    if mid is None or sigma is None:
        return pd.DataFrame()
    idx = mid.index.intersection(sigma.index)
    mid, s = mid.loc[idx], sigma.loc[idx]
    ceiling = mid * np.exp(z * s)
    floor = (mid * np.exp(-z * s)).clip(lower=0)
    df = pd.DataFrame({"actual": actual.reindex(idx), "floor": floor,
                       "mid": mid, "ceiling": ceiling})
    return df


def current_signal(actual: pd.Series, fc, z: float = 1.0) -> dict:
    """Classify where the latest observed price sits in its volatility envelope.

    Returns a zone (BELI / NORMAL / WASPADA) and a ratio in [0,1] giving the
    position of the actual between this week's Floor (0) and Ceiling (1).
    """
    band = insample_band(fc, actual, z)
    if band.empty or band.dropna().empty:
        return {"zone": "NA", "message": "data tidak cukup", "position": np.nan,
                "actual": np.nan, "floor": np.nan, "mid": np.nan,
                "ceiling": np.nan}
    last = band.dropna(subset=["actual"]).iloc[-1]
    a, lo, hi, mid = last["actual"], last["floor"], last["ceiling"], last["mid"]
    pos = float((a - lo) / (hi - lo)) if hi > lo else 0.5
    if a >= hi:
        zone, msg = "WASPADA", ("harga di/di atas Ceiling — pertimbangkan "
                                "hedging atau tetapkan HET")
    elif a <= lo:
        zone, msg = "BELI", ("harga di/di bawah Floor — peluang beli / "
                             "negosiasi kontrak murah")
    elif a > mid:
        zone, msg = "NORMAL-TINGGI", "harga di paruh atas rentang wajar"
    else:
        zone, msg = "NORMAL-RENDAH", "harga di paruh bawah rentang wajar"
    return {"zone": zone, "message": msg, "position": pos, "actual": float(a),
            "floor": float(lo), "mid": float(mid), "ceiling": float(hi)}
