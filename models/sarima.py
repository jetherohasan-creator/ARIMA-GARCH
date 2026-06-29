"""
SARIMA model.

Order selection is automatic via ``pmdarima.auto_arima`` when available, with a
pure-``statsmodels`` AIC grid-search fallback if ``pmdarima`` fails to import
(it has fragile binary dependencies).

Two operating modes (``--sarima-freq``):

* ``weekly`` (default): model the native weekly series. A full annual seasonal
  search (m=52) is prohibitively expensive and numerically fragile, so the
  weekly model is fit **non-seasonally** (seasonality at m=52 is reported by the
  EDA and captured by the monthly mode). This keeps SARIMA directly comparable,
  on the same weekly grid, to ARIMA-GARCH and Linear Regression.

* ``monthly``: resample weekly -> monthly mean and run a genuine **seasonal**
  search at m=12. This is the tractable seasonal variant; its forecast is
  produced at monthly frequency and saved as a separate artifact.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from . import ForecastResult, future_index

log = logging.getLogger("sarima")

try:
    import pmdarima as pm
    _HAVE_PM = True
except Exception as e:  # pragma: no cover
    _HAVE_PM = False
    log.warning("pmdarima unavailable (%s) — using statsmodels grid search", e)


# --------------------------------------------------------------------------- #
# Order selection
# --------------------------------------------------------------------------- #
def _grid_search(y: pd.Series, seasonal: bool, m: int,
                 max_p=3, max_q=3, max_P=1, max_Q=1):
    """Minimal AIC grid search fallback (used only if pmdarima is missing)."""
    import itertools
    best = (np.inf, None, None)
    d = 1
    D = 1 if seasonal else 0
    p_rng, q_rng = range(max_p + 1), range(max_q + 1)
    P_rng = range(max_P + 1) if seasonal else [0]
    Q_rng = range(max_Q + 1) if seasonal else [0]
    for p, q, P, Q in itertools.product(p_rng, q_rng, P_rng, Q_rng):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(
                    y, order=(p, d, q),
                    seasonal_order=(P, D, Q, m) if seasonal else (0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
            if res.aic < best[0]:
                best = (res.aic, (p, d, q),
                        (P, D, Q, m) if seasonal else (0, 0, 0, 0))
        except Exception:
            continue
    return best[1] or (1, 1, 1), best[2] or (0, 0, 0, 0)


def select_order(y: pd.Series, seasonal: bool, m: int) -> tuple:
    if _HAVE_PM:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = pm.auto_arima(
                y, seasonal=seasonal, m=m if seasonal else 1,
                max_p=4, max_q=4, max_P=2, max_Q=2,
                d=None, D=1 if seasonal else 0,
                information_criterion="aic", stepwise=True,
                suppress_warnings=True, error_action="ignore",
            )
        so = model.seasonal_order if seasonal else (0, 0, 0, 0)
        return model.order, so
    return _grid_search(y, seasonal, m)


# --------------------------------------------------------------------------- #
# Fit + forecast
# --------------------------------------------------------------------------- #
def fit_forecast(y: pd.Series, horizon: int, freq: str = "weekly",
                 alpha: float = 0.05, product: str = "",
                 seasonal_m_weekly: int = 52) -> ForecastResult:
    y = y.dropna()
    if freq == "monthly":
        ym = y.resample("MS").mean().dropna()
        seasonal, m = True, 12
        work, work_freq, steps = ym, "MS", max(1, round(horizon * 12 / 52))
    else:
        seasonal, m = False, seasonal_m_weekly
        work, work_freq, steps = y, "7D", horizon

    order, sorder = select_order(work, seasonal, m)
    log.info("[%s] SARIMA order=%s seasonal_order=%s (%s)",
             product, order, sorder, freq)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(
            work, order=order, seasonal_order=sorder,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)

    fc = res.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=alpha)
    idx = future_index(work, steps, work_freq)
    mean.index = idx
    lower = pd.Series(ci.iloc[:, 0].values, index=idx)
    upper = pd.Series(ci.iloc[:, 1].values, index=idx)

    fitted = res.fittedvalues
    fitted = fitted[fitted.index >= work.index[0]]

    # Ljung-Box on residuals for the diagnostics table.
    diag = {}
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(res.resid.dropna(), lags=[min(10, len(work) // 5)],
                            return_df=True)
        diag["ljung_box_p"] = float(lb["lb_pvalue"].iloc[-1])
    except Exception:
        pass
    diag["aic"] = float(res.aic)

    spec = f"SARIMA{order}x{sorder} [{freq}]"
    return ForecastResult(
        method="SARIMA", product=product, mean=mean, lower=lower, upper=upper,
        fitted=fitted, spec=spec, diagnostics=diag, alpha=alpha,
    )
