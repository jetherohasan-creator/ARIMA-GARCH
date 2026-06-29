"""
ARIMA-GARCH model.

We model the **log-returns** (a stationary series) with a unified ARX-GARCH
specification from the ``arch`` package:

    arch_model(100*dlog(price), mean='ARX', lags=p, vol=<GARCH|EGARCH|GJR>,
               p=1, q=1, dist='t')

Rationale for the single-model (rather than two-stage) choice: ``arch`` jointly
estimates the AR mean and the conditional-variance parameters by MLE, which is
cleaner to forecast from than stitching a separate statsmodels ARIMA to a GARCH
on its residuals. The AR order ``p`` is selected on the returns (auto_arima if
available, else a PACF heuristic). An Engle ARCH-LM test decides whether the
GARCH layer is warranted (it is reported either way).

Forecasting:
  * h-step mean log-returns are accumulated to a log-price path,
  * the **conditional variance forecast widens** the prediction interval: the
    variance of the cumulative log-return at step h is the running sum of the
    per-step conditional variances,
  * everything is inverted with ``exp`` back to USD/Ton.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from . import ForecastResult, future_index

log = logging.getLogger("arima_garch")

SCALE = 100.0  # arch prefers returns on a percentage-ish scale

try:
    import pmdarima as pm
    _HAVE_PM = True
except Exception:  # pragma: no cover
    _HAVE_PM = False


def _select_ar_order(returns: pd.Series, max_p: int = 5) -> int:
    if _HAVE_PM:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = pm.auto_arima(returns, seasonal=False, d=0,
                                  max_p=max_p, max_q=0, stepwise=True,
                                  information_criterion="aic",
                                  suppress_warnings=True, error_action="ignore")
            return max(1, m.order[0])
        except Exception:
            pass
    # PACF heuristic fallback.
    from statsmodels.tsa.stattools import pacf
    pac = pacf(returns, nlags=max_p)
    thr = 1.96 / np.sqrt(len(returns))
    sig = [i for i in range(1, len(pac)) if abs(pac[i]) > thr]
    return sig[-1] if sig else 1


def _vol_kwargs(vol: str) -> dict:
    vol = (vol or "GARCH").upper()
    if vol == "EGARCH":
        return {"vol": "EGARCH", "p": 1, "q": 1, "o": 1}
    if vol in ("GJR", "GJR-GARCH", "GJRGARCH"):
        return {"vol": "GARCH", "p": 1, "o": 1, "q": 1}  # GJR-GARCH
    return {"vol": "GARCH", "p": 1, "q": 1}


def fit_forecast(y: pd.Series, horizon: int, alpha: float = 0.05,
                 product: str = "", vol: str = "GARCH") -> ForecastResult:
    y = y.dropna()
    logp = np.log(y)
    ret = (logp.diff().dropna()) * SCALE

    p = _select_ar_order(ret)

    # Engle ARCH-LM test on a quick AR(p) fit residual.
    arch_p = np.nan
    try:
        from statsmodels.tsa.ar_model import AutoReg
        ar_res = AutoReg(ret, lags=p, old_names=False).fit()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, arch_p, _, _ = het_arch(ar_res.resid.dropna(), nlags=12)
    except Exception:
        pass
    if not (arch_p < 0.05):
        log.info("[%s] ARCH-LM p=%.3g (weak ARCH effect) — GARCH still fit "
                 "for interval widening", product, arch_p)

    vk = _vol_kwargs(vol)
    am = arch_model(ret, mean="ARX", lags=p, dist="t", **vk)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off")
    nu = float(res.params.get("nu", np.inf))
    log.info("[%s] ARIMA-GARCH: AR(%d)+%s, dist=t(nu=%.1f)",
             product, p, vk["vol"], nu)

    # h-step forecast of mean returns and conditional variance.
    fc = res.forecast(horizon=horizon, reindex=False, method="simulation"
                      if vk["vol"] == "EGARCH" else "analytic")
    mean_ret = np.asarray(fc.mean.iloc[-1].values, dtype=float) / SCALE
    var_ret = np.asarray(fc.variance.iloc[-1].values, dtype=float) / (SCALE ** 2)

    # --- Stabilise the variance cone -------------------------------------- #
    # The 2021-22 price spike drives an effectively IGARCH fit (alpha+beta ~ 1):
    # the analytic multi-step variance then grows without bound and, after the
    # exp() back-transform, produces absurd intervals. When the model is
    # (near-)IGARCH we cap each step's forecast variance at a robust long-run
    # ceiling = variance of the most recent ~2 years of returns. The full-sample
    # variance is itself inflated by the one-off 2021 spike (urea: 14%/wk full
    # vs 5%/wk recent), so the recent window is the more credible regime proxy.
    # The cone then grows at most like a random walk at recent volatility.
    recent = (ret / SCALE).iloc[-min(104, len(ret)):]
    ceiling = float(np.var(recent.values, ddof=1))
    persistence = float(res.params.get("alpha[1]", 0.0)
                        + res.params.get("beta[1]", 0.0)
                        + 0.5 * res.params.get("gamma[1]", 0.0))
    if persistence >= 0.99:
        var_ret = np.minimum(var_ret, ceiling)
        diag_cap = True
    else:
        diag_cap = False

    # Accumulate to a log-price path; cumulative variance widens the interval.
    last_logp = float(logp.iloc[-1])
    cum_mean = np.cumsum(mean_ret)
    cum_var = np.cumsum(var_ret)

    # Robustness guard. On some (esp. short backtest) training windows the ARX
    # mean fits a near-unit-root AR with a non-zero constant, so its multi-step
    # mean forecast diverges and exp() overflows to +/-inf. Library-version
    # differences can flip a window into this regime. We therefore (a) replace
    # non-finite values and (b) clamp the cumulative log-drift and the interval
    # half-width to economically sane bounds, so the point forecast can move at
    # most ~4x and the band at most ~8x over the horizon.
    drift_cap = np.log(4.0)
    se_cap = np.log(8.0)
    cum_mean = np.clip(np.nan_to_num(cum_mean, nan=0.0,
                                     posinf=drift_cap, neginf=-drift_cap),
                       -drift_cap, drift_cap)
    cum_var = np.nan_to_num(cum_var, nan=0.0, posinf=se_cap ** 2)
    se = np.clip(np.sqrt(cum_var), 0.0, se_cap)

    # Use a Student-t quantile (fat tails) but floor the d.o.f. at 5: the raw
    # MLE nu ~ 2 has near-infinite variance and over-inflates the band.
    nu_q = max(nu, 5.0) if np.isfinite(nu) else np.inf
    q = (stats.t.ppf(1 - alpha / 2, df=nu_q) if np.isfinite(nu_q)
         else stats.norm.ppf(1 - alpha / 2))

    idx = future_index(y, horizon, "7D")
    logp_fc = last_logp + cum_mean
    mean = pd.Series(np.exp(logp_fc), index=idx)
    lower = pd.Series(np.exp(logp_fc - q * se), index=idx)
    upper = pd.Series(np.exp(logp_fc + q * se), index=idx)

    # In-sample fitted prices (one-step): logp_{t-1} + conditional mean return.
    cond_mean_ret = (ret - res.resid) / SCALE
    fitted_logp = logp.shift(1).reindex(cond_mean_ret.index) + cond_mean_ret
    fitted = np.exp(fitted_logp).dropna().rename(product)

    diag = {"ar_order": p, "vol": vk["vol"], "nu": nu, "arch_lm_p": float(arch_p)
            if np.isfinite(arch_p) else np.nan, "aic": float(res.aic),
            "persistence": persistence, "igarch_var_capped": diag_cap}
    try:
        std_resid = (res.resid / res.conditional_volatility).dropna()
        lb = acorr_ljungbox(std_resid, lags=[10], return_df=True)
        diag["ljung_box_p"] = float(lb["lb_pvalue"].iloc[-1])
        lb2 = acorr_ljungbox(std_resid ** 2, lags=[10], return_df=True)
        diag["arch_remaining_p"] = float(lb2["lb_pvalue"].iloc[-1])
    except Exception:
        pass

    spec = f"ARIMA-GARCH: ARX(lags={p}) + {vk['vol']}(1,1), t-dist"
    return ForecastResult(
        method="ARIMA-GARCH", product=product, mean=mean, lower=lower,
        upper=upper, fitted=fitted, spec=spec, diagnostics=diag, alpha=alpha,
    )
