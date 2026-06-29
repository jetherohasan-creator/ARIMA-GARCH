"""
Pipeline entry point.

    python run.py [--horizon 26] [--sarima-freq weekly|monthly]
                  [--lr-mode trend|lags|materials] [--garch-vol GARCH|EGARCH|GJR]
                  [--ammonia-proxy za] [--products urea npk za tsp]
                  [--no-backtest]

Runs: clean -> EDA -> {SARIMA, ARIMA-GARCH, Linear Regression} -> walk-forward
backtest -> forecasts/plots/metrics -> outputs/report.md
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data_loader
import eda
from models import arima_garch, linreg, sarima

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("run")

METHODS = ["SARIMA", "ARIMA-GARCH", "LinearRegression"]
METHOD_FILE = {"SARIMA": "sarima", "ARIMA-GARCH": "arima_garch",
               "LinearRegression": "linreg"}

AMMONIA_DISCLAIMER = (
    "No Ammonia sheet exists in the source workbook, so Ammonia is **not "
    "modelled from real data**. When `--ammonia-proxy=za` is set, the ZA "
    "(ammonium sulphate) series is substituted purely as a methodological "
    "placeholder. **Ammonium sulphate is NOT ammonia** — its price level, "
    "volatility and drivers differ materially. Do not use the proxy output for "
    "any commercial or risk decision; replace it with a real ammonia feed."
)


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Fertilizer global price forecasting")
    p.add_argument("--horizon", type=int, default=config.DEFAULT_HORIZON,
                   help="forecast horizon in weeks (default 26)")
    p.add_argument("--sarima-freq", choices=["weekly", "monthly"],
                   default="weekly")
    p.add_argument("--lr-mode", choices=["trend", "lags", "materials"],
                   default="trend")
    p.add_argument("--garch-vol", choices=["GARCH", "EGARCH", "GJR"],
                   default="GARCH")
    p.add_argument("--ammonia-proxy", choices=["za"], default=None,
                   help="use ZA as a placeholder proxy for Ammonia (with warning)")
    p.add_argument("--products", nargs="+", default=None,
                   help="subset of products (default: all available)")
    p.add_argument("--no-backtest", action="store_true",
                   help="skip walk-forward backtest (faster)")
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_forecast(y: pd.Series, fc, label: str, path: str):
    fig, ax = plt.subplots(figsize=(12, 5))
    hist = y.dropna()
    ax.plot(hist.index, hist.values, color="black", lw=1.1, label="Actual")
    if fc.fitted is not None and len(fc.fitted):
        ax.plot(fc.fitted.index, fc.fitted.values, color="tab:blue", lw=0.9,
                alpha=0.7, label="Fitted")
    ax.plot(fc.mean.index, fc.mean.values, color="tab:red", lw=1.6,
            label="Forecast")
    ax.fill_between(fc.mean.index, fc.lower.values, fc.upper.values,
                    color="tab:red", alpha=0.18,
                    label=f"{int((1-fc.alpha)*100)}% interval")
    ax.set_title(f"{label} — {fc.method}\n{fc.spec}", fontsize=10)
    ax.set_ylabel("USD/Ton")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_forecast_csv(fc, path: str):
    df = pd.DataFrame({"forecast": fc.mean, "lower": fc.lower, "upper": fc.upper})
    df.index.name = "date"
    df.to_csv(path)


# --------------------------------------------------------------------------- #
# Per-product modelling
# --------------------------------------------------------------------------- #
def model_product(key: str, res: data_loader.CleanResult, args,
                  materials: dict | None):
    y = res.target
    forecasts = {}

    forecasts["SARIMA"] = sarima.fit_forecast(
        y, args.horizon, freq=args.sarima_freq, product=key)
    forecasts["ARIMA-GARCH"] = arima_garch.fit_forecast(
        y, args.horizon, product=key, vol=args.garch_vol)

    lr_mode = args.lr_mode
    lr_materials = None
    if key == "npk" and args.lr_mode == "materials":
        lr_materials = materials
    elif args.lr_mode == "materials" and key != "npk":
        lr_mode = "trend"  # materials only defined for NPK
    forecasts["LinearRegression"] = linreg.fit_forecast(
        y, args.horizon, mode=lr_mode, product=key, materials=lr_materials)

    # Prices cannot be negative; clip interval lower bounds at 0 (mainly affects
    # the wide OLS baselines).
    for fc in forecasts.values():
        fc.lower = fc.lower.clip(lower=0.0)

    # Persist artifacts.
    for method, fc in forecasts.items():
        fstub = METHOD_FILE[method]
        save_forecast_csv(fc, os.path.join(
            config.FORECAST_DIR, f"{key}_{fstub}.csv"))
        plot_forecast(y, fc, res.label, os.path.join(
            config.PLOT_DIR, f"{key}_{fstub}.png"))
    return forecasts


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(clean, eda_summ, all_forecasts, metrics_df, args, proxy_used):
    lines = ["# Fertilizer Global Price Forecast — Report", ""]
    lines.append(f"_Generated for horizon = **{args.horizon} weeks** "
                 f"(~{args.horizon/4.345:.1f} months). SARIMA freq = "
                 f"`{args.sarima_freq}`, LR mode = `{args.lr_mode}`, GARCH vol = "
                 f"`{args.garch_vol}`._")
    lines += ["", "## ⚠️ Ammonia disclaimer", "", AMMONIA_DISCLAIMER, ""]
    if proxy_used:
        lines.append("> **This run used `--ammonia-proxy=za`.** The `ammonia` "
                     "rows below are ZA data and are placeholders only.\n")

    # Cleaning summary
    lines += ["## 1. Data cleaning", "",
              eda_summ["clean"].to_markdown(index=False), "",
              "Key cleaning steps: native-datetime cells in column A had "
              "month/day swapped (US mis-entry) and were recovered to the modal "
              "weekday; string cells parsed with `dayfirst=True`; off-grid / "
              "out-of-sequence rows dropped; short gaps (≤2 wk) linearly "
              "interpolated; series reindexed to a regular weekly grid.", ""]

    # EDA / stationarity
    lines += ["## 2. EDA & stationarity", "",
              eda_summ["stat"].to_markdown(index=False), "",
              "_ADF H0 = unit root (small p ⇒ stationary); KPSS H0 = stationary "
              "(small p ⇒ non-stationary)._", ""]

    # Metrics
    lines += ["## 3. Backtest accuracy (walk-forward, rolling origin)", ""]
    if metrics_df is not None and not metrics_df.empty:
        show = metrics_df.copy()
        for c in ["RMSE", "MAE", "MAPE", "sMAPE", "coverage"]:
            if c in show:
                show[c] = show[c].round(2)
        lines.append(show.drop(columns=["key"], errors="ignore")
                     .to_markdown(index=False))
        lines.append("")
        # Best method per product (by RMSE).
        lines += ["### Best method per product (lowest RMSE)", ""]
        best_rows = []
        for key, grp in metrics_df.groupby("key"):
            grp2 = grp.dropna(subset=["RMSE"])
            if grp2.empty:
                continue
            b = grp2.loc[grp2["RMSE"].idxmin()]
            best_rows.append({"product": b["product"], "best_method": b["method"],
                              "RMSE": round(b["RMSE"], 2),
                              "MAPE%": round(b["MAPE"], 2)})
        if best_rows:
            lines.append(pd.DataFrame(best_rows).to_markdown(index=False))
        lines.append("")
    else:
        lines.append("_Backtest skipped (`--no-backtest`)._\n")

    # Selected specs + horizon-end forecast
    lines += ["## 4. Selected models & forecast summary", ""]
    spec_rows = []
    for key, fdict in all_forecasts.items():
        label = clean[key].label
        for method, fc in fdict.items():
            spec_rows.append({
                "product": label, "method": method, "spec": fc.spec,
                f"forecast_end (+{args.horizon}w)": round(float(fc.mean.iloc[-1]), 1),
                "lower": round(float(fc.lower.iloc[-1]), 1),
                "upper": round(float(fc.upper.iloc[-1]), 1),
            })
    lines.append(pd.DataFrame(spec_rows).to_markdown(index=False))
    lines += ["", "## 5. Assumptions & notes", "",
              "- Weekly grid anchored on the modal weekday (Thursday).",
              "- ARIMA-GARCH models log-returns (ARX mean + GARCH variance, "
              "Student-t); intervals widen with conditional volatility.",
              "- Linear `materials` mode (NPK) holds Urea/ZA/TSP at last value "
              "for future regressors (random-walk assumption).",
              "- SARIMA `weekly` mode is non-seasonal for tractability; use "
              "`--sarima-freq=monthly` for the seasonal (m=12) variant.",
              "- Forecasts are statistical projections, not price advice.", ""]

    os.makedirs(config.OUT_DIR, exist_ok=True)
    with open(config.REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    log.info("Report written to %s", config.REPORT_PATH)


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    seed_everything(config.SEED)
    for d in (config.CLEAN_DIR, config.FORECAST_DIR, config.PLOT_DIR):
        os.makedirs(d, exist_ok=True)

    # Resolve product list.
    requested = args.products or list(config.DEFAULT_PRODUCTS)
    proxy_used = False
    if "ammonia" in requested:
        requested = [p for p in requested if p != "ammonia"]
        if args.ammonia_proxy == "za" and "za" not in requested:
            requested.append("za")
        log.warning("Ammonia requested but no real data exists. %s",
                    "Using ZA as proxy." if args.ammonia_proxy else
                    "Skipping Ammonia.")
    if args.ammonia_proxy == "za":
        proxy_used = True
        log.warning("--ammonia-proxy=za active: ZA is a PLACEHOLDER, not ammonia.")

    log.info("Products: %s | horizon=%d", requested, args.horizon)

    # 1. Clean
    clean = data_loader.load_all(requested)
    clean_summ = data_loader.summarize(clean)
    print("\n=== CLEANING SUMMARY ===")
    print(clean_summ.to_string(index=False))
    for k, r in clean.items():
        print(f"\n[{r.label}] head/tail:")
        print(r.target.dropna().head(3).to_string())
        print("...")
        print(r.target.dropna().tail(3).to_string())

    # Materials lookup for NPK regression.
    def materials_for(key):
        if key != "npk":
            return None
        mats = {}
        for mk in config.NPK_MATERIAL_PRODUCTS:
            if mk in clean:
                mats[mk] = clean[mk].target
        return mats or None

    # 2. EDA
    stat_rows = []
    for k, r in clean.items():
        a = eda.assess(r.target, r.label)
        eda.plot_diagnostics(r.target, k, r.label)
        stat_rows.append({
            "product": r.label,
            "ADF_lvl_p": round(a["adf_level_p"], 3),
            "KPSS_lvl_p": round(a["kpss_level_p"], 3),
            "ADF_d1_p": round(a["adf_diff1_p"], 3),
            "suggested_d": a["suggested_d"],
            "seasonal_strength": round(a["seasonal_strength"], 3)
            if a["seasonal_strength"] == a["seasonal_strength"] else None,
            "has_seasonality": a["has_seasonality"],
        })
    stat_df = pd.DataFrame(stat_rows)
    print("\n=== EDA / STATIONARITY ===")
    print(stat_df.to_string(index=False))

    # 3. Modelling
    all_forecasts = {}
    for k, r in clean.items():
        log.info("Modelling %s ...", r.label)
        all_forecasts[k] = model_product(k, r, args, materials_for(k))

    # 4. Backtest
    metrics_df = None
    if not args.no_backtest:
        import backtest
        metrics_df = backtest.run_all(clean, METHODS, args,
                                      materials_lookup=materials_for)
        metrics_df.to_csv(config.METRICS_PATH, index=False)
        print("\n=== BACKTEST METRICS ===")
        print(metrics_df.drop(columns=["key"]).round(2).to_string(index=False))

    # 5. Report
    write_report(clean, {"clean": clean_summ, "stat": stat_df},
                 all_forecasts, metrics_df, args, proxy_used)

    print("\nDone. See outputs/ for forecasts, plots, metrics and report.md")


if __name__ == "__main__":
    main()
