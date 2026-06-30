"""
Realization tracking — log actual prices as they happen and score the forecasts.

Workflow
--------
1. ``python track_realization.py --init``
   Creates an editable template ``realizations/<product>.csv`` (date,actual) for
   every product.

2. As each week's reference price is published, record it either by editing the
   CSV directly or with::

       python track_realization.py --add urea 2026-06-11 651.2

   (date may be any weekday; it is snapped to the product's weekly grid.)

3. ``python track_realization.py``
   Compares the realized actuals against the saved forecasts in
   ``outputs/forecasts/<product>_<method>.csv`` and writes:
     * ``outputs/realization_tracking.csv``  — error of each method on the
       realized window (RMSE/MAE/MAPE/sMAPE + interval hit-rate),
     * ``outputs/plots/<product>_realization.png`` — forecast vs realized,
     * console table of which method is tracking reality best so far.

4. ``python track_realization.py --refit``
   Re-runs the whole forecasting pipeline on the **extended** series
   (clean workbook data + realized actuals) so the next forecast starts from the
   latest realized week.

The realized values are kept separate from the source workbook; they are merged
onto the cleaned series at runtime (realized values win on overlapping dates).
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import plotting
import data_loader
from backtest import mae, mape, rmse, smape

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("track")

REAL_DIR = os.path.join(config.DATA_DIR, "realizations")
TRACK_CSV = os.path.join(config.OUT_DIR, "realization_tracking.csv")

METHOD_FILE = {"SARIMA": "sarima", "ARIMA-GARCH": "arima_garch",
               "LinearRegression": "linreg"}


# --------------------------------------------------------------------------- #
# Realization storage
# --------------------------------------------------------------------------- #
def realization_path(key: str) -> str:
    return os.path.join(REAL_DIR, f"{key}.csv")


def init_templates(keys: list[str]):
    os.makedirs(REAL_DIR, exist_ok=True)
    for k in keys:
        path = realization_path(k)
        if os.path.exists(path):
            log.info("[%s] template already exists: %s", k, path)
            continue
        plotting.safe_to_csv(pd.DataFrame(columns=["date", "actual"]), path,
                             index=False)
        log.info("[%s] created template %s", k, path)
    print(f"\nEdit the CSVs in {REAL_DIR}/ (columns: date,actual) or use "
          f"`--add <product> <date> <value>`.")


def load_realizations(key: str) -> pd.Series:
    """Read realized actuals for one product (empty Series if none)."""
    path = realization_path(key)
    if not os.path.exists(path):
        return pd.Series(dtype=float, name="actual")
    df = pd.read_csv(path)
    if df.empty:
        return pd.Series(dtype=float, name="actual")
    df["date"] = pd.to_datetime(df["date"])
    s = (df.dropna(subset=["actual"]).drop_duplicates("date", keep="last")
         .set_index("date")["actual"].sort_index())
    return s.rename("actual")


def parse_user_date(date_str: str) -> pd.Timestamp:
    """Parse a user-supplied date.

    ISO ``YYYY-MM-DD`` is unambiguous and parsed as-is; ``dd/mm/yyyy`` style
    uses ``dayfirst=True``. (Forcing dayfirst on ISO wrongly swaps month/day
    when both are <= 12, e.g. 2026-06-11 -> 2026-11-06.)
    """
    s = str(date_str).strip()
    if "/" in s:
        return pd.to_datetime(s, dayfirst=True)
    return pd.to_datetime(s)


def snap_to_grid(date: pd.Timestamp, ref: pd.Series) -> pd.Timestamp:
    """Snap a date to the product's weekly cadence (modal weekday of ref)."""
    if ref.empty:
        return date
    weekday = pd.Series(ref.index.weekday).mode().iloc[0]
    # snap to nearest grid weekday (forward if within 3 days, else back)
    fwd = date + pd.Timedelta(days=(weekday - date.weekday()) % 7)
    bwd = fwd - pd.Timedelta(days=7)
    return fwd if abs((fwd - date).days) <= abs((bwd - date).days) else bwd


def add_realization(key: str, date_str: str, value: float, clean):
    os.makedirs(REAL_DIR, exist_ok=True)
    raw = parse_user_date(date_str)
    ref = clean[key].target if key in clean else pd.Series(dtype=float)
    date = snap_to_grid(raw, ref)
    if date != raw:
        log.info("[%s] snapped %s -> %s (weekly grid)", key, raw.date(),
                 date.date())
    cur = load_realizations(key)
    cur.loc[date] = float(value)
    cur = cur.sort_index()
    plotting.safe_to_csv(cur.rename("actual").to_frame().rename_axis("date"),
                         realization_path(key))
    log.info("[%s] recorded %s = %.2f USD/Ton (%d realized points total)",
             key, date.date(), value, len(cur))


# --------------------------------------------------------------------------- #
# Extended series (clean + realized)
# --------------------------------------------------------------------------- #
def extended_series(key: str, clean) -> pd.Series:
    base = clean[key].target.copy()
    real = load_realizations(key)
    if real.empty:
        return base
    grid_step = pd.Timedelta(days=7)
    out = base.copy()
    for d, v in real.items():
        out.loc[d] = v
    out = out.sort_index()
    # Fill any new weekly gaps created by appending far-future realizations.
    full = pd.date_range(out.index.min(), out.index.max(), freq=config.WEEK_FREQ)
    out = out.reindex(full).interpolate(limit=2, limit_area="inside")
    out.index.name = "date"
    return out.rename(key)


# --------------------------------------------------------------------------- #
# Scoring forecasts against realized actuals
# --------------------------------------------------------------------------- #
def score_against_realized(key: str, real: pd.Series) -> list[dict]:
    rows = []
    for method, stub in METHOD_FILE.items():
        fpath = os.path.join(config.FORECAST_DIR, f"{key}_{stub}.csv")
        if not os.path.exists(fpath):
            continue
        fc = pd.read_csv(fpath, parse_dates=["date"]).set_index("date")
        joined = fc.join(real.rename("actual"), how="inner").dropna(
            subset=["actual"])
        if joined.empty:
            continue
        a = joined["actual"].values
        f = joined["forecast"].values
        hit = float(np.mean((a >= joined["lower"].values)
                            & (a <= joined["upper"].values)) * 100)
        rows.append({
            "product": config.PRODUCTS[key].label, "key": key, "method": method,
            "n_realized": int(len(joined)),
            "RMSE": round(rmse(a, f), 2), "MAE": round(mae(a, f), 2),
            "MAPE": round(mape(a, f), 2), "sMAPE": round(smape(a, f), 2),
            "interval_hit%": round(hit, 1),
        })
    return rows


def plot_realization(key: str, clean, real: pd.Series, lookback: int = 78):
    label = config.PRODUCTS[key].label
    hist = clean[key].target.dropna()
    hist = hist.iloc[-lookback:]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist.index, hist.values, color="black", lw=1.1, label="Actual (hist)")

    colors = {"SARIMA": "tab:blue", "ARIMA-GARCH": "tab:red",
              "LinearRegression": "tab:green"}
    for method, stub in METHOD_FILE.items():
        fpath = os.path.join(config.FORECAST_DIR, f"{key}_{stub}.csv")
        if not os.path.exists(fpath):
            continue
        fc = pd.read_csv(fpath, parse_dates=["date"]).set_index("date")
        ax.plot(fc.index, fc["forecast"], lw=1.3, color=colors[method],
                alpha=0.85, label=f"{method} forecast")
        ax.fill_between(fc.index, fc["lower"], fc["upper"],
                        color=colors[method], alpha=0.08)
    if not real.empty:
        ax.scatter(real.index, real.values, color="black", s=42, zorder=5,
                   marker="o", label="Realized", edgecolor="white")
    ax.set_title(f"{label} — forecast vs realized")
    ax.set_ylabel("USD/Ton")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    path = os.path.join(config.PLOT_DIR, f"{key}_realization.png")
    plotting.safe_savefig(fig, path, dpi=110)
    return path


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Track realized prices vs forecasts")
    p.add_argument("--init", action="store_true",
                   help="create empty realization templates")
    p.add_argument("--add", nargs="+",
                   metavar="PRODUCT DATE VALUE [DATE VALUE ...]",
                   help="append realized observation(s) for one product; "
                        "accepts multiple DATE VALUE pairs")
    p.add_argument("--refit", action="store_true",
                   help="re-run forecasting on the extended (clean+realized) series")
    p.add_argument("--products", nargs="+", default=None)
    args = p.parse_args()

    keys = args.products or list(config.DEFAULT_PRODUCTS)
    clean = data_loader.load_all(keys)

    if args.init:
        init_templates(keys)
        return

    if args.add:
        prod, rest = args.add[0], args.add[1:]
        if prod not in config.PRODUCTS:
            sys.exit(f"Unknown product '{prod}'. Choose from {list(config.PRODUCTS)}")
        if len(rest) < 2 or len(rest) % 2 != 0:
            sys.exit("--add expects: PRODUCT DATE VALUE [DATE VALUE ...] "
                     "(one product, then DATE VALUE pairs)")
        for i in range(0, len(rest), 2):
            add_realization(prod, rest[i], float(rest[i + 1]), clean)
        # fall through so the user immediately sees updated tracking

    if args.refit:
        # Persist extended series so run.py picks up realized weeks, then re-run.
        os.makedirs(config.CLEAN_DIR, exist_ok=True)
        for k in keys:
            ext = extended_series(k, clean)
            plotting.safe_to_csv(
                pd.concat([ext, clean[k].support.reindex(ext.index)], axis=1),
                os.path.join(config.CLEAN_DIR, f"{k}_extended.csv"))
        log.info("Extended series written. Re-running run.py --use-extended ...")
        subprocess.run([sys.executable, "run.py", "--use-extended",
                        "--products", *keys], check=False)

    # Scoring + plots
    all_rows = []
    recorded_no_overlap = []  # has realizations, but none overlap the forecast
    for k in keys:
        real = load_realizations(k)
        if real.empty:
            log.info("[%s] no realized data yet (skip scoring)", k)
        else:
            rows = score_against_realized(k, real)
            if rows:
                all_rows += rows
            else:
                recorded_no_overlap.append((k, len(real)))
                log.info("[%s] %d realized week(s) recorded but none fall inside "
                         "the current forecast horizon — already folded into "
                         "history by --refit; add NEW forecast weeks to resume "
                         "scoring", k, len(real))
        plot_realization(k, clean, real)

    if all_rows:
        df = pd.DataFrame(all_rows)
        os.makedirs(config.OUT_DIR, exist_ok=True)
        plotting.safe_to_csv(df, TRACK_CSV, index=False)
        print("\n=== REALIZATION TRACKING (forecast vs actual) ===")
        print(df.drop(columns=["key"]).to_string(index=False))
        print("\nBest tracking method per product (lowest RMSE on realized):")
        for key, grp in df.groupby("key"):
            b = grp.loc[grp["RMSE"].idxmin()]
            print(f"  {b['product']:24s} -> {b['method']:16s} "
                  f"RMSE={b['RMSE']:.2f}  MAPE={b['MAPE']:.2f}%  "
                  f"(n={int(b['n_realized'])})")
        print(f"\nSaved -> {TRACK_CSV} and outputs/plots/<product>_realization.png")
    elif recorded_no_overlap:
        print("\nRealized data IS recorded, but after --refit your realized "
              "weeks are now part of history and the forecast restarts after "
              "them, so there is nothing new to score yet:")
        for k, n in recorded_no_overlap:
            print(f"  - {config.PRODUCTS[k].label}: {n} realized week(s) folded "
                  f"into history; forecast now starts at the new level.")
        print("Add prices for the NEW forecast weeks (as published) to resume "
              "tracking. The updated forecast is in outputs/forecasts/.")
    else:
        print("\nNo realized observations recorded yet. Use:")
        print("  python track_realization.py --init")
        print("  python track_realization.py --add urea 2026-06-11 651.2")


if __name__ == "__main__":
    main()
