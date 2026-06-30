"""
Ingest & clean the weekly international fertilizer price workbook.

The workbook has several well-hidden traps that this loader handles explicitly:

1. **Multi-row / merged headers** that differ per sheet -> we read with
   ``header=None`` and pick the target column by position (see ``config``).

2. **Mixed date storage in column A.** Some cells are real Excel ``datetime``
   objects, others are ``dd/mm/yy`` strings. Crucially, the *native datetime*
   cells were entered with **month and day swapped** (a US ``mm/dd`` mis-parse
   at data-entry time): only ~28% of them land on the series' modal weekday,
   while **100%** of the string cells do. We therefore:
       - parse string cells with ``dayfirst=True`` (reliable),
       - for datetime cells, keep as-is if already on the modal weekday,
         otherwise swap month<->day to recover the true date.
   This recovers the two leading "artifact" rows (``01/02/2020`` & ``01/09/2020``)
   as the genuine weeks 2020-01-02 and 2020-01-09 rather than discarding them.

3. **Off-grid / out-of-sequence rows** (a handful of un-recoverable dates, e.g.
   a stray 2025-06-18 at the tail of Urea) are dropped and logged.

4. **Gaps** are filled by short linear interpolation (<= 2 weeks); larger gaps
   are left as NaN and logged.

The output per product is a clean, gap-aware weekly series on a regular grid,
written to ``data/clean/<key>.csv``.
"""
from __future__ import annotations

import datetime as _dt
import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

log = logging.getLogger("data_loader")

MAX_INTERP_GAP = 2  # weeks of consecutive NaN we are willing to interpolate


@dataclass
class CleanResult:
    """Container for a cleaned product series plus cleaning diagnostics."""

    key: str
    label: str
    target: pd.Series                 # the cleaned "Harga Acuan" series (USD/Ton)
    support: pd.DataFrame             # aligned support columns (may be empty)
    modal_weekday: int
    n_raw: int
    n_offgrid_dropped: int
    n_duplicates_dropped: int
    n_parse_failed: int
    n_interpolated: int
    n_remaining_gaps: int
    grid_start: pd.Timestamp
    grid_end: pd.Timestamp


# --------------------------------------------------------------------------- #
# Date recovery
# --------------------------------------------------------------------------- #
def _recover_date(value, modal_weekday: int):
    """Recover the true date of one column-A cell.

    Returns ``pd.NaT`` if the cell is not a date at all (e.g. a header label).
    """
    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date)):
        d = _dt.date(value.year, value.month, value.day)
        if d.weekday() == modal_weekday:
            return pd.Timestamp(d)
        # Try the month<->day swap (only valid when both <= 12).
        if value.month <= 12 and value.day <= 12:
            try:
                swapped = _dt.date(value.year, value.day, value.month)
                if swapped.weekday() == modal_weekday:
                    return pd.Timestamp(swapped)
            except ValueError:
                pass
        # Leave as-is; it will be flagged as off-grid downstream.
        return pd.Timestamp(d)
    if isinstance(value, str) and "/" in value:
        return pd.to_datetime(value, dayfirst=True, errors="coerce")
    return pd.NaT


def _detect_modal_weekday(col_a: pd.Series) -> int:
    """Modal weekday inferred from the *string* date cells (the reliable ones)."""
    wds = []
    for v in col_a:
        if isinstance(v, str) and "/" in v:
            d = pd.to_datetime(v, dayfirst=True, errors="coerce")
            if pd.notna(d):
                wds.append(d.weekday())
    if not wds:  # fall back to all parseable cells
        for v in col_a:
            d = _recover_date(v, config.EXPECTED_WEEKDAY)
            if pd.notna(d):
                wds.append(d.weekday())
    return Counter(wds).most_common(1)[0][0] if wds else config.EXPECTED_WEEKDAY


# --------------------------------------------------------------------------- #
# Core per-product cleaning
# --------------------------------------------------------------------------- #
def load_product(prod: config.Product, xlsx_path: str | None = None) -> CleanResult:
    xlsx_path = xlsx_path or config.SOURCE_XLSX
    raw = pd.read_excel(xlsx_path, sheet_name=prod.sheet, header=None)

    col_a = raw.iloc[:, 0]
    modal = _detect_modal_weekday(col_a)

    # Build a frame of (recovered_date, target, *support) for genuine data rows.
    wanted_cols = {"target": prod.target_col, **prod.support_cols}
    records = []
    n_parse_failed = 0
    for _, row in raw.iterrows():
        a = row.iloc[0]
        # Only attempt rows that look like dates (skip header label rows).
        is_datey = isinstance(a, (pd.Timestamp, _dt.datetime, _dt.date)) or (
            isinstance(a, str) and "/" in a
        )
        if not is_datey:
            continue
        d = _recover_date(a, modal)
        if pd.isna(d):
            n_parse_failed += 1
            continue
        rec = {"date": d}
        for name, cidx in wanted_cols.items():
            rec[name] = row.iloc[cidx] if cidx < len(row) else np.nan
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    n_raw = len(df)

    # Drop off-grid (un-recoverable / out-of-sequence) rows.
    on_grid = df["date"].dt.weekday == modal
    n_offgrid = int((~on_grid).sum())
    if n_offgrid:
        log.warning(
            "[%s] dropping %d off-grid/out-of-sequence rows: %s",
            prod.key, n_offgrid,
            ", ".join(str(x.date()) for x in df.loc[~on_grid, "date"]),
        )
    df = df[on_grid]

    # Drop duplicate dates (keep last = most recent revision).
    dup_mask = df["date"].duplicated(keep="last")
    n_dups = int(dup_mask.sum())
    if n_dups:
        log.warning("[%s] dropping %d duplicate-date rows", prod.key, n_dups)
    df = df[~dup_mask].sort_values("date").set_index("date")

    # Coerce all value columns to numeric.
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Trim leading rows where the target is NaN (e.g. NPK starts late).
    first_valid = df["target"].first_valid_index()
    if first_valid is not None:
        df = df.loc[first_valid:]

    # Reindex onto a regular weekly grid.
    grid = pd.date_range(df.index.min(), df.index.max(), freq=config.WEEK_FREQ)
    df = df.reindex(grid)
    df.index.name = "date"

    # Interpolate short gaps in the target; log the rest.
    target = df["target"]
    gaps_before = int(target.isna().sum())
    interp = target.interpolate(method="linear", limit=MAX_INTERP_GAP,
                                limit_area="inside")
    n_interp = gaps_before - int(interp.isna().sum())
    n_remaining = int(interp.isna().sum())
    df["target"] = interp
    if n_interp:
        log.info("[%s] interpolated %d short gaps (<=%d wk)", prod.key,
                 n_interp, MAX_INTERP_GAP)
    if n_remaining:
        log.warning("[%s] %d gaps remain as NaN (gap > %d wk)", prod.key,
                    n_remaining, MAX_INTERP_GAP)

    # Final validity checks.
    assert df.index.is_monotonic_increasing, f"{prod.key}: index not monotonic"
    assert not df.index.has_duplicates, f"{prod.key}: duplicate index"

    support_cols = [c for c in df.columns if c != "target"]
    return CleanResult(
        key=prod.key,
        label=prod.label,
        target=df["target"].rename(prod.key),
        support=df[support_cols],
        modal_weekday=modal,
        n_raw=n_raw,
        n_offgrid_dropped=n_offgrid,
        n_duplicates_dropped=n_dups,
        n_parse_failed=n_parse_failed,
        n_interpolated=n_interp,
        n_remaining_gaps=n_remaining,
        grid_start=grid.min(),
        grid_end=grid.max(),
    )


def load_all(keys: list[str], xlsx_path: str | None = None) -> dict[str, CleanResult]:
    """Clean every requested product and persist to ``data/clean/<key>.csv``."""
    import os
    os.makedirs(config.CLEAN_DIR, exist_ok=True)
    results: dict[str, CleanResult] = {}
    for k in keys:
        if k not in config.PRODUCTS:
            log.error("Unknown product '%s' — skipping", k)
            continue
        res = load_product(config.PRODUCTS[k], xlsx_path)
        out = pd.concat([res.target, res.support], axis=1)
        path = os.path.join(config.CLEAN_DIR, f"{k}.csv")
        import plotting
        plotting.safe_to_csv(out, path)
        log.info(
            "[%s] cleaned -> %s | %d obs, %d valid, range %s..%s",
            k, path, len(res.target), int(res.target.notna().sum()),
            res.grid_start.date(), res.grid_end.date(),
        )
        results[k] = res
    return results


def load_from_clean_csv(key: str, path: str) -> CleanResult:
    """Build a CleanResult from a pre-cleaned CSV (e.g. an *_extended.csv that
    already merges workbook + realized actuals). Used by ``run.py --use-extended``.
    """
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df = df.asfreq(config.WEEK_FREQ)
    target = df[key] if key in df.columns else df.iloc[:, 0]
    support = df.drop(columns=[c for c in [key] if c in df.columns],
                      errors="ignore")
    prod = config.PRODUCTS[key]
    return CleanResult(
        key=key, label=prod.label, target=target.rename(key), support=support,
        modal_weekday=int(pd.Series(df.index.weekday).mode().iloc[0]),
        n_raw=len(df), n_offgrid_dropped=0, n_duplicates_dropped=0,
        n_parse_failed=0, n_interpolated=0,
        n_remaining_gaps=int(target.isna().sum()),
        grid_start=df.index.min(), grid_end=df.index.max(),
    )


def load_extended(keys: list[str]) -> dict[str, CleanResult]:
    """Load extended (clean+realized) series written by track_realization."""
    import os
    results = {}
    for k in keys:
        path = os.path.join(config.CLEAN_DIR, f"{k}_extended.csv")
        if not os.path.exists(path):
            log.warning("[%s] no extended CSV (%s) — falling back to workbook",
                        k, path)
            results[k] = load_product(config.PRODUCTS[k])
        else:
            results[k] = load_from_clean_csv(k, path)
            log.info("[%s] loaded extended series -> %s (%d obs, end %s)",
                     k, path, len(results[k].target),
                     results[k].grid_end.date())
    return results


def summarize(results: dict[str, CleanResult]) -> pd.DataFrame:
    """Tabular cleaning report for quick verification / the markdown report."""
    rows = []
    for r in results.values():
        rows.append({
            "product": r.label,
            "modal_weekday": r.modal_weekday,
            "raw_rows": r.n_raw,
            "offgrid_dropped": r.n_offgrid_dropped,
            "dups_dropped": r.n_duplicates_dropped,
            "parse_failed": r.n_parse_failed,
            "interpolated": r.n_interpolated,
            "remaining_gaps": r.n_remaining_gaps,
            "valid_obs": int(r.target.notna().sum()),
            "start": r.grid_start.date(),
            "end": r.grid_end.date(),
        })
    return pd.DataFrame(rows)
