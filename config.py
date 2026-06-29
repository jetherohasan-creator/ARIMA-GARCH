"""
Central configuration for the fertilizer global-price forecasting pipeline.

Everything that is product- or path-specific lives here so that adding a new
product (e.g. a real ``Ammonia`` sheet later) is a one-line change in the
``PRODUCTS`` dictionary, with no edits to the modelling code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
OUT_DIR = os.path.join(ROOT, "outputs")
FORECAST_DIR = os.path.join(OUT_DIR, "forecasts")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
REPORT_PATH = os.path.join(OUT_DIR, "report.md")
METRICS_PATH = os.path.join(OUT_DIR, "comparison_metrics.csv")

# Source workbook (weekly international reference prices, USD/Ton).
SOURCE_XLSX = os.path.join(
    DATA_DIR, "Data_Harga_Internasional_Untuk_MTM_28_Mei_2026.xlsx"
)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# Default forecast horizon in weeks (~6 months).
DEFAULT_HORIZON = 26

# Weekly cadence anchor; recovered automatically per sheet but kept here as the
# expected default (Thursday == 3).
EXPECTED_WEEKDAY = 3
WEEK_FREQ = "7D"


@dataclass
class Product:
    """Declarative description of one price series inside the workbook.

    Column positions are **0-based** (the workbook is read with ``header=None``
    so positional indexing is required — the header is multi-row and merged and
    cannot be used directly).
    """

    key: str                       # short id used in filenames
    label: str                     # human label for reports/plots
    sheet: str                     # worksheet name
    target_col: int                # 0-based column of the "Harga Acuan" series
    support_cols: dict = field(default_factory=dict)  # name -> 0-based col
    note: str = ""                 # free-form note surfaced in the report


# --------------------------------------------------------------------------- #
# Product registry
# --------------------------------------------------------------------------- #
# NB: Column positions verified directly against the workbook:
#   Urea  -> J = "Harga Acuan (USD/Ton)"            (0-based 9)
#   NPK   -> C = "Harga Acuan 15-10-12 (USD/Ton)"   (0-based 2)
#   ZA    -> G = "Harga Acuan (USD/Ton)"            (0-based 6)
#   TSP   -> F = "Harga Acuan (USD/Ton)"            (0-based 5)
PRODUCTS: dict[str, Product] = {
    "urea": Product(
        key="urea",
        label="Urea",
        sheet="Urea",
        target_col=9,
        support_cols={"prill_avg": 3, "granul_avg": 8},
        note="Harga Acuan FOB SEA. Prill & Granul averages available as support.",
    ),
    "npk": Product(
        key="npk",
        label="NPK 15-10-12",
        sheet="NPK",
        target_col=2,
        support_cols={"argus_raw": 1, "kakao": 3},
        note="NPK is a blend; data starts later (~278 obs). Supports "
        "materials regression vs Urea/ZA/TSP.",
    ),
    "za": Product(
        key="za",
        label="ZA (Ammonium Sulphate)",
        sheet="ZA",
        target_col=6,
        support_cols={"avg_benchmark": 5},
        note="Ammonium sulphate. NOT ammonia. Used as --ammonia-proxy only.",
    ),
    "tsp": Product(
        key="tsp",
        label="TSP",
        sheet="TSP",
        target_col=5,
        support_cols={"avg_benchmark": 4},
        note="Triple super phosphate (phosphate proxy for NPK).",
    ),
    # --- Ammonia placeholder -------------------------------------------------
    # No Ammonia sheet exists in the workbook. This entry is intentionally
    # commented out. To add it later, drop in the real sheet/columns here and
    # the rest of the pipeline picks it up automatically.
    # "ammonia": Product(
    #     key="ammonia", label="Ammonia", sheet="Ammonia",
    #     target_col=?, support_cols={}, note="Real ammonia series."),
}

# Default product set used when --products is not given.
DEFAULT_PRODUCTS = ["urea", "npk", "za", "tsp"]

# Materials available as NPK regressors (must be cleanable products).
NPK_MATERIAL_PRODUCTS = ["urea", "za", "tsp"]
