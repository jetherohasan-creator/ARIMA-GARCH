# Fertilizer Global Price Forecasting (ARIMA-GARCH · SARIMA · Linear Regression)

A modular Python pipeline that cleans, analyses and **projects global reference
prices (USD/Ton)** for fertilizer products from a weekly international price
workbook, and compares three forecasting methods per product.

| Method | What it captures |
|---|---|
| **SARIMA** | Mean dynamics (+ optional monthly seasonality, m=12) |
| **ARIMA-GARCH** | ARX mean on log-returns **+ conditional volatility**; intervals widen with volatility |
| **Linear Regression** | Trend / autoregressive lags / NPK-vs-materials baselines |

## Products

Configured in [`config.py`](config.py) via the `PRODUCTS` registry — adding a new
series is **one line**:

| Key | Product | Source sheet | Target column |
|---|---|---|---|
| `urea` | Urea | `Urea` | J — `Harga Acuan (USD/Ton)` |
| `npk` | NPK 15-10-12 | `NPK` | C — `Harga Acuan 15-10-12 (USD/Ton)` |
| `za` | ZA (ammonium sulphate) | `ZA` | G — `Harga Acuan (USD/Ton)` |
| `tsp` | TSP | `TSP` | F — `Harga Acuan (USD/Ton)` |

### ⚠️ Ammonia is not in the source data

The workbook contains **no Ammonia sheet**, so Ammonia is not modelled by
default. The architecture is parameterised so a real `Ammonia` sheet can be
dropped into `PRODUCTS` later with no code changes. The optional
`--ammonia-proxy=za` flag substitutes ZA as a **methodological placeholder
only** — ammonium sulphate is **not** ammonia, and the report carries an
explicit disclaimer.

## Install

```bash
pip install -r requirements.txt
```

`pmdarima` is optional: if it fails to import, SARIMA falls back to a
statsmodels AIC grid search automatically.

## Run

```bash
python run.py                         # all products, defaults
python run.py --lr-mode materials     # NPK regressed on Urea/ZA/TSP
python run.py --sarima-freq monthly   # seasonal SARIMA (m=12)
python run.py --ammonia-proxy za      # add ZA-as-Ammonia placeholder (warned)
python run.py --products urea npk --horizon 13 --no-backtest
```

### CLI options

| Flag | Default | Meaning |
|---|---|---|
| `--horizon` | `26` | forecast horizon in weeks (~6 months) |
| `--sarima-freq` | `weekly` | `weekly` (non-seasonal) or `monthly` (seasonal m=12) |
| `--lr-mode` | `trend` | `trend` · `lags` · `materials` (NPK only) |
| `--garch-vol` | `GARCH` | `GARCH` · `EGARCH` · `GJR` |
| `--ammonia-proxy` | — | `za` to use ZA as an Ammonia placeholder |
| `--products` | all | subset of product keys |
| `--no-backtest` | off | skip walk-forward CV (faster) |

## Pipeline

`run.py` orchestrates: **clean → EDA/stationarity → 3 models → walk-forward
backtest → forecasts/plots/metrics → report**.

### Data cleaning (the hard part)

Column A mixes native Excel `datetime` cells with `dd/mm/yy` strings. The
datetime cells were entered with **month/day swapped** (a US `mm/dd` mis-parse):
only ~28% land on the series' modal weekday vs **100%** of the string cells. The
loader recovers each datetime by swapping to the modal weekday, parses strings
with `dayfirst=True`, drops off-grid/duplicate rows, interpolates short gaps
(≤2 wk) and reindexes onto a regular weekly grid. See
[`data_loader.py`](data_loader.py) for the full rationale.

## Realization tracking (forecast vs actual)

Log the reference prices **as they are published each week** and score the
forecasts against reality with [`track_realization.py`](track_realization.py):

```bash
python track_realization.py --init                       # create input templates
python track_realization.py --add urea 2026-06-11 651.2  # log one realized price
python track_realization.py --add urea 11/06/2026 651.2  # dd/mm/yyyy also accepted
python track_realization.py                              # score forecasts vs realized
python track_realization.py --refit                      # roll forward & re-forecast
```

* Realized prices live in `data/realizations/<product>.csv` (columns
  `date,actual`); edit them by hand or via `--add`. Dates are snapped to the
  weekly grid.
* Plain run → `outputs/realization_tracking.csv` (RMSE/MAE/MAPE/sMAPE +
  interval hit-rate of each method on the realized window) and
  `outputs/plots/<product>_realization.png` (history + realized dots + each
  method's forecast and band), and prints which method is tracking best.
* `--refit` merges workbook + realized data into `data/clean/<key>_extended.csv`
  and re-runs `run.py --use-extended`, so the next forecast starts from the
  latest realized week.

## Outputs

```
data/clean/<product>.csv              # cleaned weekly series
outputs/forecasts/<product>_<method>.csv   # forecast + lower/upper interval
outputs/plots/<product>_<method>.png       # actual vs fitted vs forecast + band
outputs/plots/<product>_eda.png            # level / ACF / PACF / STL
outputs/comparison_metrics.csv             # RMSE/MAE/MAPE/sMAPE/coverage
outputs/report.md                          # executive summary + disclaimers
```

## Project layout

```
config.py          # PRODUCTS registry, paths, parameters
data_loader.py     # ingest & clean
eda.py             # ADF/KPSS, STL seasonality, diagnostic plots
models/
  sarima.py        # auto_arima / SARIMAX (+ grid-search fallback)
  arima_garch.py   # ARX + GARCH on log-returns (arch)
  linreg.py        # trend / lags / materials OLS
backtest.py        # rolling-origin walk-forward + metrics
run.py             # CLI orchestrator + report
```

## Methodological notes

- **ARIMA-GARCH intervals.** The 2021–22 price spike drives a near-IGARCH fit
  (α+β≈1) whose multi-step variance would otherwise diverge. When near-IGARCH is
  detected the per-step forecast variance is capped at the variance of the most
  recent ~2 years of returns (a more credible regime proxy than the
  spike-inflated full sample), and the t-quantile d.o.f. is floored at 5.
- **SARIMA weekly is non-seasonal** by design: an annual m=52 search is
  intractable and the EDA finds negligible weekly seasonality. Use
  `--sarima-freq=monthly` for the genuine seasonal variant.
- **`materials` mode** holds future Urea/ZA/TSP regressors at their last value
  (random-walk assumption).
- Forecasts are statistical projections, **not** price advice.
- Reproducible: `numpy`/`random` seeded (`config.SEED = 42`).
