# Fertilizer Global Price Forecast — Report

_Generated for horizon = **26 weeks** (~6.0 months). SARIMA freq = `weekly`, LR mode = `trend`, GARCH vol = `GARCH`, shrink = `0.5`._

## 0. Why the errors look large (read first)

Backtest errors are measured at a **13-week** test horizon. These are volatile commodity prices: the average absolute price move over 13 weeks is ~22% (Urea), ~19% (ZA), ~14% (TSP), ~9% (NPK), so multi-month point-forecast error is dominated by inherent volatility, not model defects. Near-term (1–4 week) error is roughly half. A **Naive random-walk** benchmark is included as the reference: for efficient commodity prices it is hard to beat, so SARIMA/ARIMA-GARCH point forecasts are shrunk toward it (`--shrink`) to minimise error while keeping their volatility-aware intervals.

## ⚠️ Ammonia disclaimer

No Ammonia sheet exists in the source workbook, so Ammonia is **not modelled from real data**. When `--ammonia-proxy=za` is set, the ZA (ammonium sulphate) series is substituted purely as a methodological placeholder. **Ammonium sulphate is NOT ammonia** — its price level, volatility and drivers differ materially. Do not use the proxy output for any commercial or risk decision; replace it with a real ammonia feed.

## 1. Data cleaning

| product                |   modal_weekday |   raw_rows |   offgrid_dropped |   dups_dropped |   parse_failed |   interpolated |   remaining_gaps |   valid_obs | start      | end        |
|:-----------------------|----------------:|-----------:|------------------:|---------------:|---------------:|---------------:|-----------------:|------------:|:-----------|:-----------|
| Urea                   |               3 |        336 |                 2 |              0 |              0 |              3 |                0 |         336 | 2020-01-02 | 2026-06-04 |
| NPK 15-10-12           |               3 |        333 |                 4 |              0 |              0 |             10 |                0 |         284 | 2020-12-24 | 2026-05-28 |
| ZA (Ammonium Sulphate) |               3 |        335 |                 0 |              0 |              0 |              2 |                0 |         335 | 2020-01-02 | 2026-05-28 |
| TSP                    |               3 |        334 |                 0 |              0 |              0 |              4 |                0 |         335 | 2020-01-02 | 2026-05-28 |

Key cleaning steps: native-datetime cells in column A had month/day swapped (US mis-entry) and were recovered to the modal weekday; string cells parsed with `dayfirst=True`; off-grid / out-of-sequence rows dropped; short gaps (≤2 wk) linearly interpolated; series reindexed to a regular weekly grid.

## 2. EDA & stationarity

| product                |   ADF_lvl_p |   KPSS_lvl_p |   ADF_d1_p |   suggested_d |   seasonal_strength | has_seasonality   |
|:-----------------------|------------:|-------------:|-----------:|--------------:|--------------------:|:------------------|
| Urea                   |       0.136 |        0.1   |          0 |             1 |               0     | False             |
| NPK 15-10-12           |       0.221 |        0.019 |          0 |             1 |               0     | False             |
| ZA (Ammonium Sulphate) |       0.263 |        0.1   |          0 |             1 |               0     | False             |
| TSP                    |       0.583 |        0.04  |          0 |             1 |               0.117 | False             |

_ADF H0 = unit root (small p ⇒ stationary); KPSS H0 = stationary (small p ⇒ non-stationary)._

## 3. Backtest accuracy (walk-forward, rolling origin)

| product                | method           |   RMSE |    MAE |   MAPE |   sMAPE |   coverage |   n_origins |
|:-----------------------|:-----------------|-------:|-------:|-------:|--------:|-----------:|------------:|
| Urea                   | Naive            |  83.62 |  73.86 |  13.88 |   15.61 |      85.9  |           6 |
| Urea                   | SARIMA           |  83.55 |  73.84 |  13.77 |   15.53 |      88.46 |           6 |
| Urea                   | ARIMA-GARCH      |  85.8  |  76.14 |  14.47 |   16.38 |      87.18 |           6 |
| Urea                   | LinearRegression | 231.75 | 224.7  |  48.75 |   67.31 |      87.18 |           6 |
| NPK 15-10-12           | Naive            |  24.77 |  20.22 |   3.98 |    4.17 |      84.62 |           6 |
| NPK 15-10-12           | SARIMA           |  25.85 |  21.11 |   4.15 |    4.37 |      88.46 |           6 |
| NPK 15-10-12           | ARIMA-GARCH      |  24.69 |  20.15 |   3.96 |    4.16 |      84.62 |           6 |
| NPK 15-10-12           | LinearRegression | 168.76 | 166.19 |  34.57 |   42.37 |      42.31 |           6 |
| ZA (Ammonium Sulphate) | Naive            |  41.28 |  35.91 |  12.14 |   13.56 |      71.79 |           6 |
| ZA (Ammonium Sulphate) | SARIMA           |  41.28 |  35.91 |  12.14 |   13.56 |      91.03 |           6 |
| ZA (Ammonium Sulphate) | ARIMA-GARCH      |  40.74 |  35.27 |  11.97 |   13.3  |      84.62 |           6 |
| ZA (Ammonium Sulphate) | LinearRegression | 145.56 | 141.23 |  54.61 |   77.69 |      85.9  |           6 |
| TSP                    | Naive            |  29.84 |  25.52 |   5.6  |    5.92 |      87.18 |           6 |
| TSP                    | SARIMA           |  30.1  |  25.75 |   5.66 |    5.99 |     100    |           6 |
| TSP                    | ARIMA-GARCH      |  29.78 |  25.45 |   5.59 |    5.91 |      89.74 |           6 |
| TSP                    | LinearRegression | 169.78 | 166.97 |  37.74 |   48.16 |      88.46 |           6 |

### Best method per product (lowest RMSE)

| product                | best_method   |   RMSE |   MAPE% |
|:-----------------------|:--------------|-------:|--------:|
| NPK 15-10-12           | ARIMA-GARCH   |  24.69 |    3.96 |
| TSP                    | ARIMA-GARCH   |  29.78 |    5.59 |
| Urea                   | SARIMA        |  83.55 |   13.77 |
| ZA (Ammonium Sulphate) | ARIMA-GARCH   |  40.74 |   11.97 |

## 4. Selected models & forecast summary

| product                | method           | spec                                           |   forecast_end (+26w) |   lower |   upper |
|:-----------------------|:-----------------|:-----------------------------------------------|----------------------:|--------:|--------:|
| Urea                   | Naive            | Naive random walk (last value carried forward) |                 646.6 |   390.9 |  1069.4 |
| Urea                   | SARIMA           | SARIMA(2, 1, 0)x(0, 0, 0, 0) [weekly]          |                 649.9 |   220.4 |  1079.5 |
| Urea                   | ARIMA-GARCH      | ARIMA-GARCH: ARX(lags=2) + GARCH(1,1), t-dist  |                 614.7 |   333.1 |  1159.6 |
| Urea                   | LinearRegression | OLS price ~ t + t^2 (R²=0.069)                 |                 371   |    32.8 |   709.3 |
| NPK 15-10-12           | Naive            | Naive random walk (last value carried forward) |                 559.9 |   458.8 |   683.2 |
| NPK 15-10-12           | SARIMA           | SARIMA(1, 1, 3)x(0, 0, 0, 0) [weekly]          |                 561.2 |   411   |   711.3 |
| NPK 15-10-12           | ARIMA-GARCH      | ARIMA-GARCH: ARX(lags=2) + GARCH(1,1), t-dist  |                 559.9 |   471.5 |   664.8 |
| NPK 15-10-12           | LinearRegression | OLS price ~ t + t^2 (R²=0.193)                 |                 343.7 |   147.2 |   540.3 |
| ZA (Ammonium Sulphate) | Naive            | Naive random walk (last value carried forward) |                 408.6 |   297.6 |   561.1 |
| ZA (Ammonium Sulphate) | SARIMA           | SARIMA(0, 1, 0)x(0, 0, 0, 0) [weekly]          |                 408.6 |   214.4 |   602.8 |
| ZA (Ammonium Sulphate) | ARIMA-GARCH      | ARIMA-GARCH: ARX(lags=1) + GARCH(1,1), t-dist  |                 410.1 |   270   |   622.3 |
| ZA (Ammonium Sulphate) | LinearRegression | OLS price ~ t + t^2 (R²=0.058)                 |                 200.8 |     0   |   405.6 |
| TSP                    | Naive            | Naive random walk (last value carried forward) |                 625.9 |   536.6 |   730.1 |
| TSP                    | SARIMA           | SARIMA(0, 1, 1)x(0, 0, 0, 0) [weekly]          |                 625.8 |   405.5 |   846.1 |
| TSP                    | ARIMA-GARCH      | ARIMA-GARCH: ARX(lags=2) + GARCH(1,1), t-dist  |                 627.6 |   512.5 |   768.5 |
| TSP                    | LinearRegression | OLS price ~ t + t^2 (R²=0.200)                 |                 393.5 |   129.7 |   657.2 |

## 5. Assumptions & notes

- Weekly grid anchored on the modal weekday (Thursday).
- ARIMA-GARCH models log-returns (ARX mean + GARCH variance, Student-t); intervals widen with conditional volatility.
- Linear `materials` mode (NPK) holds Urea/ZA/TSP at last value for future regressors (random-walk assumption).
- SARIMA `weekly` mode is non-seasonal for tractability; use `--sarima-freq=monthly` for the seasonal (m=12) variant.
- Forecasts are statistical projections, not price advice.
