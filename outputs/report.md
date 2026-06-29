# Fertilizer Global Price Forecast — Report

_Generated for horizon = **26 weeks** (~6.0 months). SARIMA freq = `weekly`, LR mode = `trend`, GARCH vol = `GARCH`._

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
| Urea                   | SARIMA           | 105.03 |  99.76 |  17.35 |   20.07 |      80.77 |           4 |
| Urea                   | ARIMA-GARCH      | 113.06 | 106    |  19.12 |   22.21 |      78.85 |           4 |
| Urea                   | LinearRegression | 243.87 | 237.71 |  46.96 |   65.06 |      75    |           4 |
| NPK 15-10-12           | SARIMA           |  30.33 |  26.06 |   5.07 |    5.39 |      86.54 |           4 |
| NPK 15-10-12           | ARIMA-GARCH      |  28.64 |  24.57 |   4.79 |    5.05 |      80.77 |           4 |
| NPK 15-10-12           | LinearRegression | 173.86 | 170.19 |  34.96 |   42.92 |      46.15 |           4 |
| ZA (Ammonium Sulphate) | SARIMA           |  62.02 |  57.18 |  18    |   20.71 |      80.77 |           4 |
| ZA (Ammonium Sulphate) | ARIMA-GARCH      |  58.74 |  53.77 |  16.93 |   19.33 |      73.08 |           4 |
| ZA (Ammonium Sulphate) | LinearRegression | 165.79 | 161.07 |  55.78 |   80.89 |      65.38 |           4 |
| TSP                    | SARIMA           |  59.26 |  53    |  11.29 |   12.16 |     100    |           4 |
| TSP                    | ARIMA-GARCH      |  57.06 |  50.65 |  10.79 |   11.61 |      61.54 |           4 |
| TSP                    | LinearRegression | 181.59 | 177.32 |  37.56 |   48.6  |      69.23 |           4 |

### Best method per product (lowest RMSE)

| product                | best_method   |   RMSE |   MAPE% |
|:-----------------------|:--------------|-------:|--------:|
| NPK 15-10-12           | ARIMA-GARCH   |  28.64 |    4.79 |
| TSP                    | ARIMA-GARCH   |  57.06 |   10.79 |
| Urea                   | SARIMA        | 105.03 |   17.35 |
| ZA (Ammonium Sulphate) | ARIMA-GARCH   |  58.74 |   16.93 |

## 4. Selected models & forecast summary

| product                | method           | spec                                          |   forecast_end (+26w) |   lower |   upper |
|:-----------------------|:-----------------|:----------------------------------------------|----------------------:|--------:|--------:|
| Urea                   | SARIMA           | SARIMA(2, 1, 0)x(0, 0, 0, 0) [weekly]         |                 653.3 |   223.7 |  1082.8 |
| Urea                   | ARIMA-GARCH      | ARIMA-GARCH: ARX(lags=2) + GARCH(1,1), t-dist |                 582.9 |   301.3 |  1127.7 |
| Urea                   | LinearRegression | OLS price ~ t + t^2 (R²=0.069)                |                 371   |    32.8 |   709.3 |
| NPK 15-10-12           | SARIMA           | SARIMA(1, 1, 3)x(0, 0, 0, 0) [weekly]         |                 562.4 |   412.3 |   712.6 |
| NPK 15-10-12           | ARIMA-GARCH      | ARIMA-GARCH: ARX(lags=2) + GARCH(1,1), t-dist |                 559.9 |   471.5 |   664.9 |
| NPK 15-10-12           | LinearRegression | OLS price ~ t + t^2 (R²=0.193)                |                 343.7 |   147.2 |   540.3 |
| ZA (Ammonium Sulphate) | SARIMA           | SARIMA(0, 1, 0)x(0, 0, 0, 0) [weekly]         |                 408.6 |   214.4 |   602.8 |
| ZA (Ammonium Sulphate) | ARIMA-GARCH      | ARIMA-GARCH: ARX(lags=1) + GARCH(1,1), t-dist |                 411.5 |   271.5 |   623.8 |
| ZA (Ammonium Sulphate) | LinearRegression | OLS price ~ t + t^2 (R²=0.058)                |                 200.8 |     0   |   405.6 |
| TSP                    | SARIMA           | SARIMA(0, 1, 1)x(0, 0, 0, 0) [weekly]         |                 625.7 |   405.4 |   846   |
| TSP                    | ARIMA-GARCH      | ARIMA-GARCH: ARX(lags=2) + GARCH(1,1), t-dist |                 629.3 |   514.2 |   770.2 |
| TSP                    | LinearRegression | OLS price ~ t + t^2 (R²=0.200)                |                 393.5 |   129.7 |   657.2 |

## 5. Assumptions & notes

- Weekly grid anchored on the modal weekday (Thursday).
- ARIMA-GARCH models log-returns (ARX mean + GARCH variance, Student-t); intervals widen with conditional volatility.
- Linear `materials` mode (NPK) holds Urea/ZA/TSP at last value for future regressors (random-walk assumption).
- SARIMA `weekly` mode is non-seasonal for tractability; use `--sarima-freq=monthly` for the seasonal (m=12) variant.
- Forecasts are statistical projections, not price advice.
