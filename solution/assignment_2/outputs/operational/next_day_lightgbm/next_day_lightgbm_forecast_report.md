# Forecast Run Report: next_day_lightgbm

- Model: `lightgbm`
- Horizon: `next_day`
- Forecast created at: `2022-03-31 23:00:00`
- Rows: 288
- Output: `/Users/kikkiliu/Documents/New project/ECESIS/solution/assignment_2/outputs/operational/next_day_lightgbm/forecast_results.csv`

## Leakage Controls
- Prediction features are generated with a configured `forecast_created_at` boundary.
- Lag features whose source time is after the forecast boundary are masked.
- Historical averages and cluster labels come from the training window only.
- No random train/test splitting is used.

## Metrics
|   rows |     mae |    rmse |     wmape |
|-------:|--------:|--------:|----------:|
|    288 | 7.20681 | 10.5479 | 0.0436781 |