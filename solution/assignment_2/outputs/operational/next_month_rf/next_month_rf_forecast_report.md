# Forecast Run Report: next_month_rf

- Model: `random_forest`
- Horizon: `next_month`
- Forecast created at: `2022-03-31 23:00:00`
- Rows: 5760
- Output: `/Users/kikkiliu/Documents/New project/ECESIS/solution/assignment_2/outputs/operational/next_month_rf/forecast_results.csv`

## Leakage Controls
- Prediction features are generated with a configured `forecast_created_at` boundary.
- Lag features whose source time is after the forecast boundary are masked.
- Historical averages and cluster labels come from the training window only.
- No random train/test splitting is used.

## Metrics
|   rows |     mae |    rmse |    wmape |
|-------:|--------:|--------:|---------:|
|   5284 | 39.8358 | 56.7859 | 0.172021 |