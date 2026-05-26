# Assignment 2 Forecasting Pipeline Report

## Scope
Assignment 2 is implemented as a leakage-safe dual-path hourly bus-level load forecasting pipeline. The target is bus-level `pd` in MW. The two compared strategies are direct bus-level forecasting and zone forecast plus bus-share allocation.

## Data Structure
Each bus is a measurement location and each zone groups buses. `HE` is hour-ending, so `HE1` maps to 00:00-00:59 and `HE24` maps to 23:00-23:59. Bus files are high-volume bus-hour records; zone files are lower-volume zone-hour aggregate records.

## Leakage Controls
The pipeline uses chronological folds only, shifts rolling windows before aggregation, masks validation lag features whose source timestamp would be after `forecast_created_at`, fits validation historical averages from training rows only, and preserves 2025 as the final holdout period.

## Prototype Execution
The executable prototype uses the 2022 Jan-Mar training window and 2022 Apr validation window on a deterministic high-volume bus subset. This keeps runtime practical while preserving hourly bus-level granularity. The fold table records larger walk-forward and final-test windows for scaling.

## Results Snapshot
| horizon    | model                    |   rows |       mae |     rmse |     wmape | level           |
|:-----------|:-------------------------|-------:|----------:|---------:|----------:|:----------------|
| next_day   | baseline_lag_168h        |    480 |   6.83816 |  13.4389 | 0.0521936 | bus             |
| next_day   | zone_allocated_hgb       |    480 |   9.91466 |  15.0315 | 0.0756755 | bus             |
| next_day   | direct_hgb               |    480 |  10.8861  |  23.7985 | 0.0830905 | bus             |
| next_day   | baseline_historical_mean |    480 |  17.9169  |  30.2283 | 0.136754  | bus             |
| next_day   | baseline_lag_168h        |     96 |  20.0894  |  37.9006 | 0.0306673 | zone_aggregated |
| next_day   | zone_allocated_hgb       |     96 |  20.2586  |  32.2743 | 0.0309255 | zone_aggregated |
| next_day   | direct_hgb               |     96 |  44.9771  |  71.8387 | 0.0686593 | zone_aggregated |
| next_day   | baseline_historical_mean |     96 |  79.12    | 133.205  | 0.12078   | zone_aggregated |
| next_month | baseline_lag_168h        |   3285 |   6.68395 |  12.7091 | 0.0479987 | bus             |
| next_month | baseline_historical_mean |  13914 |  19.5637  |  32.4995 | 0.134776  | bus             |
| next_month | zone_allocated_hgb       |  13914 |  24.3653  |  36.3403 | 0.167854  | bus             |
| next_month | direct_hgb               |  13914 |  54.908   | 136.952  | 0.378266  | bus             |
| next_month | baseline_historical_mean |   2880 |  68.5989  | 124.526  | 0.097818  | zone_aggregated |
| next_month | zone_allocated_hgb       |   2880 |  88.1262  | 149.562  | 0.125663  | zone_aggregated |
| next_month | direct_hgb               |   2880 | 209.939   | 356.013  | 0.299361  | zone_aggregated |
| next_month | baseline_lag_168h        |   2880 | 549.304   | 982.805  | 0.783275  | zone_aggregated |

## Model Tradeoffs
Direct bus forecasting is more granular and can learn bus-specific behavior, but it is noisier. Zone forecasting is smoother and often more stable, but allocation depends on historical bus shares and can miss bus-specific shifts. Baselines remain the control group for deciding whether model complexity is justified.

## Next Steps
Scale the same pipeline to expanded 2022 folds, multi-year validation, and the final 2025 holdout. Consider LightGBM/XGBoost after validating that the leakage-safe baselines and HistGradientBoosting prototype are stable.