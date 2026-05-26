# Final Model Selection Summary

## Scope Warning
These results execute the final 2023/2024 validation and 2025 holdout logic on a deterministic focused bus subset. They verify the leakage-safe selection workflow and provide model-direction evidence, but they should not be described as a full-system 331M-row final benchmark until scaled.

## Retained Models
| model                                  | reason                                                                                                 |
|:---------------------------------------|:-------------------------------------------------------------------------------------------------------|
| baseline_lag_168h                      | Strong weekly periodicity sanity baseline that serious ML models must beat.                            |
| baseline_historical_mean               | Stable, interpretable lower-complexity reference.                                                      |
| direct_xgb_cyclical_kmeans             | Best next-day prototype direct bus model with nonlinear temporal, lag, zone, and cluster interactions. |
| research_lightgbm_cyclical_kmeans      | Near-tied with XGBoost in prototype and likely scalable for larger tabular runs.                       |
| research_random_forest_cyclical_kmeans | Strong next-week and next-month prototype robustness check against boosting.                           |
| zone_allocated_hgb                     | Hierarchical stability-oriented zone forecast plus bus-share allocation alternative.                   |

## Deferred Models
| model_family                | reason                                                                                                |
|:----------------------------|:------------------------------------------------------------------------------------------------------|
| Ridge / ElasticNet          | Retained in report as interpretable baselines, but underfit nonlinear demand structure.               |
| CatBoost                    | Deferred from expensive validation because it underperformed XGBoost/LightGBM in prototype.           |
| Direct HistGradientBoosting | Useful prototype benchmark, but dominated by XGBoost/LightGBM.                                        |
| TCN / GRU / LSTM            | Moved to future work because prototype sequence performance is weak and requires more tuning/compute. |
| GMM as required feature     | Kept as interpretive clustering; KMeans remains the default final cluster feature.                    |
| Quantile LightGBM           | Optional uncertainty layer after point-forecast evaluation; does not block final model choice.        |

## Evaluation Scope
| fold_name                  | purpose             | scope                      |   train_rows |   validation_rows |   bus_count |   zone_count |
|:---------------------------|:--------------------|:---------------------------|-------------:|------------------:|------------:|-------------:|
| validate_2022_to_2023      | expanded_validation | deterministic_6_bus_subset |        52524 |             52518 |           6 |            2 |
| validate_2022_2023_to_2024 | expanded_validation | deterministic_6_bus_subset |       105042 |             52668 |           6 |            2 |
| holdout_2022_2024_to_2025  | final_holdout       | deterministic_6_bus_subset |       157710 |             52392 |           6 |            2 |

## 2025 Holdout Next-Day Bus Results
| fold_name                 | purpose       | train_start   | train_end   | validate_start   | validate_end   | horizon   | model                                  |   rows |     mae |     rmse |     wmape | level   |   runtime_seconds |
|:--------------------------|:--------------|:--------------|:------------|:-----------------|:---------------|:----------|:---------------------------------------|-------:|--------:|---------:|----------:|:--------|------------------:|
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_day  | baseline_lag_168h                      |    120 | 15.5234 |  25.8977 | 0.0504691 | bus     |           nan     |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_day  | direct_xgb_cyclical_kmeans             |    120 | 28.4541 |  36.4848 | 0.092509  | bus     |             1.355 |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_day  | research_random_forest_cyclical_kmeans |    120 | 33.2806 |  54.0646 | 0.108201  | bus     |            75.186 |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_day  | research_lightgbm_cyclical_kmeans      |    120 | 35.7221 |  56.2881 | 0.116138  | bus     |             1.555 |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_day  | baseline_historical_mean               |    120 | 65.1478 |  85.5261 | 0.211806  | bus     |           nan     |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_day  | zone_allocated_hgb                     |    120 | 91.9085 | 119.641  | 0.29881   | bus     |             1.362 |

## Stability Across 2023, 2024, and 2025
|   index | model                                  |     mean |       std |
|--------:|:---------------------------------------|---------:|----------:|
|       4 | research_random_forest_cyclical_kmeans | 0.1054   | 0.0167023 |
|       5 | zone_allocated_hgb                     | 0.399418 | 0.0912167 |
|       2 | direct_xgb_cyclical_kmeans             | 0.168546 | 0.0982664 |
|       3 | research_lightgbm_cyclical_kmeans      | 0.16762  | 0.100476  |
|       1 | baseline_lag_168h                      | 0.154991 | 0.202636  |
|       0 | baseline_historical_mean               | 0.481206 | 0.371676  |

## Selection Insight
On the focused 2025 holdout subset, `baseline_lag_168h` has the lowest next-day bus-level WMAPE (0.0505). Compared with `baseline_lag_168h` WMAPE (0.0505), the relative improvement is 0.0%.

The decision framework is valid: prototype results should only narrow the candidate set. Final recommendation should depend on 2025 holdout WMAPE, stability across 2023/2024/2025, ability to beat `lag_168h`, runtime/scalability, and operational defensibility.