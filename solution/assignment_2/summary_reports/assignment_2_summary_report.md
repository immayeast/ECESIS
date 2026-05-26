# Assignment 2 Summary Report Draft

## Executive Summary
Assignment 2 studies hourly bus-level load forecasting using large 2022-2025 parquet datasets. The work compares direct bus-level forecasting, hierarchical zone forecast plus bus-share allocation, baseline rules, cluster-aware boosting models, classical models, uncertainty experiments, and exploratory sequence models. The current conclusion is intentionally conservative: the prototype experiments identified promising model families, but focused 2025 holdout testing did not justify selecting a final advanced ML model yet. The weekly lag baseline remains the operational benchmark to beat.

## Data Structure And Forecasting Objective
Each bus is a measurement location, each zone is a group of buses, and `pd` is the load/demand target in MW. `HE` is hour-ending: `HE1` represents 00:00-00:59 and `HE24` represents 23:00-23:59. The modeling target is hourly bus-level `pd` for next-day, next-week where implemented, and next-month horizons. Zone-level data are also used to test whether smoother aggregate signals can improve stability through hierarchical forecasting.

## Validation Decision
I selected a walk-forward (rolling-origin) validation strategy because the forecasting task is inherently temporal and operationally sequential: future bus and zone loads must be predicted using only information available prior to the forecast timestamp. Unlike random train-test splitting, walk-forward validation preserves chronological order and prevents temporal leakage from future observations into the training set. Anotehr important reason would be the concern of huge computation cost to train on all the data altogether, since the dataset is so big I figured that taking a portion of the data to train and validate is good for drafting the model. The expanding-window “snowball” structure also mirrors real deployment conditions, where forecasting systems continuously accumulate historical information over time. Starting with smaller windows in early 2022 allowed efficient prototyping and leakage verification on large parquet datasets, while progressively expanding the training horizon enabled the model to learn broader weekly, seasonal, and yearly load behaviors. This approach provides a more realistic evaluation of model generalization under evolving grid demand patterns and operational conditions.

## Leakage Prevention Decisions
- Random train/test splitting was rejected because it would mix future observations into model training and create an unrealistic forecast setting (time-series data).
- Rolling features were shifted before rolling so the current target hour is not included in its own predictors.
- Historical averages and bus-share allocations are fitted from training windows only.
- KMeans clustering is fitted only on the training window for each fold before labels are attached to validation or holdout records.
- The 2025 year is treated as the final holdout period and should not influence model selection until final testing.

## EDA Findings That Guided Modeling
- Load has strong hourly cyclicality, so raw HE values alone can misrepresent the distance between HE24 and HE1.
- Weekday and weekend behavior differ, motivating day-of-week and weekend features.
- Annual and seasonal cycles are visible, motivating day-of-year and month encodings.
- Zone loads are smoother than bus loads (with less noise and less granularity), motivating a hierarchical zone-to-bus strategy.
- Bus-level loads are noisier but preserve local detail, motivating direct bus-level models and cluster-aware features.

## Feature Engineering Decisions
The core feature set includes calendar fields, lagged load, shifted rolling means and standard deviations, historical bus/HE/day-of-week averages, zone context features, and cluster labels. Cyclical sine/cosine encodings were added for hour, day-of-week, day-of-year, and month so periodic relationships are represented smoothly rather than as discontinuous integers. The sine/cosine behavior made me think of Gaussian Mixture Models (GMM).

## Baseline Models
Two baselines were kept throughout the study. `baseline_lag_168h` predicts the same hour from the previous week and is the strongest sanity check because load has a strong weekly cycle. `baseline_historical_mean` is a stable interpretable reference using historical averages. Any serious ML model should materially beat `baseline_lag_168h` on the holdout period before being recommended.

## Direct Vs Hierarchical Forecasting
Direct bus models predict bus-level demand directly and can capture local behavior, but they are more exposed to bus-level volatility. Hierarchical models first forecast zone-level load and allocate that load back to buses using historical shares. This is operationally meaningful because zone-level demand is smoother and may generalize better, but allocation can miss bus-specific changes when shares shift.

## Clustering Decision
KMeans clustering was added to group buses by normalized 24-hour load-shape profiles. The purpose was not to forecast with clusters alone, but to give direct models a structural context that transfers information across buses with similar operational demand archetypes. GMM clustering was explored as an interpretive method because it can represent overlapping archetypes, but KMeans remained the default model feature because it is simpler, stable, and easier to operationalize. This is a good result because KMeans achieved strong structural segmentation without the additional computational complexity and probabilistic instability sometimes associated with GMM-based clustering.

## Prototype Modeling Trials
The first prototype used 2022 Jan-Mar for training and 2022 Apr for validation. This was intentionally small so leakage rules and parquet handling could be verified quickly. Prototype results suggested that cluster-aware XGBoost and LightGBM were promising, while direct HistGradientBoosting, Ridge, ElasticNet, and deep sequence models were weaker. The prototype was treated only as a candidate-filtering stage, not as final evidence.

Prototype next-day bus-level results from the advanced comparison:

| horizon   | model                           |   rows |      mae |    rmse |     wmape | level   |
|:----------|:--------------------------------|-------:|---------:|--------:|----------:|:--------|
| next_day  | direct_xgb_cyclical_kmeans      |    480 |  5.08278 |  7.8375 | 0.0387953 | bus     |
| next_day  | baseline_lag_168h               |    480 |  6.83816 | 13.4389 | 0.0521936 | bus     |
| next_day  | direct_catboost_cyclical_kmeans |    480 |  8.69899 | 17.8829 | 0.0663966 | bus     |
| next_day  | zone_allocated_hgb              |    480 |  9.91466 | 15.0315 | 0.0756755 | bus     |
| next_day  | direct_hgb_cyclical             |    480 |  9.96962 | 22.0407 | 0.076095  | bus     |
| next_day  | direct_hgb_raw_time             |    480 | 10.8861  | 23.7985 | 0.0830905 | bus     |
| next_day  | direct_xgb_cyclical             |    480 | 14.2293  | 28.1737 | 0.108608  | bus     |
| next_day  | baseline_historical_mean        |    480 | 17.9169  | 30.2283 | 0.136754  | bus     |

Additional research next-day bus-level results:

| horizon   | model                                  |   rows |      mae |     rmse |     wmape | level   |
|:----------|:---------------------------------------|-------:|---------:|---------:|----------:|:--------|
| next_day  | research_lightgbm_cyclical_kmeans      |    480 |  5.12194 |  8.01497 | 0.0390942 | bus     |
| next_day  | research_random_forest_cyclical_kmeans |    480 |  6.42333 | 10.551   | 0.0490273 | bus     |
| next_day  | research_elasticnet_cyclical_kmeans    |    480 | 20.1308  | 25.3344  | 0.153652  | bus     |
| next_day  | research_ridge_cyclical_kmeans         |    480 | 21.3999  | 27.4043  | 0.163339  | bus     |

## Advanced And Research Trials
- XGBoost and LightGBM were tested because boosting models are well suited to nonlinear interactions among lags, calendar cycles, bus IDs, zones, and clusters.
- CatBoost was also tested, but it underperformed XGBoost/LightGBM in the prototype and was deferred from expensive final testing.
- RandomForest was retained as a robustness check because it performed surprisingly well on longer prototype horizons.
- Ridge and ElasticNet were kept in the report for interpretability but not prioritized because they underfit nonlinear demand structure.
- LightGBM quantile regression generated P10/P50/P90 intervals as an uncertainty layer, but it was treated as optional after point-forecast testing.
- TCN, GRU, and LSTM were moved to future work because the lightweight prototype performed weakly and deeper sequence models require more tuning, data volume, and compute.

## Focused Final Model Set
The expanded validation and 2025 holdout test were narrowed to a focused candidate set: `baseline_lag_168h`, `baseline_historical_mean`, `direct_xgb_cyclical_kmeans`, `research_lightgbm_cyclical_kmeans`, `research_random_forest_cyclical_kmeans`, and `zone_allocated_hgb`. This decision avoided spending compute on every exploratory model and aligned evaluation with the models that answered distinct operational questions: sanity baseline, stable baseline, best direct nonlinear model, scalable boosting model, robustness check, and hierarchical stability alternative.

Deferred models and rationale: Ridge/ElasticNet underfit, CatBoost underperformed in prototype, direct HGB was dominated by XGBoost/LightGBM, sequence models need future tuning, GMM is interpretive rather than required, and quantile LightGBM should follow after point-forecast selection.

## Expanded Validation And 2025 Holdout Scope
The final-selection workflow was executed on a deterministic focused bus subset to verify the expanded validation and 2025 holdout logic end-to-end. This should not be described as a full 331M-row system benchmark yet.

| fold_name                  | purpose             | scope                      |   train_rows |   validation_rows |   bus_count |   zone_count |
|:---------------------------|:--------------------|:---------------------------|-------------:|------------------:|------------:|-------------:|
| validate_2022_to_2023      | expanded_validation | deterministic_6_bus_subset |        52524 |             52518 |           6 |            2 |
| validate_2022_2023_to_2024 | expanded_validation | deterministic_6_bus_subset |       105042 |             52668 |           6 |            2 |
| holdout_2022_2024_to_2025  | final_holdout       | deterministic_6_bus_subset |       157710 |             52392 |           6 |            2 |

## Focused 2025 Holdout Findings
On the focused 2025 holdout subset, `baseline_lag_168h` had the best next-day bus-level WMAPE. This is important because it confirms the reason for not selecting the prototype winner as the final model. The ML candidates remain useful, but they did not materially beat the weekly lag baseline on this focused holdout test.

| fold_name                 | purpose       | train_start   | train_end   | validate_start   | validate_end   | horizon   | model                                  |   rows |     mae |     rmse |     wmape | level   |   runtime_seconds |
|:--------------------------|:--------------|:--------------|:------------|:-----------------|:---------------|:----------|:---------------------------------------|-------:|--------:|---------:|----------:|:--------|------------------:|
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_day  | baseline_lag_168h                      |    120 | 15.5234 |  25.8977 | 0.0504691 | bus     |           nan     |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_day  | direct_xgb_cyclical_kmeans             |    120 | 28.4541 |  36.4848 | 0.092509  | bus     |             1.355 |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_day  | research_random_forest_cyclical_kmeans |    120 | 33.2806 |  54.0646 | 0.108201  | bus     |            75.186 |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_day  | research_lightgbm_cyclical_kmeans      |    120 | 35.7221 |  56.2881 | 0.116138  | bus     |             1.555 |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_day  | baseline_historical_mean               |    120 | 65.1478 |  85.5261 | 0.211806  | bus     |           nan     |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_day  | zone_allocated_hgb                     |    120 | 91.9085 | 119.641  | 0.29881   | bus     |             1.362 |

## Current Decision
No final advanced ML model has been selected yet at this point. The current operational fallback and benchmark is `baseline_lag_168h`. The retained ML candidates should be scaled to broader bus coverage and additional folds before a final recommendation is made. A model should only be selected if it beats `baseline_lag_168h` on the 2025 holdout, remains stable across 2023 and 2024 validation, has acceptable runtime/scalability, and is explainable enough for operational use.

## Final Recommendation Update
## Verification
The expanded validation and focused 2025 holdout support the main scientific conclusion: the prototype winner did not generalize best to 2025. This validates the cautious walk-forward methodology and confirms that prototype results should be used to narrow candidates, not to choose the final model.

One nuance matters for defensibility: `research_random_forest_cyclical_kmeans` is the best retained direct ML model for the focused 2025 next-month holdout, but it is not the best overall next-month model in the current focused table. `baseline_lag_168h` has the lowest reported next-month WMAPE, though it only has rows where the lag feature is available under the strict forecast boundary. Among full-month models, `baseline_historical_mean` and `zone_allocated_hgb` are ahead of RandomForest. Therefore RandomForest should be described as the strongest direct-ML robustness candidate for longer horizons, not as the overall final next-month winner yet.

## Scenario Recommendations
| Scenario | Current recommendation | Evidence and interpretation |
|---|---|---|
| Next-day prediction | `baseline_lag_168h` | Best focused 2025 holdout next-day bus WMAPE. Weekly periodicity dominates short-horizon behavior and remains extremely difficult to beat robustly. |
| Longer-horizon direct ML candidate | `research_random_forest_cyclical_kmeans` | Most stable retained direct model across the focused validation/holdout evidence and best retained direct ML model on focused 2025 next-month. Useful robustness check against boosting. |
| Scalable production-style candidate | `research_lightgbm_cyclical_kmeans` | Fast runtime and scalable tabular learning. It did not win the focused holdout, but remains attractive for larger bus coverage and production-style retraining. |
| Research nonlinear benchmark | `direct_xgb_cyclical_kmeans` | Strongest prototype learner and valuable nonlinear benchmark, but less stable under expanded validation and 2025 holdout. |
| Zone/system-level strategy | `zone_allocated_hgb` concept, with refinement | Current implementation is not best for next-day accuracy, but hierarchical zone-to-bus forecasting remains structurally appropriate because zone loads are smoother and easier to operationalize. |
| Operational fallback | `baseline_lag_168h` | Until an ML model materially beats this baseline on broader 2025 holdout coverage, this remains the safest deployed benchmark. |

## Focused 2025 Next-Day Bus Results
| fold_name                 | purpose       | train_start   | train_end   | validate_start   | validate_end   | horizon   | model                                  |   rows |     mae |     rmse |     wmape | level   |   runtime_seconds |
|:--------------------------|:--------------|:--------------|:------------|:-----------------|:---------------|:----------|:---------------------------------------|-------:|--------:|---------:|----------:|:--------|------------------:|
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_day  | baseline_lag_168h                      |    120 | 15.5234 |  25.8977 | 0.0504691 | bus     |           nan     |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_day  | direct_xgb_cyclical_kmeans             |    120 | 28.4541 |  36.4848 | 0.092509  | bus     |             1.355 |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_day  | research_random_forest_cyclical_kmeans |    120 | 33.2806 |  54.0646 | 0.108201  | bus     |            75.186 |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_day  | research_lightgbm_cyclical_kmeans      |    120 | 35.7221 |  56.2881 | 0.116138  | bus     |             1.555 |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_day  | baseline_historical_mean               |    120 | 65.1478 |  85.5261 | 0.211806  | bus     |           nan     |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_day  | zone_allocated_hgb                     |    120 | 91.9085 | 119.641  | 0.29881   | bus     |             1.362 |

## Focused 2025 Next-Month Bus Results
| fold_name                 | purpose       | train_start   | train_end   | validate_start   | validate_end   | horizon    | model                                  |   rows |      mae |     rmse |    wmape | level   |   runtime_seconds |
|:--------------------------|:--------------|:--------------|:------------|:-----------------|:---------------|:-----------|:---------------------------------------|-------:|---------:|---------:|---------:|:--------|------------------:|
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_month | baseline_lag_168h                      |    840 |  39.9032 |  76.7432 | 0.114862 | bus     |           nan     |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_month | baseline_historical_mean               |   3720 | 121.918  | 186.906  | 0.317295 | bus     |           nan     |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_month | zone_allocated_hgb                     |   3720 | 145.35   | 217.224  | 0.378276 | bus     |             1.362 |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_month | research_random_forest_cyclical_kmeans |   3720 | 209.046  | 316.722  | 0.544046 | bus     |            75.186 |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_month | direct_xgb_cyclical_kmeans             |   3720 | 212.319  | 321.589  | 0.552564 | bus     |             1.355 |
| holdout_2022_2024_to_2025 | final_holdout | 2022-01-01    | 2024-12-31  | 2025-01-01       | 2025-12-31     | next_month | research_lightgbm_cyclical_kmeans      |   3720 | 216.625  | 328.665  | 0.563772 | bus     |             1.555 |

## Stability Summary Across 2023, 2024, And 2025
| horizon    | model                                  |     mean |       std |       min |      max |   count |
|:-----------|:---------------------------------------|---------:|----------:|----------:|---------:|--------:|
| next_day   | research_random_forest_cyclical_kmeans | 0.1054   | 0.0167023 | 0.0874739 | 0.120524 |       3 |
| next_day   | baseline_lag_168h                      | 0.154991 | 0.202636  | 0.0259572 | 0.388547 |       3 |
| next_day   | research_lightgbm_cyclical_kmeans      | 0.16762  | 0.100476  | 0.103319  | 0.283403 |       3 |
| next_day   | direct_xgb_cyclical_kmeans             | 0.168546 | 0.0982664 | 0.092509  | 0.279503 |       3 |
| next_day   | zone_allocated_hgb                     | 0.399418 | 0.0912167 | 0.29881   | 0.476722 |       3 |
| next_day   | baseline_historical_mean               | 0.481206 | 0.371676  | 0.211806  | 0.905234 |       3 |
| next_month | baseline_lag_168h                      | 0.106534 | 0.0534095 | 0.0494496 | 0.15529  |       3 |
| next_month | zone_allocated_hgb                     | 0.445713 | 0.0588211 | 0.378276  | 0.486441 |       3 |
| next_month | baseline_historical_mean               | 0.457539 | 0.180427  | 0.317295  | 0.661088 |       3 |
| next_month | direct_xgb_cyclical_kmeans             | 0.577804 | 0.0515038 | 0.543788  | 0.637059 |       3 |
| next_month | research_random_forest_cyclical_kmeans | 0.647326 | 0.171593  | 0.544046  | 0.845405 |       3 |
| next_month | research_lightgbm_cyclical_kmeans      | 0.656788 | 0.199093  | 0.521228  | 0.885365 |       3 |
| next_week  | baseline_lag_168h                      | 0.106534 | 0.0534095 | 0.0494496 | 0.15529  |       3 |
| next_week  | baseline_historical_mean               | 0.435132 | 0.222325  | 0.264164  | 0.686465 |       3 |
| next_week  | zone_allocated_hgb                     | 0.456213 | 0.130783  | 0.337265  | 0.596265 |       3 |
| next_week  | direct_xgb_cyclical_kmeans             | 0.474974 | 0.026147  | 0.45185   | 0.503347 |       3 |
| next_week  | research_random_forest_cyclical_kmeans | 0.498907 | 0.0348415 | 0.472854  | 0.538483 |       3 |
| next_week  | research_lightgbm_cyclical_kmeans      | 0.549552 | 0.141174  | 0.445813  | 0.710319 |       3 |

## Interpretation
The project discovered a realistic operational pattern: sophisticated ML models can win on bounded prototype windows, but strong periodic baselines may generalize better under true holdout conditions. XGBoost appearing best in the prototype and then losing to `lag_168h` in 2025 is not a failure; it is evidence that the validation design is doing its job. The result also supports keeping model choice scenario-specific rather than forcing one universal winner.

## Current Selection Status
No final advanced ML model should be declared yet. The current selected operational benchmark is `baseline_lag_168h` for next-day forecasting. RandomForest, LightGBM, XGBoost, and hierarchical allocation remain retained candidates for broader-scale testing and scenario-specific use, but each must be validated on wider bus and zone coverage before final selection.

