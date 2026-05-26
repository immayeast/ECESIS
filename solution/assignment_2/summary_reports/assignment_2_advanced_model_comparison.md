# Assignment 2 Advanced Model Comparison

## Summary
The advanced Assignment 2 pipeline adds cyclical temporal encodings, bus load-shape clustering, CatBoost/XGBoost comparisons, and SHAP interpretation while preserving the leakage-safe walk-forward design. Clustering is used as structural feature engineering, not as a standalone forecasting model.

## Temporal Encoding
Hourly, weekday, day-of-year, and month cycles are represented with sine/cosine pairs. This lets HE24 sit close to HE1, Sunday close to Monday, and December close to January in model feature space.

## Clustering
Bus profiles are built from training-window average hourly `pd`, normalized to preserve load shape rather than scale, then clustered with KMeans and optionally GMM. GMM was available in the prototype and produced probabilistic assignments.

|   n_buses |   n_clusters |   kmeans_inertia |   kmeans_silhouette | gmm_available   |   gmm_aic |   gmm_bic |
|----------:|-------------:|-----------------:|--------------------:|:----------------|----------:|----------:|
|        20 |            4 |          88.9907 |            0.433128 | True            |  -1907.69 |  -614.237 |

## Experiment Log
| model                           | status    | feature_set   | uses_gmm   |   runtime_seconds |
|:--------------------------------|:----------|:--------------|:-----------|------------------:|
| direct_hgb_raw_time             | completed | raw_time      | False      |             1.027 |
| direct_hgb_cyclical             | completed | cyclical      | False      |             1.133 |
| direct_xgb_cyclical             | completed | cyclical      | False      |             0.396 |
| direct_xgb_cyclical_kmeans      | completed | clustered     | False      |             0.432 |
| direct_catboost_cyclical_kmeans | completed | clustered     | False      |             1.581 |
| zone_allocated_hgb              | completed | hierarchical  | False      |           nan     |
| bus_load_shape_clustering       | completed | kmeans_gmm    | True       |           nan     |

## Evaluation
| horizon    | model                           |   rows |       mae |     rmse |     wmape | level           |
|:-----------|:--------------------------------|-------:|----------:|---------:|----------:|:----------------|
| next_day   | direct_xgb_cyclical_kmeans      |    480 |   5.08278 |   7.8375 | 0.0387953 | bus             |
| next_day   | baseline_lag_168h               |    480 |   6.83816 |  13.4389 | 0.0521936 | bus             |
| next_day   | direct_catboost_cyclical_kmeans |    480 |   8.69899 |  17.8829 | 0.0663966 | bus             |
| next_day   | zone_allocated_hgb              |    480 |   9.91466 |  15.0315 | 0.0756755 | bus             |
| next_day   | direct_hgb_cyclical             |    480 |   9.96962 |  22.0407 | 0.076095  | bus             |
| next_day   | direct_hgb_raw_time             |    480 |  10.8861  |  23.7985 | 0.0830905 | bus             |
| next_day   | direct_xgb_cyclical             |    480 |  14.2293  |  28.1737 | 0.108608  | bus             |
| next_day   | baseline_historical_mean        |    480 |  17.9169  |  30.2283 | 0.136754  | bus             |
| next_day   | direct_xgb_cyclical_kmeans      |     96 |  13.0627  |  23.1213 | 0.0199408 | zone_aggregated |
| next_day   | direct_catboost_cyclical_kmeans |     96 |  17.3561  |  31.7273 | 0.0264947 | zone_aggregated |
| next_day   | baseline_lag_168h               |     96 |  20.0894  |  37.9006 | 0.0306673 | zone_aggregated |
| next_day   | zone_allocated_hgb              |     96 |  20.2586  |  32.2743 | 0.0309255 | zone_aggregated |
| next_day   | direct_hgb_cyclical             |     96 |  39.5359  |  64.2225 | 0.0603531 | zone_aggregated |
| next_day   | direct_hgb_raw_time             |     96 |  44.9771  |  71.8387 | 0.0686593 | zone_aggregated |
| next_day   | direct_xgb_cyclical             |     96 |  59.9482  |  91.9341 | 0.0915131 | zone_aggregated |
| next_day   | baseline_historical_mean        |     96 |  79.12    | 133.205  | 0.12078   | zone_aggregated |
| next_month | baseline_lag_168h               |   3285 |   6.68395 |  12.7091 | 0.0479987 | bus             |
| next_month | baseline_historical_mean        |  13914 |  19.5637  |  32.4995 | 0.134776  | bus             |
| next_month | direct_xgb_cyclical_kmeans      |  13914 |  22.6287  |  33.5481 | 0.155891  | bus             |
| next_month | zone_allocated_hgb              |  13914 |  24.3653  |  36.3403 | 0.167854  | bus             |
| next_month | direct_catboost_cyclical_kmeans |  13914 |  28.4704  |  55.2109 | 0.196135  | bus             |
| next_month | direct_hgb_cyclical             |  13914 |  54.5253  | 135.018  | 0.375629  | bus             |
| next_month | direct_hgb_raw_time             |  13914 |  54.908   | 136.952  | 0.378266  | bus             |
| next_month | direct_xgb_cyclical             |  13914 |  55.5303  | 134.289  | 0.382552  | bus             |
| next_month | direct_xgb_cyclical_kmeans      |   2880 |  49.0837  |  83.0592 | 0.0699905 | zone_aggregated |
| next_month | baseline_historical_mean        |   2880 |  68.5989  | 124.526  | 0.097818  | zone_aggregated |
| next_month | direct_catboost_cyclical_kmeans |   2880 |  73.5086  | 130.69   | 0.104819  | zone_aggregated |
| next_month | zone_allocated_hgb              |   2880 |  88.1262  | 149.562  | 0.125663  | zone_aggregated |
| next_month | direct_xgb_cyclical             |   2880 | 195.957   | 335.894  | 0.279423  | zone_aggregated |
| next_month | direct_hgb_cyclical             |   2880 | 196.758   | 335.902  | 0.280566  | zone_aggregated |
| next_month | direct_hgb_raw_time             |   2880 | 209.939   | 356.013  | 0.299361  | zone_aggregated |
| next_month | baseline_lag_168h               |   2880 | 549.304   | 982.805  | 0.783275  | zone_aggregated |

## SHAP Summary
SHAP was run for the XGBoost + KMeans clustered model on a bounded validation sample. The largest mean absolute SHAP signal is `cluster_avg_load`, followed by recent lag and zone-context features, which supports the idea that clustering contributes useful structural context in this prototype.

| feature                      |   mean_abs_shap |
|:-----------------------------|----------------:|
| cluster_avg_load             |       64.4677   |
| lag_24h_pd                   |       11.4662   |
| zone_total_pd_lag_24h        |        4.03708  |
| doy_sin                      |        2.59708  |
| lag_168h_pd                  |        1.71     |
| zone_total_pd_lag_168h       |        1.31095  |
| rolling_7d_std_pd            |        1.28428  |
| rolling_7d_mean_pd           |        1.06906  |
| historical_avg_bus_he_dow_pd |        0.836939 |
| dow_cos                      |        0.677469 |
| hour_sin                     |        0.55252  |
| bus_unique_id                |        0.503768 |
| hour_cos                     |        0.200697 |
| kmeans_cluster               |        0.158068 |
| dow_sin                      |        0.156821 |

## Comparative Summary
| category                      | model                      | basis                                                                |        value |
|:------------------------------|:---------------------------|:---------------------------------------------------------------------|-------------:|
| best_performing_next_day_bus  | direct_xgb_cyclical_kmeans | lowest next-day bus-level WMAPE                                      |   0.0387953  |
| most_stable_bus_wmape         | baseline_historical_mean   | lowest standard deviation of bus-level WMAPE across horizons         |   0.00139858 |
| most_interpretable            | baseline_lag_168h          | simple previous-week same-hour rule                                  | nan          |
| most_structurally_informative | direct_xgb_cyclical_kmeans | uses cyclical temporal encodings and demand-archetype cluster labels | nan          |

## Interpretation
Electrical load is cyclical because human routines, industrial schedules, weather, and grid operations repeat by hour, weekday, and season. Zone loads are smoother because aggregation cancels some local bus volatility. Direct bus models capture local detail but can be noisier. Hierarchical zone allocation can generalize well when zone-level behavior is more stable. Clustering can help models share information across buses with similar demand shapes, and GMM may capture overlapping archetypes better than hard KMeans assignments when profiles are not cleanly separated.

## Compute Notes
The prototype runs on a deterministic high-volume bus subset. CatBoost and XGBoost both ran in this environment. SHAP was limited to the XGBoost + KMeans variant because attribution cost can grow quickly when scaling to more buses and folds.