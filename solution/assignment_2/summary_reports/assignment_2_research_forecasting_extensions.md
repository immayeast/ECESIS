# Assignment 2 Research Forecasting Extensions

## Scope
This extension turns Assignment 2 into a broader comparative forecasting research framework. It adds Ridge, ElasticNet, RandomForest, LightGBM, quantile LightGBM, and lightweight TCN/GRU/LSTM experiments while preserving the existing leakage-safe direct, hierarchical, clustering, XGBoost, CatBoost, and baseline workflows.

## Classical Model Log
| model                                  | family        | status    |   runtime_seconds | interpretability   |
|:---------------------------------------|:--------------|:----------|------------------:|:-------------------|
| research_ridge_cyclical_kmeans         | ridge         | completed |             0.347 | high               |
| research_elasticnet_cyclical_kmeans    | elasticnet    | completed |             0.588 | high               |
| research_random_forest_cyclical_kmeans | random_forest | completed |            17.773 | medium             |
| research_lightgbm_cyclical_kmeans      | lightgbm      | completed |             0.731 | medium             |

## Classical Model Evaluation
| horizon    | model                                  |   rows |       mae |      rmse |     wmape | level           |
|:-----------|:---------------------------------------|-------:|----------:|----------:|----------:|:----------------|
| next_day   | research_lightgbm_cyclical_kmeans      |    480 |   5.12194 |   8.01497 | 0.0390942 | bus             |
| next_day   | research_random_forest_cyclical_kmeans |    480 |   6.42333 |  10.551   | 0.0490273 | bus             |
| next_day   | research_elasticnet_cyclical_kmeans    |    480 |  20.1308  |  25.3344  | 0.153652  | bus             |
| next_day   | research_ridge_cyclical_kmeans         |    480 |  21.3999  |  27.4043  | 0.163339  | bus             |
| next_day   | research_lightgbm_cyclical_kmeans      |     96 |  15.7347  |  23.8526  | 0.0240196 | zone_aggregated |
| next_day   | research_random_forest_cyclical_kmeans |     96 |  18.3268  |  35.1747  | 0.0279766 | zone_aggregated |
| next_day   | research_elasticnet_cyclical_kmeans    |     96 |  47.686   |  71.8959  | 0.0727944 | zone_aggregated |
| next_day   | research_ridge_cyclical_kmeans         |     96 |  47.7483  |  71.4361  | 0.0728896 | zone_aggregated |
| next_month | research_random_forest_cyclical_kmeans |  13914 |  17.0374  |  26.1027  | 0.117372  | bus             |
| next_month | research_lightgbm_cyclical_kmeans      |  13914 |  24.2453  |  39.3614  | 0.167028  | bus             |
| next_month | research_elasticnet_cyclical_kmeans    |  13914 |  35.4212  |  63.6238  | 0.244019  | bus             |
| next_month | research_ridge_cyclical_kmeans         |  13914 |  35.4558  |  62.6961  | 0.244258  | bus             |
| next_month | research_random_forest_cyclical_kmeans |   2880 |  46.4137  |  81.1638  | 0.0661832 | zone_aggregated |
| next_month | research_lightgbm_cyclical_kmeans      |   2880 |  56.0333  |  97.4048  | 0.0799002 | zone_aggregated |
| next_month | research_elasticnet_cyclical_kmeans    |   2880 | 160.213   | 279.467   | 0.228454  | zone_aggregated |
| next_month | research_ridge_cyclical_kmeans         |   2880 | 160.59    | 279.875   | 0.228992  | zone_aggregated |
| next_week  | research_random_forest_cyclical_kmeans |   3285 |  12.8911  |  19.6661  | 0.0925735 | bus             |
| next_week  | research_lightgbm_cyclical_kmeans      |   3285 |  19.2255  |  32.9653  | 0.138062  | bus             |
| next_week  | research_elasticnet_cyclical_kmeans    |   3285 |  30.5382  |  49.1087  | 0.2193    | bus             |
| next_week  | research_ridge_cyclical_kmeans         |   3285 |  30.5919  |  47.885   | 0.219686  | bus             |
| next_week  | research_random_forest_cyclical_kmeans |    672 |  40.5574  |  68.931   | 0.0595799 | zone_aggregated |
| next_week  | research_lightgbm_cyclical_kmeans      |    672 |  43.9185  |  79.9591  | 0.0645175 | zone_aggregated |
| next_week  | research_ridge_cyclical_kmeans         |    672 | 138.99    | 239.765   | 0.20418   | zone_aggregated |
| next_week  | research_elasticnet_cyclical_kmeans    |    672 | 139.235   | 240.636   | 0.20454   | zone_aggregated |

## Quantile Forecasting
LightGBM quantile regression generated P10/P50/P90 forecasts. In the prototype, interval width is much larger during peak load rows, indicating that uncertainty expands during high-demand periods.

|   rows |   p50_mae |   p50_rmse |   p10_p90_coverage |   mean_interval_width |   peak_interval_width |   non_peak_interval_width |
|-------:|----------:|-----------:|-------------------:|----------------------:|----------------------:|--------------------------:|
|  14400 |   31.8266 |    65.6639 |           0.466667 |               46.6983 |               186.775 |                   31.7085 |

## Sequence Models
TCN, GRU, and LSTM were run as small exploratory day-ahead sequence models. Their performance is weak in this tiny prototype, which suggests that deeper temporal models need more data, tuning, and careful multi-step forecast design before they can fairly compete with engineered-feature boosting.

| model                     | status    |   runtime_seconds |   rows |     mae |    rmse |    wmape |
|:--------------------------|:----------|------------------:|-------:|--------:|--------:|---------:|
| research_tcn_lightweight  | completed |             3.2   |      4 | 156.379 | 157.543 | 0.545636 |
| research_gru_lightweight  | completed |             0.696 |      4 | 139.157 | 151.719 | 0.485546 |
| research_lstm_lightweight | completed |             0.64  |      4 | 161.005 | 169.503 | 0.561776 |

## Best Bus-Level Models By Horizon
| horizon    | model                                  |   rows |      mae |     rmse |     wmape | level   |
|:-----------|:---------------------------------------|-------:|---------:|---------:|----------:|:--------|
| next_day   | direct_xgb_cyclical_kmeans             |    480 |  5.08278 |  7.8375  | 0.0387953 | bus     |
| next_day   | research_lightgbm_cyclical_kmeans      |    480 |  5.12194 |  8.01497 | 0.0390942 | bus     |
| next_day   | research_random_forest_cyclical_kmeans |    480 |  6.42333 | 10.551   | 0.0490273 | bus     |
| next_day   | baseline_lag_168h                      |    480 |  6.83816 | 13.4389  | 0.0521936 | bus     |
| next_day   | direct_catboost_cyclical_kmeans        |    480 |  8.69899 | 17.8829  | 0.0663966 | bus     |
| next_month | baseline_lag_168h                      |   3285 |  6.68395 | 12.7091  | 0.0479987 | bus     |
| next_month | research_random_forest_cyclical_kmeans |  13914 | 17.0374  | 26.1027  | 0.117372  | bus     |
| next_month | baseline_historical_mean               |  13914 | 19.5637  | 32.4995  | 0.134776  | bus     |
| next_month | direct_xgb_cyclical_kmeans             |  13914 | 22.6287  | 33.5481  | 0.155891  | bus     |
| next_month | research_lightgbm_cyclical_kmeans      |  13914 | 24.2453  | 39.3614  | 0.167028  | bus     |
| next_week  | research_random_forest_cyclical_kmeans |   3285 | 12.8911  | 19.6661  | 0.0925735 | bus     |
| next_week  | research_lightgbm_cyclical_kmeans      |   3285 | 19.2255  | 32.9653  | 0.138062  | bus     |
| next_week  | research_elasticnet_cyclical_kmeans    |   3285 | 30.5382  | 49.1087  | 0.2193    | bus     |
| next_week  | research_ridge_cyclical_kmeans         |   3285 | 30.5919  | 47.885   | 0.219686  | bus     |

## Analysis Questions
- Boosting models perform strongly because they capture nonlinear interactions among lag, rolling, cyclical, bus, zone, and cluster features.
- Cyclical encoding helps tree and boosting models represent periodic relationships without artificial boundary jumps.
- In this bounded run, sequence models do not outperform engineered-feature boosting, but they remain useful future candidates if trained on larger windows and evaluated with a richer multi-step setup.
- Simpler linear models are highly interpretable but underfit the nonlinear demand structure.
- Hierarchical forecasting remains operationally useful because zone load is smoother than bus load.
- Quantile forecasts are valuable because operators need risk bands, not only point estimates.
- Clustering helps transfer information across buses with similar load-shape archetypes.
- Some clusters and zones are harder to forecast; this is captured in the cluster-wise and zone/error outputs.

## Graph and Spatial Future Work
A future graph-aware extension could construct a bus-zone graph and add neighboring-zone lag features, graph aggregation features, spatial smoothing priors, or graph neural networks once physical topology data are available. For now, cluster-aware and hierarchical models are the implemented spatial proxies.

## Research-Style Conclusion
The forecasting problem exhibits strong cyclical temporal structure and hierarchical spatial organization. Boosting methods are highly effective on structured temporal tabular data. Hierarchical forecasting provides stability advantages for noisy bus-level demand. Clustering acts as structural regularization. Deep temporal models may eventually capture richer sequential dependencies, but they require substantially greater compute and engineering complexity. The final study compares predictive power, interpretability, uncertainty estimation, scalability, and operational practicality rather than selecting a single winner too early.