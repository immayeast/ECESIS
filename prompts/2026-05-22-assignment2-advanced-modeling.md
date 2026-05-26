# Assignment 2 Advanced Modeling Prompt - 2026-05-22

## User Prompt Summary

Extend Assignment 2 with advanced temporal encoding, load-shape clustering, clustering-enhanced forecasting features, comparative model analysis, visualizations, and reporting while preserving leakage-safe walk-forward validation and the existing direct/hierarchical pipeline.

## High-Impact AI Output Summary

- Added cyclical sine/cosine encodings for hour, day-of-week, day-of-year, and month.
- Added bus load-shape clustering using normalized hourly profiles, KMeans, and optional GMM.
- Integrated cluster labels and cluster context into optional forecasting features.
- Added advanced model comparison across raw-time HGB, cyclical HGB, cyclical XGBoost, XGBoost plus KMeans features, hierarchical allocation, and baselines.
- Initially logged CatBoost as unavailable instead of failing the workflow, then re-ran the advanced comparison after CatBoost became available.
- Generated advanced comparison outputs, cluster outputs, plots, and a markdown report.
- Added SHAP summary outputs for the XGBoost + KMeans clustered model.
