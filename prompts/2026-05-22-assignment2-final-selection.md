# Assignment 2 Final Model Selection Prompt - 2026-05-22

## User Prompt Summary

Narrow expanded validation and 2025 holdout testing to a focused final model set instead of testing every experimental model. Keep the strongest baselines, XGBoost, LightGBM, RandomForest, and hierarchical zone allocation; defer underperforming or exploratory models. Produce final validation, 2025 holdout, and model-selection summary outputs.

The user noted that ChatGPT helped compile their thoughts into this implementation prompt.

## High-Impact AI Output Summary

- Added `final_model_selection.py` to run focused 2023/2024 validation and 2025 holdout logic.
- Added retained and deferred model rationale outputs.
- Generated final validation and 2025 holdout result CSVs plus per-zone, per-cluster, per-bus, runtime, and scope diagnostics.
- Added `07_final_model_selection.ipynb` in artifact-reading mode.
- Confirmed the focused 2025 subset supports the caution that prototype winners should not be treated as final models.
