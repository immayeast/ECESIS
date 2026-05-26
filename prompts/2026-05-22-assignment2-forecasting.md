# Assignment 2 Forecasting Prompt - 2026-05-22

## User Prompt Summary

Implement Assignment 2 as a leakage-safe dual-path hourly bus-level load forecasting pipeline. Compare direct bus-level forecasting with zone forecast plus historical bus-share allocation. Add feature engineering, baselines, walk-forward models, evaluation, and reporting while keeping 2025 held out and avoiding random splitting.

## High-Impact AI Output Summary

- Added reusable Assignment 2 source modules for data loading, features, baselines, models, evaluation, and pipeline orchestration.
- Added notebooks for baseline/features, walk-forward model execution, and evaluation/reporting.
- Updated EDA with a zone-to-bus reconciliation sample.
- Generated requested outputs under `solution/assignment_2/outputs/`: direct forecasts, zone-allocation forecasts, baseline forecasts, evaluation summary, and walk-forward fold table.
- Preserved leakage controls: shifted rolling features, train-only historical averages, forecast boundary masking, chronological folds, and a 2025 final holdout plan.
