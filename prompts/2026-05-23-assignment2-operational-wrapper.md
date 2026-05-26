# Assignment 2 Operational Wrapper Prompt - 2026-05-23

## User Prompt Summary

Wrap up Assignment 2 with an operational-level interface so new data can run through the same pipeline and directly report forecast outputs. Add YAML configs, `run_pipeline.py`, `predict.py`, and `train.py`; minimum requirement is CLI-based inference, with optional Streamlit demo.

## High-Impact AI Output Summary

- Added `src/operational.py` to train, save, load, and predict from YAML configs while preserving leakage controls.
- Added CLI scripts: `train.py`, `predict.py`, and `run_pipeline.py`.
- Added example configs: `configs/next_day_lightgbm.yaml` and `configs/next_month_rf.yaml`.
- Added `OPERATIONAL_README.md` and optional `streamlit_app.py`.
- Validated both example configs and generated operational `forecast_results.csv` outputs.
