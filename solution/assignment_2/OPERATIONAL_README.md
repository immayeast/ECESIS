 # Assignment 2 Operational Forecast Wrapper

This folder contains a lightweight CLI interface for running Assignment 2 forecasts through the same leakage-safe feature and modeling pipeline used in the notebooks.

## Files

| File | Purpose |
|---|---|
| `configs/next_day_lightgbm.yaml` | Example next-day LightGBM forecast config |
| `configs/next_month_rf.yaml` | Example next-month RandomForest forecast config |
| `train.py` | Trains and saves a model artifact for a config |
| `predict.py` | Loads a saved model artifact and writes forecast outputs |
| `run_pipeline.py` | Runs training and prediction together |
| `streamlit_app.py` | Optional small viewer for generated forecast outputs |

## Commands

Run training and prediction together:

```bash
python run_pipeline.py --config configs/next_day_lightgbm.yaml
```

Run training only:

```bash
python train.py --config configs/next_day_lightgbm.yaml
```

Run inference only after a model artifact exists:

```bash
python predict.py --config configs/next_day_lightgbm.yaml
```

Run inference and train automatically if the artifact is missing:

```bash
python predict.py --config configs/next_day_lightgbm.yaml --train-if-missing
```

Optional Streamlit viewer:

```bash
streamlit run streamlit_app.py
```

## Outputs

Each config writes outputs under its configured `output_dir`, including:

- `forecast_results.csv`
- `<run_name>_forecast_report.md`
- `<run_name>_metrics.csv` when actual `pd` values are available
- `<run_name>_train_metadata.json`
- `<run_name>_clusters.csv`

## Leakage Controls Enforced

- Features are generated relative to `forecast_created_at`.
- Lag features whose source timestamp is after `forecast_created_at` are masked.
- Historical averages are computed from the training window only.
- KMeans clusters are fit from the training window only.
- No random train/test splitting is used.

## Scaling Notes

The example configs use a deterministic bus subset so they run quickly on a laptop. To scale up, increase or remove `data.n_buses`, provide explicit `data.bus_ids`, or configure zone filters. The same train/predict interface will still write a direct `forecast_results.csv`.
