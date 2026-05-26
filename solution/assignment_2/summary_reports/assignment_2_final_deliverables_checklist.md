# Assignment 2 Final Deliverables Checklist

## Required Output Files

| Deliverable | Location | Status |
|---|---|---|
| Expanded validation results | `solution/assignment_2/outputs/final_model_validation_results.csv` | Complete |
| 2025 holdout results | `solution/assignment_2/outputs/final_2025_holdout_results.csv` | Complete |
| Final model selection summary | `solution/assignment_2/outputs/final_model_selection_summary.md` | Complete |

## Required Report Coverage

| Topic | Primary report location |
|---|---|
| Leakage controls | `summary_reports/assignment_2_summary_report_draft.md`, sections `Validation Decision` and `Leakage Prevention Decisions` |
| Walk-forward validation | `summary_reports/assignment_2_summary_report_draft.md`, section `Validation Decision` |
| Model choices | `summary_reports/assignment_2_summary_report_draft.md`, sections `Baseline Models`, `Prototype Modeling Trials`, `Advanced And Research Trials`, and `Focused Final Model Set` |
| Final model selection | `outputs/final_model_selection_summary.md` and `summary_reports/assignment_2_final_recommendation_update.md` |
| Direct vs hierarchical tradeoff | `summary_reports/assignment_2_summary_report_draft.md`, section `Direct Vs Hierarchical Forecasting` |
| Why `lag_168h` won next-day | `summary_reports/assignment_2_summary_report_draft.md`, section `Focused 2025 Holdout Findings`, and `summary_reports/assignment_2_final_recommendation_update.md` |

## Current Final Interpretation

No final advanced ML model is selected yet. The focused 2025 holdout showed that `baseline_lag_168h` is the strongest current next-day operational benchmark. The retained ML models remain candidates for broader-scale validation, but they should not be declared final until they materially beat the weekly lag baseline on wider 2025 holdout coverage.

## Operational Wrapper

| Interface | Location | Status |
|---|---|---|
| Train CLI | `solution/assignment_2/train.py` | Complete |
| Predict CLI | `solution/assignment_2/predict.py` | Complete |
| Train + predict CLI | `solution/assignment_2/run_pipeline.py` | Complete |
| Next-day LightGBM config | `solution/assignment_2/configs/next_day_lightgbm.yaml` | Complete |
| Next-month RandomForest config | `solution/assignment_2/configs/next_month_rf.yaml` | Complete |
| Operational README | `solution/assignment_2/OPERATIONAL_README.md` | Complete |
| Optional Streamlit viewer | `solution/assignment_2/streamlit_app.py` | Complete |

Validated commands:

```bash
cd solution/assignment_2
python run_pipeline.py --config configs/next_day_lightgbm.yaml
python predict.py --config configs/next_day_lightgbm.yaml
python run_pipeline.py --config configs/next_month_rf.yaml
```

Example forecast outputs:

- `solution/assignment_2/outputs/operational/next_day_lightgbm/forecast_results.csv`
- `solution/assignment_2/outputs/operational/next_month_rf/forecast_results.csv`
