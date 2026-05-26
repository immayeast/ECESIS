from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(
    df: pd.DataFrame,
    actual_col: str = "actual_pd",
    pred_col: str = "predicted_pd",
) -> dict[str, float]:
    clean = df.dropna(subset=[actual_col, pred_col])
    if clean.empty:
        return {"rows": 0, "mae": np.nan, "rmse": np.nan, "wmape": np.nan}
    error = clean[pred_col] - clean[actual_col]
    denom = clean[actual_col].abs().sum()
    return {
        "rows": int(len(clean)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "wmape": float(error.abs().sum() / denom) if denom else np.nan,
    }


def summarize_metrics(
    predictions: pd.DataFrame,
    group_cols: list[str],
    actual_col: str = "actual_pd",
    pred_col: str = "predicted_pd",
) -> pd.DataFrame:
    rows = []
    for keys, grp in predictions.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(regression_metrics(grp, actual_col, pred_col))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_bus_predictions_to_zone(predictions: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "zone_name", "timestamp", "date", "he"]
    if "horizon" in predictions.columns:
        group_cols.insert(1, "horizon")
    return (
        predictions.groupby(group_cols, as_index=False)
        .agg(actual_pd=("actual_pd", "sum"), predicted_pd=("predicted_pd", "sum"))
    )


def make_fold_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold_name": "prototype_2022_q1_to_apr",
                "train_start": "2022-01-01",
                "train_end": "2022-03-31",
                "validate_start": "2022-04-01",
                "validate_end": "2022-04-30",
                "purpose": "prototype",
            },
            {
                "fold_name": "expanded_2022_jan_jun_to_jul",
                "train_start": "2022-01-01",
                "train_end": "2022-06-30",
                "validate_start": "2022-07-01",
                "validate_end": "2022-07-31",
                "purpose": "expanded_2022",
            },
            {
                "fold_name": "expanded_2022_jan_jul_to_aug",
                "train_start": "2022-01-01",
                "train_end": "2022-07-31",
                "validate_start": "2022-08-01",
                "validate_end": "2022-08-31",
                "purpose": "expanded_2022",
            },
            {
                "fold_name": "multi_year_2022_to_2023",
                "train_start": "2022-01-01",
                "train_end": "2022-12-31",
                "validate_start": "2023-01-01",
                "validate_end": "2023-12-31",
                "purpose": "multi_year_validation",
            },
            {
                "fold_name": "multi_year_2022_2023_to_2024",
                "train_start": "2022-01-01",
                "train_end": "2023-12-31",
                "validate_start": "2024-01-01",
                "validate_end": "2024-12-31",
                "purpose": "multi_year_validation",
            },
            {
                "fold_name": "final_2022_2024_to_2025",
                "train_start": "2022-01-01",
                "train_end": "2024-12-31",
                "validate_start": "2025-01-01",
                "validate_end": "2025-12-31",
                "purpose": "final_test",
            },
        ]
    )
