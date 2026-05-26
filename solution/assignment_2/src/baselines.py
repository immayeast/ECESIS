from __future__ import annotations

import pandas as pd


def bus_baseline_predictions(feature_df: pd.DataFrame) -> pd.DataFrame:
    out = feature_df.copy()
    out["prediction_lag_168h"] = out["lag_168h_pd"]
    out["prediction_historical_mean"] = out["historical_avg_bus_he_dow_pd"]
    if "lag_8760h_pd" in out.columns:
        out["prediction_lag_8760h"] = out["lag_8760h_pd"]
    return out


def long_baseline_frame(df: pd.DataFrame, actual_col: str = "pd") -> pd.DataFrame:
    id_cols = ["bus_unique_id", "zone_name", "timestamp", "date", "he"]
    pred_cols = [col for col in df.columns if col.startswith("prediction_")]
    rows = []
    for col in pred_cols:
        tmp = df[id_cols + [actual_col, col]].rename(
            columns={"bus_unique_id": "bus_id", actual_col: "actual_pd", col: "predicted_pd"}
        )
        tmp["model"] = col.replace("prediction_", "baseline_")
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
