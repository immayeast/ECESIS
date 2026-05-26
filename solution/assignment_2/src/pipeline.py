from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from baselines import bus_baseline_predictions, long_baseline_frame
from evaluation import aggregate_bus_predictions_to_zone, make_fold_table, summarize_metrics
from features import (
    add_calendar_features,
    apply_forecast_boundary,
    attach_historical_bus_share,
    build_bus_feature_frame,
    build_zone_feature_frame,
    compute_historical_bus_share,
    fit_group_mean,
)
from load_data import assignment2_output_dir, first_buses_from_first_batch, read_bus
from models import fit_predict_regressor


TRAIN_START = "2022-01-01"
TRAIN_END = "2022-03-31"
VALIDATE_START = "2022-04-01"
VALIDATE_END = "2022-04-30"
FORECAST_CREATED_AT = "2022-03-31 23:00:00"


BUS_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "quarter",
    "day_of_year",
    "lag_24h_pd",
    "lag_168h_pd",
    "rolling_7d_mean_pd",
    "rolling_28d_mean_pd",
    "rolling_7d_std_pd",
    "rolling_28d_std_pd",
    "historical_avg_bus_he_dow_pd",
    "zone_total_pd_lag_24h",
    "zone_total_pd_lag_168h",
    "bus_unique_id",
    "zone_name",
]

ZONE_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "quarter",
    "day_of_year",
    "lag_24h_pd",
    "lag_168h_pd",
    "rolling_7d_mean_pd",
    "rolling_28d_mean_pd",
    "rolling_7d_std_pd",
    "rolling_28d_std_pd",
    "historical_avg_zone_he_dow_pd",
    "zone_name",
]


def _override_validation_history(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    group_cols: list[str],
    output_col: str,
) -> pd.DataFrame:
    mapping = fit_group_mean(train, group_cols, "pd", output_col)
    keep = validation.drop(columns=[output_col], errors="ignore")
    return keep.merge(mapping, on=group_cols, how="left")


def _make_horizon_frames(validation: pd.DataFrame) -> dict[str, pd.DataFrame]:
    validation = validation.copy()
    first_day = pd.Timestamp(VALIDATE_START)
    return {
        "next_day": validation[validation["date"] == first_day].copy(),
        "next_month": validation.copy(),
    }


def _prediction_frame(df: pd.DataFrame, model_name: str, pred_col: str = "predicted_pd") -> pd.DataFrame:
    cols = ["bus_unique_id", "zone_name", "timestamp", "date", "he", "pd", pred_col]
    out = df[cols].rename(columns={"bus_unique_id": "bus_id", "pd": "actual_pd", pred_col: "predicted_pd"})
    out["model"] = model_name
    return out


def build_prototype_dataset(n_buses: int = 50) -> tuple[pd.DataFrame, list[str]]:
    bus_ids = first_buses_from_first_batch(2022, n_buses=n_buses)
    bus = read_bus(
        [2022],
        columns=["bus_unique_id", "bus_type", "base_kv", "zone_name", "pd", "pg", "date", "he"],
        start_date=TRAIN_START,
        end_date=VALIDATE_END,
        bus_ids=bus_ids,
    )
    bus = bus.sort_values(["bus_unique_id", "timestamp"]).reset_index(drop=True)
    return bus, bus_ids


def direct_bus_model_predictions(bus_features: pd.DataFrame) -> pd.DataFrame:
    train = bus_features[bus_features["date"] <= pd.Timestamp(TRAIN_END)].copy()
    validation = bus_features[
        (bus_features["date"] >= pd.Timestamp(VALIDATE_START))
        & (bus_features["date"] <= pd.Timestamp(VALIDATE_END))
    ].copy()
    validation = _override_validation_history(
        train,
        validation,
        ["bus_unique_id", "he", "day_of_week"],
        "historical_avg_bus_he_dow_pd",
    )
    validation = apply_forecast_boundary(validation, FORECAST_CREATED_AT)
    _, pred = fit_predict_regressor(
        train,
        validation,
        BUS_FEATURES,
        categorical_cols=["bus_unique_id", "zone_name"],
        target_col="pd",
    )
    return pred


def baseline_predictions(bus_features: pd.DataFrame) -> pd.DataFrame:
    train = bus_features[bus_features["date"] <= pd.Timestamp(TRAIN_END)].copy()
    validation = bus_features[
        (bus_features["date"] >= pd.Timestamp(VALIDATE_START))
        & (bus_features["date"] <= pd.Timestamp(VALIDATE_END))
    ].copy()
    validation = _override_validation_history(
        train,
        validation,
        ["bus_unique_id", "he", "day_of_week"],
        "historical_avg_bus_he_dow_pd",
    )
    validation = apply_forecast_boundary(validation, FORECAST_CREATED_AT)
    baseline_wide = bus_baseline_predictions(validation)
    return long_baseline_frame(baseline_wide)


def zone_allocated_predictions(bus: pd.DataFrame) -> pd.DataFrame:
    bus = add_calendar_features(bus)
    train_bus = bus[bus["date"] <= pd.Timestamp(TRAIN_END)].copy()
    validation_bus = bus[
        (bus["date"] >= pd.Timestamp(VALIDATE_START))
        & (bus["date"] <= pd.Timestamp(VALIDATE_END))
    ].copy()

    zone = (
        bus.groupby(["zone_name", "date", "he", "timestamp"], as_index=False)["pd"]
        .sum()
        .sort_values(["zone_name", "timestamp"])
    )
    zone_features = build_zone_feature_frame(zone)
    train_zone = zone_features[zone_features["date"] <= pd.Timestamp(TRAIN_END)].copy()
    validation_zone = zone_features[
        (zone_features["date"] >= pd.Timestamp(VALIDATE_START))
        & (zone_features["date"] <= pd.Timestamp(VALIDATE_END))
    ].copy()
    validation_zone = _override_validation_history(
        train_zone,
        validation_zone,
        ["zone_name", "he", "day_of_week"],
        "historical_avg_zone_he_dow_pd",
    )
    validation_zone = apply_forecast_boundary(validation_zone, FORECAST_CREATED_AT)
    _, zone_pred = fit_predict_regressor(
        train_zone,
        validation_zone,
        ZONE_FEATURES,
        categorical_cols=["zone_name"],
        target_col="pd",
    )
    zone_pred = zone_pred.rename(columns={"pd": "actual_zone_pd", "predicted_pd": "predicted_zone_pd"})

    share = compute_historical_bus_share(train_bus)
    allocated = attach_historical_bus_share(validation_bus, share)
    allocated = allocated.merge(
        zone_pred[["zone_name", "timestamp", "predicted_zone_pd"]],
        on=["zone_name", "timestamp"],
        how="left",
    )
    allocated["historical_bus_share"] = allocated["historical_bus_share"].fillna(0)
    allocated["predicted_pd"] = allocated["predicted_zone_pd"] * allocated["historical_bus_share"]
    allocated["predicted_pd"] = allocated["predicted_pd"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
    return allocated


def add_horizons(predictions: pd.DataFrame) -> pd.DataFrame:
    frames = []
    first_day = pd.Timestamp(VALIDATE_START)
    for horizon, frame in [
        ("next_day", predictions[predictions["date"] == first_day].copy()),
        ("next_month", predictions.copy()),
    ]:
        frame["horizon"] = horizon
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def evaluate_all(direct: pd.DataFrame, allocated: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    all_bus = pd.concat([direct, allocated, baselines], ignore_index=True, sort=False)
    bus_metrics = summarize_metrics(all_bus, ["horizon", "model"])
    bus_metrics["level"] = "bus"
    zone_predictions = aggregate_bus_predictions_to_zone(
        all_bus.rename(columns={"bus_id": "bus_unique_id"})
    )
    zone_metrics = summarize_metrics(zone_predictions, ["horizon", "model"])
    zone_metrics["level"] = "zone_aggregated"
    return pd.concat([bus_metrics, zone_metrics], ignore_index=True, sort=False)


def run_prototype_pipeline(output_dir: Path | None = None, n_buses: int = 50) -> dict[str, pd.DataFrame]:
    output_dir = output_dir or assignment2_output_dir(name="outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    bus, bus_ids = build_prototype_dataset(n_buses=n_buses)
    bus_features = build_bus_feature_frame(bus)

    direct_pred = direct_bus_model_predictions(bus_features)
    direct_out = add_horizons(_prediction_frame(direct_pred, "direct_hgb"))

    allocated_pred = zone_allocated_predictions(bus)
    allocated_out = add_horizons(_prediction_frame(allocated_pred, "zone_allocated_hgb"))

    baseline_out = add_horizons(baseline_predictions(bus_features))

    evaluation_summary = evaluate_all(direct_out, allocated_out, baseline_out)
    walk_forward = make_fold_table()
    walk_forward["executed_in_prototype"] = walk_forward["fold_name"].eq("prototype_2022_q1_to_apr")
    walk_forward["prototype_bus_count"] = len(bus_ids)

    direct_out.to_csv(output_dir / "bus_forecast_direct.csv", index=False)
    allocated_out.to_csv(output_dir / "bus_forecast_zone_allocated.csv", index=False)
    baseline_out.to_csv(output_dir / "baseline_forecasts.csv", index=False)
    evaluation_summary.to_csv(output_dir / "evaluation_summary.csv", index=False)
    walk_forward.to_csv(output_dir / "walk_forward_results.csv", index=False)

    return {
        "bus_forecast_direct": direct_out,
        "bus_forecast_zone_allocated": allocated_out,
        "baseline_forecasts": baseline_out,
        "evaluation_summary": evaluation_summary,
        "walk_forward_results": walk_forward,
    }
