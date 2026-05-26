from __future__ import annotations

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import yaml

from advanced_experiments import BUS_FEATURES_CLUSTERED
from baselines import bus_baseline_predictions, long_baseline_frame
from clustering import attach_cluster_features, cluster_bus_load_shapes
from evaluation import regression_metrics
from features import apply_forecast_boundary, build_bus_feature_frame, fit_group_mean
from load_data import find_repo_root, first_buses_from_first_batch, read_bus
from models import fit_predict_regressor
from pipeline import _prediction_frame


DEFAULT_CATEGORICAL = ["bus_unique_id", "zone_name", "kmeans_cluster"]


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(Path(path).resolve())
    return config


def resolve_assignment_root(config: dict) -> Path:
    if config.get("assignment_root"):
        return Path(config["assignment_root"]).expanduser().resolve()
    repo_root = find_repo_root()
    return repo_root / "solution" / "assignment_2"


def resolve_output_dir(config: dict) -> Path:
    root = resolve_assignment_root(config)
    output_dir = Path(config.get("output_dir", root / "outputs" / "operational"))
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_artifact_dir(config: dict) -> Path:
    root = resolve_assignment_root(config)
    artifact_dir = Path(config.get("artifact_dir", root / "model_artifacts"))
    if not artifact_dir.is_absolute():
        artifact_dir = root / artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def model_artifact_path(config: dict) -> Path:
    return resolve_artifact_dir(config) / f"{config['run_name']}.joblib"


def _years_between(start: str, end: str) -> list[int]:
    return list(range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))


def select_bus_ids(config: dict) -> list[str] | None:
    data = config.get("data", {})
    if data.get("bus_ids"):
        return [str(x) for x in data["bus_ids"]]
    n_buses = data.get("n_buses")
    if n_buses:
        return first_buses_from_first_batch(int(data.get("bus_selection_year", 2022)), n_buses=int(n_buses))
    return None


def _read_window(config: dict, start: str, end: str, bus_ids: list[str] | None) -> pd.DataFrame:
    data = config.get("data", {})
    years = _years_between(start, end)
    return read_bus(
        years,
        columns=["bus_unique_id", "bus_type", "base_kv", "zone_name", "pd", "pg", "date", "he"],
        start_date=start,
        end_date=end,
        bus_ids=bus_ids,
        zones=data.get("zones"),
    ).sort_values(["bus_unique_id", "timestamp"])


def _fit_clusters(train_features: pd.DataFrame, config: dict) -> pd.DataFrame:
    cluster_cfg = config.get("clustering", {})
    if not cluster_cfg.get("enabled", True):
        bus_meta = (
            train_features.groupby("bus_unique_id", as_index=False)
            .agg(zone=("zone_name", "first"), average_pd=("pd", "mean"))
            .rename(columns={"bus_unique_id": "bus_id"})
        )
        bus_meta["kmeans_cluster"] = "disabled"
        bus_meta["cluster_confidence"] = 0.0
        return bus_meta
    n_clusters = int(cluster_cfg.get("n_clusters", 4))
    n_clusters = min(max(2, n_clusters), max(2, train_features["bus_unique_id"].nunique()))
    result = cluster_bus_load_shapes(train_features, n_clusters=n_clusters, use_gmm=bool(cluster_cfg.get("use_gmm", False)))
    return result.bus_clusters


def _prepare_train_frame(config: dict, bus_ids: list[str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_cfg = config["train"]
    train_bus = _read_window(config, train_cfg["start_date"], train_cfg["end_date"], bus_ids)
    train_features = build_bus_feature_frame(train_bus)
    clusters = _fit_clusters(train_features, config)
    train_features = attach_cluster_features(train_features, clusters)
    return train_features, clusters


def train_from_config(config: dict) -> dict:
    start = time.perf_counter()
    bus_ids = select_bus_ids(config)
    train_features, clusters = _prepare_train_frame(config, bus_ids)
    model_cfg = config["model"]
    model_type = model_cfg["type"]
    artifact_path = model_artifact_path(config)
    artifact = {
        "config": config,
        "bus_ids": bus_ids,
        "clusters": clusters,
        "feature_cols": BUS_FEATURES_CLUSTERED,
        "categorical_cols": DEFAULT_CATEGORICAL,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }
    if model_type.startswith("baseline_"):
        artifact["model"] = None
    else:
        model, _ = fit_predict_regressor(
            train_features,
            train_features.head(1),
            feature_cols=BUS_FEATURES_CLUSTERED,
            categorical_cols=DEFAULT_CATEGORICAL,
            target_col="pd",
            model_name=model_type,
        )
        artifact["model"] = model
    joblib.dump(artifact, artifact_path)
    metadata = {
        "run_name": config["run_name"],
        "artifact_path": str(artifact_path),
        "model_type": model_type,
        "train_rows": int(len(train_features)),
        "bus_count": int(train_features["bus_unique_id"].nunique()),
        "zone_count": int(train_features["zone_name"].nunique()),
        "runtime_seconds": round(time.perf_counter() - start, 3),
    }
    output_dir = resolve_output_dir(config)
    (output_dir / f"{config['run_name']}_train_metadata.json").write_text(json.dumps(metadata, indent=2))
    clusters.to_csv(output_dir / f"{config['run_name']}_clusters.csv", index=False)
    return metadata


def _prepare_prediction_frame(config: dict, artifact: dict) -> pd.DataFrame:
    train_cfg = config["train"]
    pred_cfg = config["predict"]
    start = train_cfg["start_date"]
    end = pred_cfg["end_date"]
    bus = _read_window(config, start, end, artifact.get("bus_ids"))
    features = build_bus_feature_frame(bus)
    train = features[(features["date"] >= pd.Timestamp(train_cfg["start_date"])) & (features["date"] <= pd.Timestamp(train_cfg["end_date"]))].copy()
    pred = features[(features["date"] >= pd.Timestamp(pred_cfg["start_date"])) & (features["date"] <= pd.Timestamp(pred_cfg["end_date"]))].copy()
    history = fit_group_mean(train, ["bus_unique_id", "he", "day_of_week"], "pd", "historical_avg_bus_he_dow_pd")
    pred = pred.drop(columns=["historical_avg_bus_he_dow_pd"], errors="ignore").merge(
        history, on=["bus_unique_id", "he", "day_of_week"], how="left"
    )
    pred = apply_forecast_boundary(pred, pred_cfg["forecast_created_at"])
    pred = attach_cluster_features(pred, artifact["clusters"])
    return pred


def _filter_horizon(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    horizon = config.get("forecast", {}).get("horizon", "custom")
    start = pd.Timestamp(config["predict"]["start_date"])
    if horizon == "next_day":
        return frame[frame["date"] == start].copy()
    if horizon == "next_week":
        return frame[(frame["date"] >= start) & (frame["date"] <= start + pd.Timedelta(days=6))].copy()
    if horizon == "next_month":
        month_end = min(start + pd.offsets.MonthEnd(0), pd.Timestamp(config["predict"]["end_date"]))
        return frame[(frame["date"] >= start) & (frame["date"] <= month_end)].copy()
    return frame.copy()


def predict_from_config(config: dict, train_if_missing: bool = False) -> dict:
    artifact_path = model_artifact_path(config)
    if not artifact_path.exists():
        if train_if_missing:
            train_from_config(config)
        else:
            raise FileNotFoundError(f"Missing model artifact: {artifact_path}. Run train.py first.")
    artifact = joblib.load(artifact_path)
    pred_features = _prepare_prediction_frame(config, artifact)
    pred_features = _filter_horizon(pred_features, config)
    model_type = config["model"]["type"]
    if model_type == "baseline_lag_168h":
        wide = bus_baseline_predictions(pred_features)
        out = long_baseline_frame(wide)
        out = out[out["model"] == "baseline_lag_168h"].copy()
    elif model_type == "baseline_historical_mean":
        wide = bus_baseline_predictions(pred_features)
        out = long_baseline_frame(wide)
        out = out[out["model"] == "baseline_historical_mean"].copy()
    else:
        model = artifact["model"]
        pred = pred_features.copy()
        pred["predicted_pd"] = model.predict(pred[artifact["feature_cols"]]).clip(min=0)
        out = _prediction_frame(pred, config["run_name"])
    out["run_name"] = config["run_name"]
    out["forecast_horizon"] = config.get("forecast", {}).get("horizon", "custom")
    output_dir = resolve_output_dir(config)
    output_path = output_dir / config.get("forecast_results_file", "forecast_results.csv")
    out.to_csv(output_path, index=False)
    metrics = {}
    if "actual_pd" in out.columns and out["actual_pd"].notna().any():
        metrics = regression_metrics(out)
        pd.DataFrame([{**metrics, "run_name": config["run_name"], "model": model_type}]).to_csv(
            output_dir / f"{config['run_name']}_metrics.csv", index=False
        )
    report = [
        f"# Forecast Run Report: {config['run_name']}",
        "",
        f"- Model: `{model_type}`",
        f"- Horizon: `{config.get('forecast', {}).get('horizon', 'custom')}`",
        f"- Forecast created at: `{config['predict']['forecast_created_at']}`",
        f"- Rows: {len(out)}",
        f"- Output: `{output_path}`",
        "",
        "## Leakage Controls",
        "- Prediction features are generated with a configured `forecast_created_at` boundary.",
        "- Lag features whose source time is after the forecast boundary are masked.",
        "- Historical averages and cluster labels come from the training window only.",
        "- No random train/test splitting is used.",
    ]
    if metrics:
        report.extend(["", "## Metrics", pd.DataFrame([metrics]).to_markdown(index=False)])
    report_path = output_dir / f"{config['run_name']}_forecast_report.md"
    report_path.write_text("\n".join(report))
    return {"forecast_results": str(output_path), "report": str(report_path), "rows": len(out), **metrics}


def run_train_and_predict(config_path: str | Path) -> dict:
    config = load_config(config_path)
    train_meta = train_from_config(config)
    pred_meta = predict_from_config(config, train_if_missing=False)
    return {"train": train_meta, "predict": pred_meta}
