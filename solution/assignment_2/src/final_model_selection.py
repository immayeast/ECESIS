from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

from advanced_experiments import BUS_FEATURES_CLUSTERED
from baselines import bus_baseline_predictions, long_baseline_frame
from clustering import attach_cluster_features, cluster_bus_load_shapes
from evaluation import aggregate_bus_predictions_to_zone, summarize_metrics
from features import apply_forecast_boundary, build_bus_feature_frame, fit_group_mean
from load_data import first_buses_from_first_batch, read_bus
from models import fit_predict_regressor
from pipeline import _prediction_frame, zone_allocated_predictions


@dataclass(frozen=True)
class FinalFold:
    fold_name: str
    train_start: str
    train_end: str
    validate_start: str
    validate_end: str
    purpose: str


FINAL_FOLDS = [
    FinalFold("validate_2022_to_2023", "2022-01-01", "2022-12-31", "2023-01-01", "2023-12-31", "expanded_validation"),
    FinalFold("validate_2022_2023_to_2024", "2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31", "expanded_validation"),
    FinalFold("holdout_2022_2024_to_2025", "2022-01-01", "2024-12-31", "2025-01-01", "2025-12-31", "final_holdout"),
]


RETAINED_MODELS = pd.DataFrame(
    [
        {"model": "baseline_lag_168h", "reason": "Strong weekly periodicity sanity baseline that serious ML models must beat."},
        {"model": "baseline_historical_mean", "reason": "Stable, interpretable lower-complexity reference."},
        {"model": "direct_xgb_cyclical_kmeans", "reason": "Best next-day prototype direct bus model with nonlinear temporal, lag, zone, and cluster interactions."},
        {"model": "research_lightgbm_cyclical_kmeans", "reason": "Near-tied with XGBoost in prototype and likely scalable for larger tabular runs."},
        {"model": "research_random_forest_cyclical_kmeans", "reason": "Strong next-week and next-month prototype robustness check against boosting."},
        {"model": "zone_allocated_hgb", "reason": "Hierarchical stability-oriented zone forecast plus bus-share allocation alternative."},
    ]
)

DEFERRED_MODELS = pd.DataFrame(
    [
        {"model_family": "Ridge / ElasticNet", "reason": "Retained in report as interpretable baselines, but underfit nonlinear demand structure."},
        {"model_family": "CatBoost", "reason": "Deferred from expensive validation because it underperformed XGBoost/LightGBM in prototype."},
        {"model_family": "Direct HistGradientBoosting", "reason": "Useful prototype benchmark, but dominated by XGBoost/LightGBM."},
        {"model_family": "TCN / GRU / LSTM", "reason": "Moved to future work because prototype sequence performance is weak and requires more tuning/compute."},
        {"model_family": "GMM as required feature", "reason": "Kept as interpretive clustering; KMeans remains the default final cluster feature."},
        {"model_family": "Quantile LightGBM", "reason": "Optional uncertainty layer after point-forecast evaluation; does not block final model choice."},
    ]
)


def add_final_horizons(predictions: pd.DataFrame, validate_start: str, validate_end: str) -> pd.DataFrame:
    start = pd.Timestamp(validate_start)
    week_end = min(start + pd.Timedelta(days=6), pd.Timestamp(validate_end))
    month_end = min(start + pd.offsets.MonthEnd(0), pd.Timestamp(validate_end))
    frames = []
    for horizon, frame in [
        ("next_day", predictions[predictions["date"] == start].copy()),
        ("next_week", predictions[(predictions["date"] >= start) & (predictions["date"] <= week_end)].copy()),
        ("next_month", predictions[(predictions["date"] >= start) & (predictions["date"] <= month_end)].copy()),
    ]:
        frame["horizon"] = horizon
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _prediction_metrics(predictions: pd.DataFrame, fold: FinalFold) -> pd.DataFrame:
    bus_metrics = summarize_metrics(predictions, ["horizon", "model"])
    bus_metrics["level"] = "bus"
    zone = aggregate_bus_predictions_to_zone(predictions.rename(columns={"bus_id": "bus_unique_id"}))
    zone_metrics = summarize_metrics(zone, ["horizon", "model"])
    zone_metrics["level"] = "zone_aggregated"
    out = pd.concat([bus_metrics, zone_metrics], ignore_index=True)
    out.insert(0, "fold_name", fold.fold_name)
    out.insert(1, "purpose", fold.purpose)
    out.insert(2, "train_start", fold.train_start)
    out.insert(3, "train_end", fold.train_end)
    out.insert(4, "validate_start", fold.validate_start)
    out.insert(5, "validate_end", fold.validate_end)
    return out


def _per_zone_metrics(predictions: pd.DataFrame, fold: FinalFold) -> pd.DataFrame:
    pred = predictions.copy()
    pred["abs_error"] = (pred["predicted_pd"] - pred["actual_pd"]).abs()
    pred["abs_actual"] = pred["actual_pd"].abs()
    out = (
        pred.groupby(["horizon", "model", "zone_name"], as_index=False)
        .agg(rows=("actual_pd", "count"), mae=("abs_error", "mean"), abs_error_sum=("abs_error", "sum"), abs_actual_sum=("abs_actual", "sum"))
    )
    out["wmape"] = out["abs_error_sum"] / out["abs_actual_sum"].replace(0, np.nan)
    out.insert(0, "fold_name", fold.fold_name)
    return out.drop(columns=["abs_error_sum", "abs_actual_sum"])


def _per_cluster_metrics(predictions: pd.DataFrame, clusters: pd.DataFrame, fold: FinalFold) -> pd.DataFrame:
    pred = predictions.merge(clusters[["bus_id", "kmeans_cluster"]], on="bus_id", how="left")
    pred["abs_error"] = (pred["predicted_pd"] - pred["actual_pd"]).abs()
    pred["abs_actual"] = pred["actual_pd"].abs()
    out = (
        pred.groupby(["horizon", "model", "kmeans_cluster"], as_index=False)
        .agg(rows=("actual_pd", "count"), mae=("abs_error", "mean"), abs_error_sum=("abs_error", "sum"), abs_actual_sum=("abs_actual", "sum"))
    )
    out["wmape"] = out["abs_error_sum"] / out["abs_actual_sum"].replace(0, np.nan)
    out.insert(0, "fold_name", fold.fold_name)
    return out.drop(columns=["abs_error_sum", "abs_actual_sum"])


def _per_bus_distribution(predictions: pd.DataFrame, fold: FinalFold) -> pd.DataFrame:
    pred = predictions.copy()
    pred["abs_error"] = (pred["predicted_pd"] - pred["actual_pd"]).abs()
    bus = pred.groupby(["horizon", "model", "bus_id"], as_index=False)["abs_error"].mean()
    out = (
        bus.groupby(["horizon", "model"], as_index=False)["abs_error"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "bus_count", "mean": "bus_mae_mean", "median": "bus_mae_median"})
    )
    out = out.drop(columns=["index"], errors="ignore")
    out.insert(0, "fold_name", fold.fold_name)
    return out


def _prepare_fold_data(fold: FinalFold, bus_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = sorted(set(range(pd.Timestamp(fold.train_start).year, pd.Timestamp(fold.validate_end).year + 1)))
    bus = read_bus(
        years,
        columns=["bus_unique_id", "bus_type", "base_kv", "zone_name", "pd", "pg", "date", "he"],
        start_date=fold.train_start,
        end_date=fold.validate_end,
        bus_ids=bus_ids,
    ).sort_values(["bus_unique_id", "timestamp"])
    bus_features = build_bus_feature_frame(bus)
    train = bus_features[(bus_features["date"] >= pd.Timestamp(fold.train_start)) & (bus_features["date"] <= pd.Timestamp(fold.train_end))].copy()
    validation = bus_features[(bus_features["date"] >= pd.Timestamp(fold.validate_start)) & (bus_features["date"] <= pd.Timestamp(fold.validate_end))].copy()

    history = fit_group_mean(train, ["bus_unique_id", "he", "day_of_week"], "pd", "historical_avg_bus_he_dow_pd")
    validation = validation.drop(columns=["historical_avg_bus_he_dow_pd"], errors="ignore").merge(
        history, on=["bus_unique_id", "he", "day_of_week"], how="left"
    )
    forecast_created_at = pd.Timestamp(fold.train_end) + pd.Timedelta(hours=23)
    validation = apply_forecast_boundary(validation, forecast_created_at)

    cluster_result = cluster_bus_load_shapes(train, n_clusters=min(4, max(2, len(bus_ids) // 2)), use_gmm=False)
    train_clustered = attach_cluster_features(train, cluster_result.bus_clusters)
    validation_clustered = attach_cluster_features(validation, cluster_result.bus_clusters)
    return bus, train_clustered, validation_clustered, cluster_result.bus_clusters


def _baselines(validation: pd.DataFrame, fold: FinalFold) -> pd.DataFrame:
    wide = bus_baseline_predictions(validation)
    base = long_baseline_frame(wide)
    return add_final_horizons(base, fold.validate_start, fold.validate_end)


def _direct_model(train, validation, fold: FinalFold, model_name: str, output_model: str) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    _, pred = fit_predict_regressor(
        train,
        validation,
        BUS_FEATURES_CLUSTERED,
        categorical_cols=["bus_unique_id", "zone_name", "kmeans_cluster"],
        target_col="pd",
        model_name=model_name,
    )
    return add_final_horizons(_prediction_frame(pred, output_model), fold.validate_start, fold.validate_end), time.perf_counter() - start


def _hierarchical(bus: pd.DataFrame, fold: FinalFold) -> tuple[pd.DataFrame, float]:
    import pipeline as pipeline_module

    old = (
        pipeline_module.TRAIN_START,
        pipeline_module.TRAIN_END,
        pipeline_module.VALIDATE_START,
        pipeline_module.VALIDATE_END,
        pipeline_module.FORECAST_CREATED_AT,
    )
    try:
        pipeline_module.TRAIN_START = fold.train_start
        pipeline_module.TRAIN_END = fold.train_end
        pipeline_module.VALIDATE_START = fold.validate_start
        pipeline_module.VALIDATE_END = fold.validate_end
        pipeline_module.FORECAST_CREATED_AT = str(pd.Timestamp(fold.train_end) + pd.Timedelta(hours=23))
        start = time.perf_counter()
        pred = zone_allocated_predictions(bus)
        out = add_final_horizons(_prediction_frame(pred, "zone_allocated_hgb"), fold.validate_start, fold.validate_end)
        return out, time.perf_counter() - start
    finally:
        (
            pipeline_module.TRAIN_START,
            pipeline_module.TRAIN_END,
            pipeline_module.VALIDATE_START,
            pipeline_module.VALIDATE_END,
            pipeline_module.FORECAST_CREATED_AT,
        ) = old


def run_final_model_selection(output_dir: Path, n_buses: int = 10) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bus_ids = first_buses_from_first_batch(2022, n_buses=n_buses)
    validation_rows = []
    holdout_rows = []
    runtime_rows = []
    zone_rows = []
    cluster_rows = []
    bus_dist_rows = []
    scope_rows = []

    for fold in FINAL_FOLDS:
        bus, train, validation, clusters = _prepare_fold_data(fold, bus_ids)
        clusters.to_csv(output_dir / f"final_clusters_{fold.fold_name}.csv", index=False)
        predictions = []

        base = _baselines(validation, fold)
        predictions.append(base)
        runtime_rows.append({"fold_name": fold.fold_name, "model": "baseline_lag_168h / baseline_historical_mean", "runtime_seconds": 0.0})

        for model_name, output_model in [
            ("xgb", "direct_xgb_cyclical_kmeans"),
            ("lightgbm", "research_lightgbm_cyclical_kmeans"),
            ("random_forest", "research_random_forest_cyclical_kmeans"),
        ]:
            pred, runtime = _direct_model(train, validation, fold, model_name, output_model)
            predictions.append(pred)
            runtime_rows.append({"fold_name": fold.fold_name, "model": output_model, "runtime_seconds": round(runtime, 3)})

        hier, runtime = _hierarchical(bus, fold)
        predictions.append(hier)
        runtime_rows.append({"fold_name": fold.fold_name, "model": "zone_allocated_hgb", "runtime_seconds": round(runtime, 3)})

        all_pred = pd.concat(predictions, ignore_index=True, sort=False)
        metrics = _prediction_metrics(all_pred, fold)
        zone_rows.append(_per_zone_metrics(all_pred, fold))
        cluster_rows.append(_per_cluster_metrics(all_pred, clusters, fold))
        bus_dist_rows.append(_per_bus_distribution(all_pred, fold))
        scope_rows.append(
            {
                "fold_name": fold.fold_name,
                "purpose": fold.purpose,
                "scope": f"deterministic_{n_buses}_bus_subset",
                "train_rows": len(train),
                "validation_rows": len(validation),
                "bus_count": validation["bus_unique_id"].nunique(),
                "zone_count": validation["zone_name"].nunique(),
            }
        )
        if fold.purpose == "final_holdout":
            holdout_rows.append(metrics)
        else:
            validation_rows.append(metrics)

    validation_results = pd.concat(validation_rows, ignore_index=True)
    holdout_results = pd.concat(holdout_rows, ignore_index=True)
    runtime = pd.DataFrame(runtime_rows)
    per_zone = pd.concat(zone_rows, ignore_index=True)
    per_cluster = pd.concat(cluster_rows, ignore_index=True)
    bus_dist = pd.concat(bus_dist_rows, ignore_index=True)
    scope = pd.DataFrame(scope_rows)

    validation_results = validation_results.merge(runtime, on=["fold_name", "model"], how="left")
    holdout_results = holdout_results.merge(runtime, on=["fold_name", "model"], how="left")
    validation_results.to_csv(output_dir / "final_model_validation_results.csv", index=False)
    holdout_results.to_csv(output_dir / "final_2025_holdout_results.csv", index=False)
    runtime.to_csv(output_dir / "final_model_runtime.csv", index=False)
    per_zone.to_csv(output_dir / "final_per_zone_wmape.csv", index=False)
    per_cluster.to_csv(output_dir / "final_per_cluster_wmape.csv", index=False)
    bus_dist.to_csv(output_dir / "final_per_bus_error_distribution.csv", index=False)
    scope.to_csv(output_dir / "final_evaluation_scope.csv", index=False)
    RETAINED_MODELS.to_csv(output_dir / "final_retained_model_rationale.csv", index=False)
    DEFERRED_MODELS.to_csv(output_dir / "final_deferred_model_rationale.csv", index=False)
    _write_selection_summary(output_dir, validation_results, holdout_results, scope)
    return {
        "final_model_validation_results": validation_results,
        "final_2025_holdout_results": holdout_results,
        "final_evaluation_scope": scope,
    }


def _write_selection_summary(output_dir: Path, validation: pd.DataFrame, holdout: pd.DataFrame, scope: pd.DataFrame) -> None:
    bus_holdout = holdout[(holdout["level"] == "bus") & (holdout["horizon"] == "next_day")].sort_values("wmape")
    best = bus_holdout.iloc[0] if not bus_holdout.empty else None
    lag = bus_holdout[bus_holdout["model"] == "baseline_lag_168h"]
    lag_wmape = lag["wmape"].iloc[0] if not lag.empty else np.nan
    improvement = (lag_wmape - best["wmape"]) / lag_wmape if best is not None and lag_wmape else np.nan
    stability = (
        pd.concat([validation, holdout], ignore_index=True)
        .query("level == 'bus' and horizon == 'next_day'")
        .groupby("model", as_index=False)["wmape"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("std")
    )
    lines = [
        "# Final Model Selection Summary",
        "",
        "## Scope Warning",
        "These results execute the final 2023/2024 validation and 2025 holdout logic on a deterministic focused bus subset. They verify the leakage-safe selection workflow and provide model-direction evidence, but they should not be described as a full-system 331M-row final benchmark until scaled.",
        "",
        "## Retained Models",
        RETAINED_MODELS.to_markdown(index=False),
        "",
        "## Deferred Models",
        DEFERRED_MODELS.to_markdown(index=False),
        "",
        "## Evaluation Scope",
        scope.to_markdown(index=False),
        "",
        "## 2025 Holdout Next-Day Bus Results",
        bus_holdout.to_markdown(index=False),
        "",
        "## Stability Across 2023, 2024, and 2025",
        stability.to_markdown(index=False),
        "",
        "## Selection Insight",
    ]
    if best is not None:
        lines.append(
            f"On the focused 2025 holdout subset, `{best['model']}` has the lowest next-day bus-level WMAPE ({best['wmape']:.4f}). "
            f"Compared with `baseline_lag_168h` WMAPE ({lag_wmape:.4f}), the relative improvement is {improvement:.1%}."
        )
    lines.extend(
        [
            "",
            "The decision framework is valid: prototype results should only narrow the candidate set. Final recommendation should depend on 2025 holdout WMAPE, stability across 2023/2024/2025, ability to beat `lag_168h`, runtime/scalability, and operational defensibility.",
        ]
    )
    (output_dir / "final_model_selection_summary.md").write_text("\n".join(lines))
