from __future__ import annotations

from pathlib import Path
import os
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from clustering import attach_cluster_features, cluster_bus_load_shapes
from evaluation import aggregate_bus_predictions_to_zone, summarize_metrics
from features import apply_forecast_boundary, build_bus_feature_frame, fit_group_mean
from models import fit_predict_regressor
from pipeline import (
    BUS_FEATURES,
    FORECAST_CREATED_AT,
    TRAIN_END,
    VALIDATE_END,
    VALIDATE_START,
    _override_validation_history,
    _prediction_frame,
    add_horizons,
    baseline_predictions,
    build_prototype_dataset,
    evaluate_all,
    zone_allocated_predictions,
)


CYCLICAL_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "month_sin", "month_cos"]
RAW_TIME_FEATURES = ["hour", "day_of_week", "is_weekend", "month", "quarter", "day_of_year"]
NON_TIME_BUS_FEATURES = [col for col in BUS_FEATURES if col not in RAW_TIME_FEATURES]
BUS_FEATURES_CYCLICAL = CYCLICAL_FEATURES + NON_TIME_BUS_FEATURES
BUS_FEATURES_CLUSTERED = BUS_FEATURES_CYCLICAL + ["kmeans_cluster", "cluster_confidence", "cluster_avg_load"]


def _validation_with_train_history(bus_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return train, validation


def _fit_direct_variant(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    model_name: str,
    output_model_name: str,
    feature_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    _, pred = fit_predict_regressor(
        train,
        validation,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        target_col="pd",
        model_name=model_name,
    )
    return add_horizons(_prediction_frame(pred, output_model_name))


def _plot_cluster_centers(centers: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    x = range(1, 25)
    for _, row in centers.iterrows():
        y = [row[f"he_{hour:02d}"] for hour in x]
        plt.plot(x, y, label=f"cluster {int(row['kmeans_cluster'])}")
    plt.title("KMeans normalized hourly load-shape centers")
    plt.xlabel("HE")
    plt.ylabel("normalized pd")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_kmeans_centers.png", dpi=150)
    plt.close()


def _plot_cluster_sizes(bus_clusters: pd.DataFrame, output_dir: Path) -> None:
    counts = bus_clusters["kmeans_cluster"].value_counts().sort_index()
    plt.figure(figsize=(7, 4))
    counts.plot(kind="bar")
    plt.title("KMeans cluster size distribution")
    plt.xlabel("cluster")
    plt.ylabel("bus count")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_size_distribution.png", dpi=150)
    plt.close()


def _plot_gmm_distribution(bus_clusters: pd.DataFrame, output_dir: Path) -> None:
    if "gmm_cluster" not in bus_clusters.columns or bus_clusters["gmm_cluster"].isna().all():
        return
    counts = bus_clusters["gmm_cluster"].value_counts().sort_index()
    plt.figure(figsize=(7, 4))
    counts.plot(kind="bar")
    plt.title("GMM cluster assignment distribution")
    plt.xlabel("GMM cluster")
    plt.ylabel("bus count")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_gmm_distribution.png", dpi=150)
    plt.close()


def _plot_actual_vs_predicted(predictions: pd.DataFrame, output_dir: Path) -> None:
    one_bus = predictions["bus_id"].iloc[0]
    subset = predictions[(predictions["bus_id"] == one_bus) & (predictions["horizon"] == "next_day")].sort_values(
        ["model", "timestamp"]
    )
    plt.figure(figsize=(11, 5))
    actual = subset.drop_duplicates("timestamp").sort_values("timestamp")
    plt.plot(actual["timestamp"], actual["actual_pd"], label="actual", linewidth=2)
    for model, grp in subset.groupby("model"):
        plt.plot(grp["timestamp"], grp["predicted_pd"], label=model, alpha=0.8)
    plt.title(f"Actual vs predicted next-day pd: {one_bus}")
    plt.xlabel("timestamp")
    plt.ylabel("pd")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted_next_day.png", dpi=150)
    plt.close()


def _plot_residuals(predictions: pd.DataFrame, output_dir: Path) -> None:
    frame = predictions.copy()
    frame["residual"] = frame["predicted_pd"] - frame["actual_pd"]
    plt.figure(figsize=(9, 5))
    for model, grp in frame.groupby("model"):
        grp["residual"].plot(kind="hist", bins=40, alpha=0.35, label=model)
    plt.title("Residual distributions")
    plt.xlabel("predicted - actual pd")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "residual_distributions.png", dpi=150)
    plt.close()


def _plot_model_wmape(evaluation: pd.DataFrame, output_dir: Path) -> None:
    bus = evaluation[evaluation["level"] == "bus"].copy()
    bus["label"] = bus["horizon"] + " | " + bus["model"]
    bus = bus.sort_values("wmape")
    plt.figure(figsize=(10, 6))
    plt.barh(bus["label"], bus["wmape"])
    plt.title("Bus-level WMAPE by horizon and model")
    plt.xlabel("WMAPE")
    plt.tight_layout()
    plt.savefig(output_dir / "model_wmape_comparison.png", dpi=150)
    plt.close()


def _plot_feature_importance(model, feature_cols: list[str], output_dir: Path, filename: str) -> None:
    fitted = model.named_steps.get("model")
    if not hasattr(fitted, "feature_importances_"):
        return
    importances = pd.DataFrame({"feature": feature_cols, "importance": fitted.feature_importances_[: len(feature_cols)]})
    importances = importances.sort_values("importance", ascending=False).head(20)
    importances.to_csv(output_dir / filename.replace(".png", ".csv"), index=False)
    plt.figure(figsize=(8, 6))
    plt.barh(importances["feature"][::-1], importances["importance"][::-1])
    plt.title("Feature importance")
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def _plot_shap_summary(model, sample_frame: pd.DataFrame, feature_cols: list[str], output_dir: Path, filename: str) -> None:
    try:
        import shap
    except Exception:
        return
    if sample_frame.empty:
        return
    sample = sample_frame[feature_cols].head(500)
    preprocess = model.named_steps.get("preprocess")
    fitted_model = model.named_steps.get("model")
    if preprocess is None or fitted_model is None:
        return
    transformed = preprocess.transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = []
    for name, _, cols in preprocess.transformers_:
        if name == "remainder":
            continue
        feature_names.extend(list(cols))
    if len(feature_names) != transformed.shape[1]:
        feature_names = [f"feature_{idx}" for idx in range(transformed.shape[1])]
    explainer = shap.TreeExplainer(fitted_model)
    shap_values = explainer.shap_values(transformed)
    shap_abs = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_abs.to_csv(output_dir / filename.replace(".png", ".csv"), index=False)
    plt.figure(figsize=(8, 6))
    top = shap_abs.head(20).iloc[::-1]
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.title("SHAP mean absolute impact")
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def _metrics_for_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    bus_metrics = summarize_metrics(predictions, ["horizon", "model"])
    bus_metrics["level"] = "bus"
    zone = aggregate_bus_predictions_to_zone(predictions.rename(columns={"bus_id": "bus_unique_id"}))
    zone_metrics = summarize_metrics(zone, ["horizon", "model"])
    zone_metrics["level"] = "zone_aggregated"
    return pd.concat([bus_metrics, zone_metrics], ignore_index=True)


def run_advanced_experiments(
    output_dir: Path,
    n_buses: int = 20,
    n_clusters: int = 4,
    use_gmm: bool = True,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bus, _ = build_prototype_dataset(n_buses=n_buses)
    bus_features = build_bus_feature_frame(bus)
    train, validation = _validation_with_train_history(bus_features)

    cluster_result = cluster_bus_load_shapes(train, n_clusters=n_clusters, use_gmm=use_gmm)
    cluster_result.bus_clusters.to_csv(output_dir / "bus_clusters.csv", index=False)
    cluster_result.cluster_stats.to_csv(output_dir / "bus_cluster_stats.csv", index=False)
    cluster_result.kmeans_centers.to_csv(output_dir / "bus_cluster_kmeans_centers.csv", index=False)
    cluster_result.cluster_diagnostics.to_csv(output_dir / "bus_cluster_diagnostics.csv", index=False)

    train_clustered = attach_cluster_features(train, cluster_result.bus_clusters)
    validation_clustered = attach_cluster_features(validation, cluster_result.bus_clusters)

    predictions = []
    experiment_rows = []
    feature_importance_models = []
    shap_models = []
    variants = [
        ("hgb", "direct_hgb_raw_time", BUS_FEATURES, ["bus_unique_id", "zone_name"], train, validation),
        ("hgb", "direct_hgb_cyclical", BUS_FEATURES_CYCLICAL, ["bus_unique_id", "zone_name"], train, validation),
        ("xgb", "direct_xgb_cyclical", BUS_FEATURES_CYCLICAL, ["bus_unique_id", "zone_name"], train, validation),
        (
            "xgb",
            "direct_xgb_cyclical_kmeans",
            BUS_FEATURES_CLUSTERED,
            ["bus_unique_id", "zone_name", "kmeans_cluster"],
            train_clustered,
            validation_clustered,
        ),
        (
            "catboost",
            "direct_catboost_cyclical_kmeans",
            BUS_FEATURES_CLUSTERED,
            ["bus_unique_id", "zone_name", "kmeans_cluster"],
            train_clustered,
            validation_clustered,
        ),
    ]

    for model_name, output_model_name, features, categorical, train_frame, validation_frame in variants:
        start = time.perf_counter()
        try:
            model, pred = fit_predict_regressor(
                train_frame,
                validation_frame,
                feature_cols=features,
                categorical_cols=categorical,
                target_col="pd",
                model_name=model_name,
            )
            out = add_horizons(_prediction_frame(pred, output_model_name))
            predictions.append(out)
            experiment_rows.append(
                {
                    "model": output_model_name,
                    "status": "completed",
                    "feature_set": "clustered" if "kmeans_cluster" in features else ("cyclical" if "hour_sin" in features else "raw_time"),
                    "uses_gmm": False,
                    "runtime_seconds": round(time.perf_counter() - start, 3),
                }
            )
            if model_name == "xgb":
                feature_importance_models.append((model, features, f"feature_importance_{output_model_name}.png"))
                if "kmeans_cluster" in features:
                    shap_models.append((model, validation_frame, features, f"shap_summary_{output_model_name}.png"))
        except Exception as exc:
            experiment_rows.append(
                {
                    "model": output_model_name,
                    "status": f"skipped: {type(exc).__name__}: {exc}",
                    "feature_set": "clustered" if "kmeans_cluster" in features else ("cyclical" if "hour_sin" in features else "raw_time"),
                    "uses_gmm": False,
                    "runtime_seconds": round(time.perf_counter() - start, 3),
                }
            )

    allocated = add_horizons(_prediction_frame(zone_allocated_predictions(bus), "zone_allocated_hgb"))
    baselines = add_horizons(baseline_predictions(bus_features))
    predictions.extend([allocated, baselines])
    all_predictions = pd.concat(predictions, ignore_index=True, sort=False)
    advanced_eval = _metrics_for_predictions(all_predictions)
    experiments = pd.DataFrame(experiment_rows)
    experiments.loc[len(experiments)] = {
        "model": "zone_allocated_hgb",
        "status": "completed",
        "feature_set": "hierarchical",
        "uses_gmm": False,
        "runtime_seconds": None,
    }
    experiments.loc[len(experiments)] = {
        "model": "bus_load_shape_clustering",
        "status": "completed",
        "feature_set": "kmeans_gmm" if cluster_result.gmm_available else "kmeans_only",
        "uses_gmm": cluster_result.gmm_available,
        "runtime_seconds": None,
    }

    all_predictions.to_csv(output_dir / "advanced_model_predictions.csv", index=False)
    advanced_eval.to_csv(output_dir / "advanced_evaluation_summary.csv", index=False)
    experiments.to_csv(output_dir / "advanced_model_experiment_log.csv", index=False)
    completed = advanced_eval[(advanced_eval["level"] == "bus") & (advanced_eval["horizon"] == "next_day")].copy()
    best = completed.sort_values("wmape").head(1)
    stable = advanced_eval[advanced_eval["level"] == "bus"].groupby("model", as_index=False)["wmape"].std().rename(
        columns={"wmape": "wmape_std_across_horizons"}
    )
    comparative = pd.DataFrame(
        [
            {
                "category": "best_performing_next_day_bus",
                "model": best["model"].iloc[0] if not best.empty else None,
                "basis": "lowest next-day bus-level WMAPE",
                "value": best["wmape"].iloc[0] if not best.empty else None,
            },
            {
                "category": "most_stable_bus_wmape",
                "model": stable.sort_values("wmape_std_across_horizons")["model"].iloc[0] if not stable.empty else None,
                "basis": "lowest standard deviation of bus-level WMAPE across horizons",
                "value": stable.sort_values("wmape_std_across_horizons")["wmape_std_across_horizons"].iloc[0]
                if not stable.empty
                else None,
            },
            {
                "category": "most_interpretable",
                "model": "baseline_lag_168h",
                "basis": "simple previous-week same-hour rule",
                "value": None,
            },
            {
                "category": "most_structurally_informative",
                "model": "direct_xgb_cyclical_kmeans",
                "basis": "uses cyclical temporal encodings and demand-archetype cluster labels",
                "value": None,
            },
        ]
    )
    comparative.to_csv(output_dir / "advanced_comparative_summary.csv", index=False)

    _plot_cluster_centers(cluster_result.kmeans_centers, output_dir)
    _plot_cluster_sizes(cluster_result.bus_clusters, output_dir)
    _plot_gmm_distribution(cluster_result.bus_clusters, output_dir)
    _plot_actual_vs_predicted(all_predictions, output_dir)
    _plot_residuals(all_predictions, output_dir)
    _plot_model_wmape(advanced_eval, output_dir)
    for model, features, filename in feature_importance_models:
        _plot_feature_importance(model, features, output_dir, filename)
    for model, sample_frame, features, filename in shap_models:
        _plot_shap_summary(model, sample_frame, features, output_dir, filename)

    return {
        "bus_clusters": cluster_result.bus_clusters,
        "bus_cluster_stats": cluster_result.cluster_stats,
        "advanced_model_predictions": all_predictions,
        "advanced_evaluation_summary": advanced_eval,
        "advanced_model_experiment_log": experiments,
    }
