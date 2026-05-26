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
import numpy as np
import pandas as pd

from advanced_experiments import BUS_FEATURES_CLUSTERED, _metrics_for_predictions
from clustering import attach_cluster_features, cluster_bus_load_shapes
from evaluation import regression_metrics, summarize_metrics
from features import apply_forecast_boundary, build_bus_feature_frame
from models import ModelSpec, fit_predict_regressor, make_lightgbm_quantile_regressor
from pipeline import (
    FORECAST_CREATED_AT,
    TRAIN_END,
    VALIDATE_END,
    VALIDATE_START,
    _override_validation_history,
    _prediction_frame,
    baseline_predictions,
    build_prototype_dataset,
)


RANDOM_STATE = 42


def add_research_horizons(predictions: pd.DataFrame) -> pd.DataFrame:
    first_day = pd.Timestamp(VALIDATE_START)
    first_week_end = first_day + pd.Timedelta(days=6)
    frames = []
    for horizon, frame in [
        ("next_day", predictions[predictions["date"] == first_day].copy()),
        ("next_week", predictions[(predictions["date"] >= first_day) & (predictions["date"] <= first_week_end)].copy()),
        ("next_month", predictions.copy()),
    ]:
        frame["horizon"] = horizon
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _train_validation_frames(n_buses: int = 20):
    bus, _ = build_prototype_dataset(n_buses=n_buses)
    bus_features = build_bus_feature_frame(bus)
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
    cluster_result = cluster_bus_load_shapes(train, n_clusters=4, use_gmm=True)
    train_clustered = attach_cluster_features(train, cluster_result.bus_clusters)
    validation_clustered = attach_cluster_features(validation, cluster_result.bus_clusters)
    return bus, bus_features, train_clustered, validation_clustered, cluster_result.bus_clusters


def _fit_variant(train, validation, model_name: str, output_model_name: str, feature_cols, categorical_cols):
    start = time.perf_counter()
    model, pred = fit_predict_regressor(
        train,
        validation,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        model_name=model_name,
    )
    out = add_research_horizons(_prediction_frame(pred, output_model_name))
    runtime = time.perf_counter() - start
    return model, out, runtime


def run_classical_model_experiments(output_dir: Path, n_buses: int = 20) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, train, validation, clusters = _train_validation_frames(n_buses=n_buses)
    variants = [
        ("ridge", "research_ridge_cyclical_kmeans"),
        ("elasticnet", "research_elasticnet_cyclical_kmeans"),
        ("random_forest", "research_random_forest_cyclical_kmeans"),
        ("lightgbm", "research_lightgbm_cyclical_kmeans"),
    ]
    predictions = []
    log_rows = []
    categorical = ["bus_unique_id", "zone_name", "kmeans_cluster"]
    for model_name, output_model_name in variants:
        try:
            model, pred, runtime = _fit_variant(
                train,
                validation,
                model_name,
                output_model_name,
                BUS_FEATURES_CLUSTERED,
                categorical,
            )
            predictions.append(pred)
            log_rows.append(
                {
                    "model": output_model_name,
                    "family": model_name,
                    "status": "completed",
                    "runtime_seconds": round(runtime, 3),
                    "interpretability": "high" if model_name in {"ridge", "elasticnet"} else "medium",
                }
            )
            if model_name in {"ridge", "elasticnet"}:
                _export_linear_coefficients(model, BUS_FEATURES_CLUSTERED, output_dir, output_model_name)
        except Exception as exc:
            log_rows.append(
                {
                    "model": output_model_name,
                    "family": model_name,
                    "status": f"skipped: {type(exc).__name__}: {exc}",
                    "runtime_seconds": np.nan,
                    "interpretability": "unknown",
                }
            )
    all_predictions = pd.concat(predictions, ignore_index=True, sort=False) if predictions else pd.DataFrame()
    evaluation = _metrics_for_predictions(all_predictions) if not all_predictions.empty else pd.DataFrame()
    log = pd.DataFrame(log_rows)
    all_predictions.to_csv(output_dir / "research_classical_model_predictions.csv", index=False)
    evaluation.to_csv(output_dir / "research_classical_evaluation_summary.csv", index=False)
    log.to_csv(output_dir / "research_classical_model_log.csv", index=False)
    clusters.to_csv(output_dir / "research_cluster_assignments.csv", index=False)
    _plot_model_runtime(log, output_dir)
    return {
        "research_classical_model_predictions": all_predictions,
        "research_classical_evaluation_summary": evaluation,
        "research_classical_model_log": log,
    }


def _export_linear_coefficients(model, feature_cols: list[str], output_dir: Path, model_name: str) -> None:
    fitted = model.named_steps["model"]
    coef = getattr(fitted, "coef_", None)
    if coef is None:
        return
    names = []
    preprocessor = model.named_steps["preprocess"]
    for name, _, cols in preprocessor.transformers_:
        if name != "remainder":
            names.extend(list(cols))
    if len(names) != len(coef):
        names = [f"feature_{idx}" for idx in range(len(coef))]
    frame = pd.DataFrame({"feature": names, "coefficient": coef}).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)
    frame.to_csv(output_dir / f"linear_coefficients_{model_name}.csv", index=False)


def run_quantile_experiment(output_dir: Path, n_buses: int = 20) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, train, validation, _ = _train_validation_frames(n_buses=n_buses)
    categorical = ["bus_unique_id", "zone_name", "kmeans_cluster"]
    numeric = [col for col in BUS_FEATURES_CLUSTERED if col not in categorical]
    spec = ModelSpec(numeric_features=numeric, categorical_features=categorical)
    q_preds = validation[["bus_unique_id", "zone_name", "timestamp", "date", "he", "pd"]].rename(
        columns={"bus_unique_id": "bus_id", "pd": "actual_pd"}
    )
    log_rows = []
    for alpha, col in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]:
        start = time.perf_counter()
        try:
            model = make_lightgbm_quantile_regressor(spec, alpha)
            clean_train = train.dropna(subset=["pd"]).copy()
            model.fit(clean_train[BUS_FEATURES_CLUSTERED], clean_train["pd"])
            q_preds[col] = model.predict(validation[BUS_FEATURES_CLUSTERED]).clip(min=0)
            log_rows.append({"quantile": col, "alpha": alpha, "status": "completed", "runtime_seconds": round(time.perf_counter() - start, 3)})
        except Exception as exc:
            q_preds[col] = np.nan
            log_rows.append({"quantile": col, "alpha": alpha, "status": f"skipped: {type(exc).__name__}: {exc}", "runtime_seconds": np.nan})
    q_preds["p10"], q_preds["p50"], q_preds["p90"] = np.minimum(q_preds["p10"], q_preds["p50"]), q_preds["p50"], np.maximum(q_preds["p90"], q_preds["p50"])
    q_preds["interval_width"] = q_preds["p90"] - q_preds["p10"]
    q_preds["covered_p10_p90"] = q_preds["actual_pd"].between(q_preds["p10"], q_preds["p90"])
    q_preds["peak_load_flag"] = q_preds["actual_pd"] >= q_preds["actual_pd"].quantile(0.9)
    summary = pd.DataFrame(
        [
            {
                "rows": len(q_preds),
                "p50_mae": (q_preds["p50"] - q_preds["actual_pd"]).abs().mean(),
                "p50_rmse": np.sqrt(((q_preds["p50"] - q_preds["actual_pd"]) ** 2).mean()),
                "p10_p90_coverage": q_preds["covered_p10_p90"].mean(),
                "mean_interval_width": q_preds["interval_width"].mean(),
                "peak_interval_width": q_preds.loc[q_preds["peak_load_flag"], "interval_width"].mean(),
                "non_peak_interval_width": q_preds.loc[~q_preds["peak_load_flag"], "interval_width"].mean(),
            }
        ]
    )
    q_preds.to_csv(output_dir / "research_quantile_forecasts.csv", index=False)
    summary.to_csv(output_dir / "research_quantile_summary.csv", index=False)
    pd.DataFrame(log_rows).to_csv(output_dir / "research_quantile_model_log.csv", index=False)
    _plot_prediction_interval(q_preds, output_dir)
    _plot_uncertainty_by_he(q_preds, output_dir)
    return {"research_quantile_forecasts": q_preds, "research_quantile_summary": summary}


def _make_sequence_examples(bus_features: pd.DataFrame, seq_len: int = 168):
    rows_x, rows_y, meta = [], [], []
    feature_col = "pd"
    for bus_id, grp in bus_features.sort_values("timestamp").groupby("bus_unique_id"):
        values = grp[feature_col].astype(float).interpolate(limit_direction="both").fillna(0).to_numpy(dtype="float32")
        timestamps = grp["timestamp"].to_numpy()
        dates = grp["date"].to_numpy()
        he = grp["he"].to_numpy()
        zone = grp["zone_name"].to_numpy()
        for idx in range(seq_len, len(grp)):
            target_ts = pd.Timestamp(timestamps[idx])
            # Exploratory sequence validation is restricted to Apr 1 so the input window ends before each day-ahead target.
            if target_ts.date() == pd.Timestamp(VALIDATE_START).date() and pd.Timestamp(timestamps[idx - 1]) > pd.Timestamp(FORECAST_CREATED_AT):
                continue
            rows_x.append(values[idx - seq_len : idx])
            rows_y.append(values[idx])
            meta.append(
                {
                    "bus_id": bus_id,
                    "zone_name": zone[idx],
                    "timestamp": target_ts,
                    "date": pd.Timestamp(dates[idx]),
                    "he": he[idx],
                }
            )
    return np.asarray(rows_x, dtype="float32"), np.asarray(rows_y, dtype="float32"), pd.DataFrame(meta)


def run_sequence_model_experiments(output_dir: Path, n_buses: int = 12, seq_len: int = 168, epochs: int = 8) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        from torch import nn
    except Exception as exc:
        log = pd.DataFrame([{"model": "sequence_models", "status": f"skipped: {type(exc).__name__}: {exc}"}])
        log.to_csv(output_dir / "research_sequence_model_log.csv", index=False)
        return {"research_sequence_model_log": log}

    torch.manual_seed(RANDOM_STATE)
    bus, _ = build_prototype_dataset(n_buses=n_buses)
    features = build_bus_feature_frame(bus)
    x, y, meta = _make_sequence_examples(features, seq_len=seq_len)
    train_mask = meta["date"] <= pd.Timestamp(TRAIN_END)
    val_mask = meta["date"] == pd.Timestamp(VALIDATE_START)
    x_train, y_train = x[train_mask.to_numpy()], y[train_mask.to_numpy()]
    x_val, y_val, meta_val = x[val_mask.to_numpy()], y[val_mask.to_numpy()], meta[val_mask].copy()
    if len(x_train) == 0 or len(x_val) == 0:
        log = pd.DataFrame([{"model": "sequence_models", "status": "skipped: insufficient leakage-safe sequence examples"}])
        log.to_csv(output_dir / "research_sequence_model_log.csv", index=False)
        return {"research_sequence_model_log": log}

    mean, std = float(y_train.mean()), float(y_train.std() or 1.0)
    x_train_t = torch.tensor((x_train - mean) / std).unsqueeze(-1)
    y_train_t = torch.tensor((y_train - mean) / std).unsqueeze(-1)
    x_val_t = torch.tensor((x_val - mean) / std).unsqueeze(-1)
    y_val_t = torch.tensor(y_val).unsqueeze(-1)

    models = {
        "research_tcn_lightweight": _TinyTCN(nn),
        "research_gru_lightweight": _TinyRNN(nn, kind="gru"),
        "research_lstm_lightweight": _TinyRNN(nn, kind="lstm"),
    }
    predictions = []
    log_rows = []
    for name, model in models.items():
        start = time.perf_counter()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model(x_train_t)
            loss = loss_fn(pred, y_train_t)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(x_val_t).squeeze().numpy() * std + mean
        pred = np.clip(pred, 0, None)
        frame = meta_val.copy()
        frame["actual_pd"] = y_val_t.squeeze().numpy()
        frame["predicted_pd"] = pred
        frame["model"] = name
        frame["horizon"] = "next_day"
        predictions.append(frame)
        metrics = regression_metrics(frame)
        log_rows.append({"model": name, "status": "completed", "runtime_seconds": round(time.perf_counter() - start, 3), **metrics})
    all_pred = pd.concat(predictions, ignore_index=True)
    log = pd.DataFrame(log_rows)
    all_pred.to_csv(output_dir / "research_sequence_predictions.csv", index=False)
    log.to_csv(output_dir / "research_sequence_model_log.csv", index=False)
    _plot_sequence_actual_vs_pred(all_pred, output_dir)
    return {"research_sequence_predictions": all_pred, "research_sequence_model_log": log}


def _TinyTCN(nn):
    class TinyTCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=5, padding=4, dilation=2),
                nn.ReLU(),
                nn.Conv1d(16, 8, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(8, 1)

        def forward(self, x):
            x = x.transpose(1, 2)
            return self.head(self.net(x).squeeze(-1))

    return TinyTCN()


def _TinyRNN(nn, kind: str):
    import torch

    class TinyRNN(nn.Module):
        def __init__(self):
            super().__init__()
            rnn_cls = nn.GRU if kind == "gru" else nn.LSTM
            self.rnn = rnn_cls(input_size=1, hidden_size=16, batch_first=True)
            self.head = nn.Linear(16, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :])

    return TinyRNN()


def build_comprehensive_comparison(output_dir: Path) -> pd.DataFrame:
    frames = []
    for path in [
        output_dir / "advanced_evaluation_summary.csv",
        output_dir / "research_classical_evaluation_summary.csv",
    ]:
        if path.exists():
            frames.append(pd.read_csv(path))
    seq_path = output_dir / "research_sequence_model_log.csv"
    if seq_path.exists():
        seq = pd.read_csv(seq_path)
        if {"mae", "rmse", "wmape"}.issubset(seq.columns):
            seq_eval = seq.rename(columns={"rows": "rows"})[["model", "rows", "mae", "rmse", "wmape"]].copy()
            seq_eval["horizon"] = "next_day"
            seq_eval["level"] = "bus"
            frames.append(seq_eval)
    comp = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not comp.empty:
        comp.to_csv(output_dir / "research_comprehensive_model_comparison.csv", index=False)
    return comp


def run_research_experiments(output_dir: Path, n_buses: int = 20) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    results.update(run_classical_model_experiments(output_dir, n_buses=n_buses))
    results.update(run_quantile_experiment(output_dir, n_buses=n_buses))
    results.update(run_sequence_model_experiments(output_dir, n_buses=min(4, n_buses), seq_len=72, epochs=3))
    results["research_comprehensive_model_comparison"] = build_comprehensive_comparison(output_dir)
    _plot_error_by_he(output_dir)
    _plot_cluster_performance(output_dir)
    _write_research_notes(output_dir)
    return results


def _plot_prediction_interval(q_preds: pd.DataFrame, output_dir: Path) -> None:
    one_bus = q_preds["bus_id"].iloc[0]
    day = q_preds[(q_preds["bus_id"] == one_bus) & (q_preds["date"] == pd.Timestamp(VALIDATE_START))].sort_values("timestamp")
    if day.empty:
        return
    plt.figure(figsize=(11, 5))
    plt.plot(day["timestamp"], day["actual_pd"], label="actual", color="black")
    plt.plot(day["timestamp"], day["p50"], label="P50", color="tab:blue")
    plt.fill_between(day["timestamp"], day["p10"], day["p90"], alpha=0.25, label="P10-P90")
    plt.title(f"Prediction interval: {one_bus}")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("pd")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "research_prediction_interval_next_day.png", dpi=150)
    plt.close()


def _plot_uncertainty_by_he(q_preds: pd.DataFrame, output_dir: Path) -> None:
    by_he = q_preds.groupby("he", as_index=False)["interval_width"].mean()
    plt.figure(figsize=(8, 4))
    plt.plot(by_he["he"], by_he["interval_width"], marker="o")
    plt.title("Mean prediction interval width by HE")
    plt.xlabel("HE")
    plt.ylabel("P90 - P10")
    plt.tight_layout()
    plt.savefig(output_dir / "research_uncertainty_by_he.png", dpi=150)
    plt.close()


def _plot_model_runtime(log: pd.DataFrame, output_dir: Path) -> None:
    done = log.dropna(subset=["runtime_seconds"]).copy()
    if done.empty:
        return
    plt.figure(figsize=(8, 4))
    plt.barh(done["model"], done["runtime_seconds"])
    plt.xlabel("seconds")
    plt.title("Classical model runtime comparison")
    plt.tight_layout()
    plt.savefig(output_dir / "research_runtime_comparison.png", dpi=150)
    plt.close()


def _plot_sequence_actual_vs_pred(predictions: pd.DataFrame, output_dir: Path) -> None:
    one_bus = predictions["bus_id"].iloc[0]
    subset = predictions[predictions["bus_id"] == one_bus].sort_values(["model", "timestamp"])
    plt.figure(figsize=(10, 5))
    actual = subset.drop_duplicates("timestamp")
    plt.plot(actual["timestamp"], actual["actual_pd"], label="actual", color="black")
    for model, grp in subset.groupby("model"):
        plt.plot(grp["timestamp"], grp["predicted_pd"], label=model, alpha=0.8)
    plt.title(f"Exploratory sequence model predictions: {one_bus}")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("pd")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "research_sequence_actual_vs_predicted.png", dpi=150)
    plt.close()


def _plot_error_by_he(output_dir: Path) -> None:
    path = output_dir / "research_comprehensive_model_comparison.csv"
    pred_path = output_dir / "research_classical_model_predictions.csv"
    if not pred_path.exists():
        return
    pred = pd.read_csv(pred_path)
    pred["abs_error"] = (pred["predicted_pd"] - pred["actual_pd"]).abs()
    by_he = pred.groupby(["model", "he"], as_index=False)["abs_error"].mean()
    plt.figure(figsize=(10, 5))
    for model, grp in by_he.groupby("model"):
        plt.plot(grp["he"], grp["abs_error"], label=model)
    plt.title("Classical model MAE by HE")
    plt.xlabel("HE")
    plt.ylabel("MAE")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "research_error_by_he.png", dpi=150)
    plt.close()


def _plot_cluster_performance(output_dir: Path) -> None:
    pred_path = output_dir / "research_classical_model_predictions.csv"
    cluster_path = output_dir / "research_cluster_assignments.csv"
    if not pred_path.exists() or not cluster_path.exists():
        return
    pred = pd.read_csv(pred_path)
    clusters = pd.read_csv(cluster_path)[["bus_id", "kmeans_cluster"]]
    frame = pred.merge(clusters, on="bus_id", how="left")
    frame["abs_error"] = (frame["predicted_pd"] - frame["actual_pd"]).abs()
    by_cluster = frame.groupby(["model", "kmeans_cluster"], as_index=False)["abs_error"].mean()
    by_cluster.to_csv(output_dir / "research_cluster_wise_error.csv", index=False)
    pivot = by_cluster.pivot(index="kmeans_cluster", columns="model", values="abs_error")
    pivot.plot(kind="bar", figsize=(10, 5))
    plt.title("Cluster-wise MAE")
    plt.xlabel("KMeans cluster")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(output_dir / "research_cluster_wise_error.png", dpi=150)
    plt.close()


def _write_research_notes(output_dir: Path) -> None:
    notes = pd.DataFrame(
        [
            {
                "topic": "graph_spatial_future_work",
                "summary": "A future graph-aware extension could connect buses to zones and zones to neighboring zones, then add graph aggregation features such as neighboring-zone lagged load, zone residual smoothing, or bus-zone hierarchy embeddings. A full graph neural network is left as future work because topology data are not yet available in Assignment 2.",
            },
            {
                "topic": "sequence_model_scope",
                "summary": "TCN, GRU, and LSTM are implemented as lightweight exploratory day-ahead sequence models. They are not tuned and are intended to compare inductive bias and compute cost against engineered-feature boosting.",
            },
        ]
    )
    notes.to_csv(output_dir / "research_future_work_notes.csv", index=False)
