from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


@dataclass(frozen=True)
class ClusterResult:
    bus_clusters: pd.DataFrame
    cluster_stats: pd.DataFrame
    kmeans_centers: pd.DataFrame
    profile_matrix: pd.DataFrame
    cluster_diagnostics: pd.DataFrame
    gmm_available: bool


def build_hourly_load_profiles(train_bus: pd.DataFrame) -> pd.DataFrame:
    profile = (
        train_bus.groupby(["bus_unique_id", "he"], as_index=False)["pd"]
        .mean()
        .pivot(index="bus_unique_id", columns="he", values="pd")
        .sort_index()
    )
    profile = profile.reindex(columns=range(1, 25))
    profile = profile.interpolate(axis=1, limit_direction="both").fillna(0)
    profile.columns = [f"he_{int(col):02d}" for col in profile.columns]
    return profile


def _zscore_rows(profile: pd.DataFrame) -> pd.DataFrame:
    values = profile.to_numpy(dtype=float)
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    normalized = (values - mean) / std
    return pd.DataFrame(normalized, index=profile.index, columns=profile.columns)


def cluster_bus_load_shapes(
    train_bus: pd.DataFrame,
    n_clusters: int = 4,
    use_gmm: bool = True,
) -> ClusterResult:
    profile = build_hourly_load_profiles(train_bus)
    normalized = _zscore_rows(profile)
    n_clusters = int(min(max(2, n_clusters), len(normalized)))

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=RANDOM_STATE)
    kmeans_labels = kmeans.fit_predict(normalized)
    distances = kmeans.transform(normalized)
    nearest = distances.min(axis=1)
    denom = nearest.max() if nearest.max() > 0 else 1.0
    kmeans_confidence = 1 - nearest / denom

    bus_meta = (
        train_bus.groupby("bus_unique_id", as_index=False)
        .agg(zone=("zone_name", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]), average_pd=("pd", "mean"))
    )
    clusters = pd.DataFrame(
        {
            "bus_id": normalized.index,
            "kmeans_cluster": kmeans_labels,
            "kmeans_confidence": kmeans_confidence,
        }
    ).merge(bus_meta, left_on="bus_id", right_on="bus_unique_id", how="left").drop(columns=["bus_unique_id"])

    gmm_available = False
    gmm_aic = np.nan
    gmm_bic = np.nan
    if use_gmm and len(normalized) >= n_clusters * 2:
        gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=RANDOM_STATE)
        gmm_labels = gmm.fit_predict(normalized)
        gmm_proba = gmm.predict_proba(normalized)
        clusters["gmm_cluster"] = gmm_labels
        clusters["cluster_confidence"] = gmm_proba.max(axis=1)
        gmm_aic = gmm.aic(normalized)
        gmm_bic = gmm.bic(normalized)
        gmm_available = True
    else:
        clusters["gmm_cluster"] = pd.NA
        clusters["cluster_confidence"] = clusters["kmeans_confidence"]

    cluster_stats = (
        clusters.groupby("kmeans_cluster", as_index=False)
        .agg(bus_count=("bus_id", "count"), average_pd=("average_pd", "mean"), confidence_mean=("cluster_confidence", "mean"))
        .sort_values("kmeans_cluster")
    )
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=normalized.columns)
    centers.insert(0, "kmeans_cluster", range(n_clusters))
    diagnostics = pd.DataFrame(
        [
            {
                "n_buses": len(normalized),
                "n_clusters": n_clusters,
                "kmeans_inertia": kmeans.inertia_,
                "kmeans_silhouette": silhouette_score(normalized, kmeans_labels) if n_clusters < len(normalized) else np.nan,
                "gmm_available": gmm_available,
                "gmm_aic": gmm_aic,
                "gmm_bic": gmm_bic,
            }
        ]
    )
    return ClusterResult(clusters, cluster_stats, centers, normalized, diagnostics, gmm_available)


def attach_cluster_features(df: pd.DataFrame, bus_clusters: pd.DataFrame) -> pd.DataFrame:
    cluster_cols = ["bus_id", "kmeans_cluster", "gmm_cluster", "cluster_confidence", "average_pd"]
    available = [col for col in cluster_cols if col in bus_clusters.columns]
    out = df.merge(bus_clusters[available], left_on="bus_unique_id", right_on="bus_id", how="left")
    out = out.drop(columns=["bus_id"], errors="ignore")
    out = out.rename(columns={"average_pd": "cluster_avg_load"})
    out["kmeans_cluster"] = out["kmeans_cluster"].astype("Int64").astype(str).replace("<NA>", "unknown")
    if "gmm_cluster" in out.columns:
        out["gmm_cluster"] = out["gmm_cluster"].astype("Int64").astype(str).replace("<NA>", "unknown")
    out["cluster_confidence"] = pd.to_numeric(out["cluster_confidence"], errors="coerce").fillna(0)
    out["cluster_avg_load"] = pd.to_numeric(out["cluster_avg_load"], errors="coerce")
    return out
