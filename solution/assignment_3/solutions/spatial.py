"""Spatial candidate generation for Dayzer-to-Panorama bus matching."""

from __future__ import annotations

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in kilometers."""
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def distance_score(distance_km: float, radius_km: float = 20.0) -> float:
    """Convert distance to a 0-1 score, where nearby buses score higher."""
    if pd.isna(distance_km):
        return 0.0
    return float(np.exp(-max(distance_km, 0.0) / max(radius_km, 1e-6)))


def generate_spatial_candidates(
    dayzer: pd.DataFrame,
    pano: pd.DataFrame,
    radius_km: float = 20.0,
    top_k: int = 10,
    min_candidates: int = 3,
) -> pd.DataFrame:
    """Generate candidate pairs using radius search plus nearest-neighbor fallback.

    This intentionally avoids a hard dependency on scikit-learn. The input sizes
    are modest enough for a per-Dayzer vectorized distance calculation.
    """
    pano_coord_mask = pano["latitude"].notna() & pano["longitude"].notna()
    pano_geo = pano.loc[pano_coord_mask].copy()
    pano_lat = pano_geo["latitude"].to_numpy(float)
    pano_lon = pano_geo["longitude"].to_numpy(float)

    records: list[dict] = []
    for dz_idx, dz in dayzer.iterrows():
        has_coords = pd.notna(dz["latitude"]) and pd.notna(dz["longitude"]) and len(pano_geo) > 0
        selected_idx: list[int] = []
        distances: dict[int, float] = {}

        if has_coords:
            dist = haversine_km(float(dz["latitude"]), float(dz["longitude"]), pano_lat, pano_lon)
            order = np.argsort(dist)
            radius_positions = np.where(dist <= radius_km)[0]
            selected_positions = list(radius_positions)
            if len(selected_positions) < min_candidates:
                selected_positions = sorted(set(selected_positions) | set(order[:top_k]))
            else:
                selected_positions = sorted(set(selected_positions) | set(order[: min(top_k, len(order))]))
            selected_idx = list(pano_geo.index[selected_positions])
            distances = {idx: float(dist[pos]) for pos, idx in zip(selected_positions, selected_idx)}

        if not selected_idx:
            selected_idx = fallback_name_voltage_candidates(dz, pano, top_k=top_k)
            distances = {idx: np.nan for idx in selected_idx}

        for rank, pano_idx in enumerate(sorted(selected_idx, key=lambda idx: distances.get(idx, np.inf)), start=1):
            pn = pano.loc[pano_idx]
            records.append(
                {
                    "dayzer_bus_id": dz["dayzer_bus_id"],
                    "dayzer_bus_name": dz["raw_bus_name"],
                    "pano_bus_id": pn["pano_bus_id"],
                    "pano_bus_name": pn["raw_bus_name"],
                    "distance_km": distances.get(pano_idx, np.nan),
                    "dayzer_kv": dz["voltage_kv"],
                    "pano_kv": pn["voltage_kv"],
                    "distance_rank": rank,
                }
            )
    return pd.DataFrame.from_records(records).drop_duplicates(["dayzer_bus_id", "pano_bus_id"])


def fallback_name_voltage_candidates(dz: pd.Series, pano: pd.DataFrame, top_k: int = 10) -> list[int]:
    """Candidate fallback when coordinates are missing."""
    dz_tokens = set(str(dz.get("normalized_bus_name", "")).split())
    dz_band = dz.get("voltage_band", "unknown")

    scored = []
    for idx, pn in pano.iterrows():
        pn_tokens = set(str(pn.get("normalized_bus_name", "")).split())
        union = dz_tokens | pn_tokens
        token_score = len(dz_tokens & pn_tokens) / len(union) if union else 0.0
        band_bonus = 0.25 if dz_band != "unknown" and dz_band == pn.get("voltage_band") else 0.0
        scored.append((token_score + band_bonus, idx))
    scored.sort(reverse=True)
    return [idx for _, idx in scored[:top_k]]
