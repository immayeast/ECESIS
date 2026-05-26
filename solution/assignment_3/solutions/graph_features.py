"""Graph feature construction for bus-network-aware matching."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from normalize import jaccard, name_tokens, normalize_bus_name, voltage_band


def build_name_lookup(bus: pd.DataFrame, source: str) -> dict[str, dict[str, Any]]:
    """Map raw and normalized bus names to metadata."""
    lookup: dict[str, dict[str, Any]] = {}
    for _, row in bus.iterrows():
        meta = {
            "id": row[f"{source}_bus_id"],
            "name": row["raw_bus_name"],
            "normalized": row["normalized_bus_name"],
            "kv": row["voltage_kv"],
            "band": row["voltage_band"],
        }
        lookup[str(row["raw_bus_name"])] = meta
        lookup[row["normalized_bus_name"]] = meta
    return lookup


def build_graph_from_branches(branches: pd.DataFrame, bus: pd.DataFrame, source: str) -> dict[str, set[str]]:
    """Build an undirected adjacency map keyed by standardized bus IDs."""
    lookup = build_name_lookup(bus, source)
    adjacency: dict[str, set[str]] = defaultdict(set)
    id_set = set(bus[f"{source}_bus_id"])
    for bus_id in id_set:
        adjacency[bus_id]

    if source == "dayzer":
        from_col, to_col = "FROM_BUS", "TO_BUS"
    else:
        from_col, to_col = "FROM_BUS_NAME", "TO_BUS_NAME"

    if from_col not in branches.columns or to_col not in branches.columns:
        return adjacency

    for _, row in branches.iterrows():
        left = resolve_endpoint(row[from_col], lookup)
        right = resolve_endpoint(row[to_col], lookup)
        if left and right and left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    return adjacency


def resolve_endpoint(value: object, lookup: dict[str, dict[str, Any]]) -> str | None:
    """Resolve a branch endpoint to a bus ID using raw then normalized name."""
    if pd.isna(value):
        return None
    raw = str(value)
    if raw in lookup:
        return lookup[raw]["id"]
    norm = normalize_bus_name(raw)
    if norm in lookup:
        return lookup[norm]["id"]
    return None


def endpoint_validity(branches: pd.DataFrame, bus: pd.DataFrame, source: str) -> dict[str, int]:
    """Count branch endpoints that resolve to known buses."""
    lookup = build_name_lookup(bus, source)
    if source == "dayzer":
        endpoint_cols = ["FROM_BUS", "TO_BUS"]
    else:
        endpoint_cols = ["FROM_BUS_NAME", "TO_BUS_NAME"]
    total = 0
    valid = 0
    for col in endpoint_cols:
        if col not in branches.columns:
            continue
        total += len(branches[col])
        valid += branches[col].map(lambda x: resolve_endpoint(x, lookup) is not None).sum()
    return {"endpoint_count": int(total), "valid_endpoint_count": int(valid), "invalid_endpoint_count": int(total - valid)}


def graph_fingerprints(bus: pd.DataFrame, adjacency: dict[str, set[str]], source: str) -> pd.DataFrame:
    """Compute local topology and neighbor attribute fingerprints."""
    meta = bus.set_index(f"{source}_bus_id").to_dict("index")
    rows = []
    for bus_id, neighbors in adjacency.items():
        neighbor_bands = Counter()
        neighbor_tokens: set[str] = set()
        same_voltage = 0
        high_voltage = 0
        low_voltage = 0
        own_kv = meta.get(bus_id, {}).get("voltage_kv")
        for nb in neighbors:
            nb_meta = meta.get(nb, {})
            band = nb_meta.get("voltage_band", "unknown")
            neighbor_bands[band] += 1
            neighbor_tokens |= name_tokens(nb_meta.get("raw_bus_name", ""))
            nb_kv = nb_meta.get("voltage_kv")
            if pd.notna(own_kv) and pd.notna(nb_kv):
                if abs(float(own_kv) - float(nb_kv)) < 1e-6:
                    same_voltage += 1
                elif float(nb_kv) > float(own_kv):
                    high_voltage += 1
                else:
                    low_voltage += 1
        rows.append(
            {
                f"{source}_bus_id": bus_id,
                "degree": len(neighbors),
                "neighbor_count": len(neighbors),
                "same_voltage_neighbor_count": same_voltage,
                "high_voltage_neighbor_count": high_voltage,
                "low_voltage_neighbor_count": low_voltage,
                "neighbor_voltage_profile": dict(neighbor_bands),
                "neighbor_name_tokens": " ".join(sorted(neighbor_tokens)),
            }
        )
    return pd.DataFrame(rows)


def profile_similarity(left: dict[str, int] | str | float, right: dict[str, int] | str | float) -> float:
    """Compare sparse neighbor voltage profiles."""
    if not isinstance(left, dict):
        left = {}
    if not isinstance(right, dict):
        right = {}
    keys = set(left) | set(right)
    if not keys:
        return 0.5
    overlap = sum(min(left.get(k, 0), right.get(k, 0)) for k in keys)
    total = sum(max(left.get(k, 0), right.get(k, 0)) for k in keys)
    return overlap / total if total else 0.5


def add_graph_pair_features(
    candidates: pd.DataFrame,
    dayzer_features: pd.DataFrame,
    pano_features: pd.DataFrame,
) -> pd.DataFrame:
    """Join bus graph features and calculate pair-level topology scores."""
    out = candidates.merge(dayzer_features, on="dayzer_bus_id", how="left")
    out = out.merge(pano_features, on="pano_bus_id", how="left", suffixes=("_dayzer", "_pano"))
    out["degree_dayzer"] = out["degree_dayzer"].fillna(0)
    out["degree_pano"] = out["degree_pano"].fillna(0)
    max_degree = out[["degree_dayzer", "degree_pano"]].max(axis=1).replace(0, 1)
    out["degree_similarity"] = 1 - (out["degree_dayzer"] - out["degree_pano"]).abs() / max_degree
    out["neighbor_voltage_overlap"] = [
        profile_similarity(l, r)
        for l, r in zip(out["neighbor_voltage_profile_dayzer"], out["neighbor_voltage_profile_pano"])
    ]
    out["neighbor_token_overlap"] = [
        jaccard(set(str(l).split()), set(str(r).split()))
        for l, r in zip(out["neighbor_name_tokens_dayzer"].fillna(""), out["neighbor_name_tokens_pano"].fillna(""))
    ]
    out["topology_score"] = (
        0.45 * out["degree_similarity"].fillna(0)
        + 0.35 * out["neighbor_voltage_overlap"].fillna(0)
        + 0.20 * out["neighbor_token_overlap"].fillna(0)
    )
    return out
