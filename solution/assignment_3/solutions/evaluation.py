"""Inspection and reporting utilities for Assignment 3."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def inspect_table(df: pd.DataFrame, name: str) -> dict[str, object]:
    """Return compact data-quality diagnostics."""
    return {
        "table": name,
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_values": df.isna().sum().to_dict(),
    }


def bus_diagnostics(bus: pd.DataFrame, source: str) -> dict[str, object]:
    """Diagnostics for a standardized bus table."""
    return {
        "source": source,
        "row_count": int(len(bus)),
        "unique_bus_names": int(bus["raw_bus_name"].nunique()),
        "duplicate_bus_names": int(bus["raw_bus_name"].duplicated().sum()),
        "lat_lon_coverage": float((bus["latitude"].notna() & bus["longitude"].notna()).mean()),
        "duplicate_coordinates": int(bus[["latitude", "longitude"]].duplicated().sum()),
        "kv_distribution": bus["voltage_kv"].value_counts(dropna=False).sort_index().to_dict(),
    }


def coordinate_diagnostics(df: pd.DataFrame, source: str, lat_col: str = "LAT", lon_col: str = "LON") -> dict[str, object]:
    """Summarize latitude/longitude coverage before spatial matching."""
    lat = pd.to_numeric(df[lat_col], errors="coerce") if lat_col in df.columns else pd.Series(index=df.index, dtype=float)
    lon = pd.to_numeric(df[lon_col], errors="coerce") if lon_col in df.columns else pd.Series(index=df.index, dtype=float)
    both_present = lat.notna() & lon.notna()
    invalid_range = both_present & (~lat.between(-90, 90) | ~lon.between(-180, 180))
    zero_coordinate = both_present & ((lat == 0) | (lon == 0))
    duplicate_coordinates = df.loc[both_present, [lat_col, lon_col]].duplicated().sum() if lat_col in df.columns and lon_col in df.columns else 0
    return {
        "source": source,
        "row_count": int(len(df)),
        "missing_latitude": int(lat.isna().sum()),
        "missing_longitude": int(lon.isna().sum()),
        "missing_either_coordinate": int((~both_present).sum()),
        "coordinate_coverage_pct": round(float(both_present.mean() * 100), 2) if len(df) else 0.0,
        "invalid_nonmissing_coordinate_range": int(invalid_range.sum()),
        "zero_coordinate_rows": int(zero_coordinate.sum()),
        "duplicate_coordinate_rows": int(duplicate_coordinates),
        "latitude_min": float(lat.min()) if lat.notna().any() else None,
        "latitude_max": float(lat.max()) if lat.notna().any() else None,
        "longitude_min": float(lon.min()) if lon.notna().any() else None,
        "longitude_max": float(lon.max()) if lon.notna().any() else None,
    }


def make_manual_review_sample(results: pd.DataFrame, sample_n: int = 20, random_state: int = 42) -> pd.DataFrame:
    """Create a balanced review queue across confidence strata."""
    buckets = []
    specs = [
        ("high", results["confidence_label"].eq("high")),
        ("medium", results["confidence_label"].eq("medium")),
        ("ambiguous", results["ambiguity_flag"].eq(True)),
        ("low_unmatched", results["confidence_label"].eq("low") | results["pano_bus"].isna()),
    ]
    for bucket, mask in specs:
        subset = results.loc[mask].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("composite_score", ascending=False)
        if len(subset) > sample_n:
            subset = subset.sample(sample_n, random_state=random_state).sort_values("composite_score", ascending=False)
        subset.insert(0, "review_bucket", bucket)
        buckets.append(subset)
    return pd.concat(buckets, ignore_index=True) if buckets else pd.DataFrame()


def write_summary_report(
    path: Path,
    dayzer_bus_count: int,
    pano_bus_count: int,
    results: pd.DataFrame,
    endpoint_summary: dict[str, dict[str, int]],
) -> None:
    """Write the assignment summary report."""
    matched = int(results["pano_bus"].notna().sum())
    unmatched = int(results["pano_bus"].isna().sum())
    confidence_counts = results["confidence_label"].value_counts(dropna=False).to_dict()
    ambiguity_rate = float(results["ambiguity_flag"].mean()) if len(results) else 0.0
    high = confidence_counts.get("high", 0)
    medium = confidence_counts.get("medium", 0)
    low = confidence_counts.get("low", 0)
    ambiguous = confidence_counts.get("ambiguous", 0)

    text = f"""# Assignment 3 Bus Mapping Summary

## 1. Problem framing

This assignment treats bus mapping as graph-aware entity resolution. Each bus is an electrical node and each branch is an edge. The task is similar to aligning two maps of the same city: names may differ, but coordinates, voltage levels, and surrounding network structure provide evidence about which intersections correspond.

## 2. Method

The implementation normalizes bus names conservatively, generates geographic candidate pairs, computes attribute similarity features, adds graph fingerprint features from branch connectivity, and ranks candidates with an interpretable composite score. Ambiguity is handled explicitly with score gaps, conflict flags, and confidence labels.

## 3. Why each signal matters

- Names can differ across systems because of abbreviations, punctuation, or modeling conventions.
- Voltage constrains plausible matches because a 345 kV bus is usually not interchangeable with a 69 kV bus.
- Coordinates provide physical evidence for substations and nearby bus assets.
- Topology validates neighborhood consistency when two buses connect to similar electrical surroundings.

## 4. Results

- Dayzer buses: {dayzer_bus_count:,}
- Panorama buses: {pano_bus_count:,}
- Matched Dayzer buses: {matched:,}
- Unmatched Dayzer buses: {unmatched:,}
- High confidence: {high:,}
- Medium confidence: {medium:,}
- Low confidence: {low:,}
- Ambiguous: {ambiguous:,}
- Ambiguity rate: {ambiguity_rate:.1%}

Branch endpoint validity:

- Dayzer valid endpoints: {endpoint_summary.get('dayzer', {}).get('valid_endpoint_count', 0):,} / {endpoint_summary.get('dayzer', {}).get('endpoint_count', 0):,}
- Panorama valid endpoints: {endpoint_summary.get('pano', {}).get('valid_endpoint_count', 0):,} / {endpoint_summary.get('pano', {}).get('endpoint_count', 0):,}

## 5. Insights

Spatial proximity is useful but not sufficient, especially in dense substations where several buses share very similar coordinates. Voltage conflicts strongly reduce confidence and often identify false geographic neighbors. Topology helps when names are inconsistent by checking whether the proposed pair has a similar local electrical neighborhood. One-to-one conflicts are a practical way to surface difficult areas where one dataset may aggregate or split buses differently.

## 6. Limitations

The method is sensitive to coordinate errors and missing metadata. The two systems may use non-identical network modeling conventions, including bus aggregation, bus splitting, isolated buses, and different treatment of station-level assets. Topology features are local and interpretable, but they are not a full global graph alignment.

## 7. Future work

Future iterations could add manual labels and supervised refinement, graph embeddings, global graph alignment, and topology-aware probabilistic matching. The current outputs preserve feature columns so manual validation can be used to recalibrate the scoring model.
"""
    path.write_text(text)
