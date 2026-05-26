"""Evaluation diagnostics for Assignment 3 bus mapping.

These diagnostics do not estimate true accuracy because no labeled ground truth
is available. They measure operational consistency: voltage feasibility,
duplicate assignment pressure, topology support, distance plausibility, and
manual-review coverage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42


def voltage_conflict_rate(df: pd.DataFrame, matched_mask: pd.Series | None = None) -> float:
    if matched_mask is None:
        matched_mask = pd.Series(True, index=df.index)
    subset = df.loc[matched_mask & df["dayzer_kv"].notna() & df["pano_kv"].notna()]
    if subset.empty:
        return np.nan
    return float(((subset["dayzer_kv"] - subset["pano_kv"]).abs() > 1).mean())


def exact_voltage_rate(df: pd.DataFrame, matched_mask: pd.Series | None = None) -> float:
    if matched_mask is None:
        matched_mask = pd.Series(True, index=df.index)
    subset = df.loc[matched_mask & df["dayzer_kv"].notna() & df["pano_kv"].notna()]
    if subset.empty:
        return np.nan
    return float(((subset["dayzer_kv"] - subset["pano_kv"]).abs() < 1e-9).mean())


def summarize_variant(
    df: pd.DataFrame,
    variant: str,
    pano_col: str = "pano_bus_id",
    ambiguity_col: str | None = None,
    total_expected: int | None = None,
) -> dict[str, object]:
    """Summarize a mapping-like table."""
    matched = df[pano_col].notna() if pano_col in df.columns else pd.Series(False, index=df.index)
    distances = df.loc[matched, "distance_km"] if "distance_km" in df.columns else pd.Series(dtype=float)
    ambiguity = df[ambiguity_col].mean() if ambiguity_col and ambiguity_col in df.columns and len(df) else np.nan
    total = total_expected if total_expected is not None else len(df)
    ambiguity_count = int(df[ambiguity_col].sum()) if ambiguity_col and ambiguity_col in df.columns and len(df) else np.nan
    return {
        "variant": variant,
        "matched_buses": int(matched.sum()),
        "unmatched_count": int(total - matched.sum()),
        "duplicate_pano_assignments": int(df.loc[matched, pano_col].duplicated().sum()) if pano_col in df.columns else np.nan,
        "exact_voltage_match_rate": exact_voltage_rate(df, matched),
        "voltage_conflict_rate_gt_1kv": voltage_conflict_rate(df, matched),
        "median_distance_km": float(distances.median()) if distances.notna().any() else np.nan,
        "p90_distance_km": float(distances.quantile(0.90)) if distances.notna().any() else np.nan,
        "ambiguity_count": ambiguity_count,
        "ambiguity_rate": float(ambiguity) if pd.notna(ambiguity) else np.nan,
    }


def top_by_score(scored: pd.DataFrame, score_col: str, variant: str) -> pd.DataFrame:
    ranked = scored.sort_values(["dayzer_bus_id", score_col, "voltage_score", "name_score"], ascending=[True, False, False, False]).copy()
    ranked["rank"] = ranked.groupby("dayzer_bus_id").cumcount() + 1
    ranked["next_score"] = ranked.groupby("dayzer_bus_id")[score_col].shift(-1)
    ranked["score_gap"] = ranked[score_col] - ranked["next_score"].fillna(0)
    top = ranked.loc[ranked["rank"].eq(1)].copy()
    duplicate_flag = top["pano_bus_id"].duplicated(keep=False)
    top["diagnostic_ambiguity_flag"] = duplicate_flag | (top["score_gap"] < 0.05) | ((top["dayzer_kv"] - top["pano_kv"]).abs() > 1)
    top["variant"] = variant
    return top


def build_ablation(scored: pd.DataFrame, final: pd.DataFrame, baseline: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Create requested ablation variants."""
    variants: dict[str, pd.DataFrame] = {}

    spatial = scored.sort_values(["dayzer_bus_id", "distance_km", "distance_rank"], ascending=[True, True, True]).groupby("dayzer_bus_id", as_index=False).first()
    spatial["diagnostic_ambiguity_flag"] = spatial["pano_bus_id"].duplicated(keep=False) | spatial["distance_km"].isna()
    variants["A_pure_latlon_nearest"] = spatial

    scored = scored.copy()
    scored["name_kv_score"] = 0.65 * scored["name_score"].fillna(0) + 0.35 * scored["voltage_score"].fillna(0)
    variants["B_name_kv_only"] = top_by_score(scored, "name_kv_score", "B_name_kv_only")

    scored["name_kv_latlon_score"] = (
        0.30 * scored["name_score"].fillna(0)
        + 0.30 * scored["voltage_score"].fillna(0)
        + 0.40 * scored["distance_score"].fillna(0)
    )
    variants["C_name_kv_latlon"] = top_by_score(scored, "name_kv_latlon_score", "C_name_kv_latlon")

    baseline_for_summary = baseline.rename(columns={"pano_bus": "pano_bus_name"}).copy()
    variants["D_graph_aware_no_propagation"] = baseline_for_summary
    final_for_summary = final.rename(columns={"pano_bus": "pano_bus_name"}).copy()
    variants["E_full_graph_aware_final"] = final_for_summary

    rows = []
    for name, df in variants.items():
        if name.startswith(("A_", "B_", "C_")):
            rows.append(summarize_variant(df, name, ambiguity_col="diagnostic_ambiguity_flag", total_expected=len(final)))
        else:
            rows.append(summarize_variant(df, name, ambiguity_col="ambiguity_flag", total_expected=len(final)))
    return pd.DataFrame(rows), variants


def build_sensitivity(scored: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity diagnostics over radius, topology weight, and confidence threshold."""
    rows = []

    for radius in [5, 10, 20, 50]:
        eligible = scored[(scored["distance_km"].le(radius)) | (scored["distance_km"].isna())].copy()
        if eligible.empty:
            continue
        top = top_by_score(eligible, "composite_score", f"radius_km={radius}")
        high_count = int((top["composite_score"] >= 0.85).sum())
        rows.append(
            {
                "parameter": "radius_km",
                "value": radius,
                "high_confidence_count": high_count,
                **summarize_variant(top, f"radius_km={radius}", ambiguity_col="diagnostic_ambiguity_flag", total_expected=final["dayzer_bus_id"].nunique()),
            }
        )

    for topology_weight in [0.10, 0.20, 0.30]:
        name_weight = max(0.0, 1.0 - 0.30 - 0.25 - topology_weight)
        temp = scored.copy()
        temp["sensitivity_score"] = (
            0.30 * temp["distance_score"].fillna(0)
            + 0.25 * temp["voltage_score"].fillna(0)
            + name_weight * temp["name_score"].fillna(0)
            + topology_weight * temp["topology_score"].fillna(0)
        )
        top = top_by_score(temp, "sensitivity_score", f"topology_weight={topology_weight}")
        high_count = int((top["sensitivity_score"] >= 0.85).sum())
        rows.append(
            {
                "parameter": "topology_weight",
                "value": topology_weight,
                "name_weight": name_weight,
                "high_confidence_count": high_count,
                **summarize_variant(top, f"topology_weight={topology_weight}", ambiguity_col="diagnostic_ambiguity_flag", total_expected=len(final)),
            }
        )

    for threshold in [0.65, 0.75, 0.85]:
        temp = final.copy()
        temp["threshold_match"] = temp["pano_bus_id"].where((temp["propagation_score"] >= threshold) & (~temp["ambiguity_flag"]))
        threshold_df = temp.rename(columns={"threshold_match": "threshold_pano_bus_id"})
        high_count = int(((temp["propagation_score"] >= threshold) & (~temp["ambiguity_flag"])).sum())
        rows.append(
            {
                "parameter": "confidence_threshold",
                "value": threshold,
                "high_confidence_count": high_count,
                **summarize_variant(
                    threshold_df,
                    f"confidence_threshold={threshold}",
                    pano_col="threshold_pano_bus_id",
                    ambiguity_col="ambiguity_flag",
                    total_expected=len(final),
                ),
            }
        )

    out = pd.DataFrame(rows)
    if "high_confidence_count" not in out:
        out["high_confidence_count"] = np.nan
    return out


def build_one_to_one_diagnostics(scored: pd.DataFrame, baseline: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    pre_top = scored.loc[scored["candidate_rank"].eq(1)].copy()
    pre_duplicate = int(pre_top["pano_bus_id"].dropna().duplicated().sum())
    baseline_duplicate = int(baseline["pano_bus_id"].dropna().duplicated().sum())
    conflict_rows = int(final.get("one_to_one_conflict_flag", pd.Series(False, index=final.index)).sum())
    final_duplicate = int(final["pano_bus_id"].dropna().duplicated().sum())
    return pd.DataFrame(
        [
            {"metric": "top_candidate_duplicate_pano_assignments_before_resolution", "value": pre_duplicate},
            {"metric": "baseline_duplicate_pano_assignments_after_baseline_resolution", "value": baseline_duplicate},
            {"metric": "lower_scoring_duplicates_marked_conflict_or_unmatched", "value": conflict_rows},
            {"metric": "final_duplicate_nonblank_pano_assignments", "value": final_duplicate},
        ]
    )


def build_topology_summary(scored: pd.DataFrame, final: pd.DataFrame, spatial: pd.DataFrame) -> pd.DataFrame:
    matched = final[final["pano_bus_id"].notna()].copy()
    accepted = final[final.get("accepted_graph_match", False) == True].copy()
    topology_low_name = matched[(matched["topology_score"] >= 0.80) & (matched["name_score"] < 0.55)]
    spatial_lookup = spatial.set_index("dayzer_bus_id")["pano_bus_id"].to_dict()
    matched["latlon_baseline_pano_bus_id"] = matched["dayzer_bus_id"].map(spatial_lookup)
    changed = matched[matched["pano_bus_id"].ne(matched["latlon_baseline_pano_bus_id"])]
    tie_helped = matched[(matched["score_gap_to_second_best"].fillna(1) < 0.05) & (matched["topology_score"] >= 0.80)]
    return pd.DataFrame(
        [
            {"metric": "accepted_topology_supported_matches", "value": int(len(accepted))},
            {"metric": "matched_high_topology_low_or_moderate_name", "value": int(len(topology_low_name))},
            {"metric": "matched_candidate_changed_from_latlon_nearest", "value": int(len(changed))},
            {"metric": "matched_small_gap_high_topology_cases", "value": int(len(tie_helped))},
        ]
    )


def build_distance_voltage_diagnostics(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    matched = final[final["pano_bus_id"].notna()].copy()
    high = matched[matched["confidence_label"].eq("high")]
    rows.extend(
        [
            {"section": "distance", "metric": "matched_mean_distance_km", "value": float(matched["distance_km"].mean())},
            {"section": "distance", "metric": "matched_median_distance_km", "value": float(matched["distance_km"].median())},
            {"section": "distance", "metric": "matched_p90_distance_km", "value": float(matched["distance_km"].quantile(0.90))},
            {"section": "distance", "metric": "matched_max_distance_km", "value": float(matched["distance_km"].max())},
            {"section": "distance", "metric": "high_confidence_median_distance_km", "value": float(high["distance_km"].median())},
        ]
    )
    for label, group in final.groupby("confidence_label"):
        matched_group = group[group["pano_bus_id"].notna()]
        rows.append({"section": "voltage", "metric": f"{label}_exact_voltage_match_rate", "value": exact_voltage_rate(matched_group)})
        rows.append({"section": "voltage", "metric": f"{label}_voltage_conflict_rate_gt_1kv", "value": voltage_conflict_rate(matched_group)})
    return pd.DataFrame(rows)


def sample_bucket(df: pd.DataFrame, label: str, mask: pd.Series, n: int = 50) -> pd.DataFrame:
    subset = df.loc[mask].copy()
    if subset.empty:
        return subset
    subset = subset.sort_values("propagation_score" if "propagation_score" in subset else "composite_score", ascending=False)
    if len(subset) > n:
        subset = subset.sample(n, random_state=RANDOM_STATE).sort_values("propagation_score" if "propagation_score" in subset else "composite_score", ascending=False)
    subset.insert(0, "review_bucket", label)
    return subset


def build_manual_review_extended(final: pd.DataFrame, spatial: pd.DataFrame) -> pd.DataFrame:
    spatial_names = spatial.set_index("dayzer_bus_id")[["pano_bus_name", "pano_bus_id"]].rename(
        columns={"pano_bus_name": "latlon_baseline_pano_bus", "pano_bus_id": "latlon_baseline_pano_bus_id"}
    )
    df = final.merge(spatial_names, on="dayzer_bus_id", how="left")
    df["graph_aware_pano_bus"] = df["pano_bus"]
    disagreed = df["pano_bus_id"].fillna("__missing__").ne(df["latlon_baseline_pano_bus_id"].fillna("__missing__"))

    buckets = [
        sample_bucket(df, "high_confidence", df["confidence_label"].eq("high")),
        sample_bucket(df, "medium_confidence", df["confidence_label"].eq("medium")),
        sample_bucket(df, "ambiguous", df["ambiguity_flag"].eq(True)),
        sample_bucket(df, "low_or_unmatched", df["confidence_label"].eq("low") | df["pano_bus"].isna()),
        sample_bucket(df, "topology_supported", df.get("accepted_graph_match", False) == True),
        sample_bucket(df, "latlon_disagreed_with_graph", disagreed),
    ]
    review = pd.concat([b for b in buckets if not b.empty], ignore_index=True)
    keep = [
        "review_bucket",
        "dayzer_bus",
        "pano_bus",
        "dayzer_kv",
        "pano_kv",
        "distance_km",
        "name_score",
        "voltage_score",
        "topology_score",
        "composite_score",
        "propagation_score",
        "match_stage",
        "confidence_label",
        "ambiguity_flag",
        "latlon_baseline_pano_bus",
        "graph_aware_pano_bus",
    ]
    review = review[keep].copy()
    review["review_label"] = ""
    review["review_notes"] = ""
    return review


def build_case_studies(scored: pd.DataFrame, final: pd.DataFrame, spatial: pd.DataFrame) -> pd.DataFrame:
    spatial_lookup = spatial.set_index("dayzer_bus_id")[["pano_bus_id", "pano_bus_name"]].to_dict("index")
    cases = []

    def add_case(case_type: str, row: pd.Series, decision: str, reason: str) -> None:
        latlon = spatial_lookup.get(row["dayzer_bus_id"], {})
        cases.append(
            {
                "case_type": case_type,
                "dayzer_bus": row.get("dayzer_bus", row.get("dayzer_bus_name")),
                "pano_candidate": row.get("pano_bus", row.get("pano_bus_name")),
                "latlon_baseline_pano_bus": latlon.get("pano_bus_name"),
                "dayzer_kv": row.get("dayzer_kv"),
                "pano_kv": row.get("pano_kv"),
                "distance_km": row.get("distance_km"),
                "name_score": row.get("name_score"),
                "topology_score": row.get("topology_score"),
                "propagation_support_count": row.get("propagation_support_count"),
                "decision": decision,
                "reason": reason,
            }
        )

    high = final[final["confidence_label"].eq("high")].sort_values("propagation_score", ascending=False).head(1)
    if not high.empty:
        add_case("high_confidence_clean_match", high.iloc[0], "accepted", "Strong voltage, score, and graph-aware evidence.")

    disagreed = final[
        final["pano_bus_id"].notna()
        & final["dayzer_bus_id"].map(lambda x: spatial_lookup.get(x, {}).get("pano_bus_id")).notna()
        & final["pano_bus_id"].ne(final["dayzer_bus_id"].map(lambda x: spatial_lookup.get(x, {}).get("pano_bus_id")))
        & (final["voltage_score"] >= 0.95)
    ].sort_values("propagation_score", ascending=False).head(1)
    if not disagreed.empty:
        add_case("latlon_baseline_failure_fixed", disagreed.iloc[0], "accepted", "Graph-aware final candidate differs from nearest-coordinate baseline and preserves electrical/topology consistency.")

    topo = final[(final["topology_score"] >= 0.80) & (final["name_score"] < 0.55) & final["pano_bus_id"].notna()].sort_values("propagation_score", ascending=False).head(1)
    if not topo.empty:
        add_case("topology_rescued_weak_name", topo.iloc[0], "accepted", "Name evidence is moderate/weak, but voltage and local topology support the match.")

    amb = final[final["ambiguity_flag"].eq(True)].sort_values("propagation_score", ascending=False).head(1)
    if not amb.empty:
        add_case("ambiguous_unresolved", amb.iloc[0], "flagged", "Candidate remains ambiguous because of conflict, small score gap, or evidence conflict.")

    spatial = spatial.copy()
    spatial["voltage_diff"] = (spatial["dayzer_kv"] - spatial["pano_kv"]).abs()
    conflict = spatial[spatial["voltage_diff"] > 1].sort_values("distance_km").head(1)
    if not conflict.empty:
        row = conflict.iloc[0]
        final_row = final[final["dayzer_bus_id"].eq(row["dayzer_bus_id"])].head(1)
        reason = "Nearest-coordinate candidate has voltage conflict > 1 kV; graph-aware model avoids accepting that candidate as a clean final match."
        if not final_row.empty:
            add_case("voltage_conflict_rejected", final_row.iloc[0], "rejected_or_replaced", reason)
            cases[-1]["latlon_baseline_pano_bus"] = row["pano_bus_name"]
        else:
            add_case("voltage_conflict_rejected", row, "rejected_or_replaced", reason)

    return pd.DataFrame(cases)


def run_diagnostics(repo_root: Path) -> dict[str, pd.DataFrame]:
    out_dir = repo_root / "solution" / "assignment_3" / "outputs"
    scored = pd.read_csv(out_dir / "bus_candidate_pairs_scored.csv")
    final = pd.read_csv(out_dir / "bus_mapping_results.csv")
    baseline = pd.read_csv(out_dir / "bus_mapping_results_baseline.csv")

    ablation, variants = build_ablation(scored, final, baseline)
    spatial = variants["A_pure_latlon_nearest"]
    sensitivity = build_sensitivity(scored, final)
    one_to_one = build_one_to_one_diagnostics(scored, baseline, final)
    topology = build_topology_summary(scored, final, spatial)
    distance_voltage = build_distance_voltage_diagnostics(final)
    manual_review = build_manual_review_extended(final, spatial)
    case_studies = build_case_studies(scored, final, spatial)

    outputs = {
        "bus_mapping_ablation_results.csv": ablation,
        "bus_mapping_sensitivity_results.csv": sensitivity,
        "one_to_one_resolution_diagnostics.csv": one_to_one,
        "topology_contribution_summary.csv": topology,
        "distance_voltage_diagnostics.csv": distance_voltage,
        "bus_mapping_manual_review_extended.csv": manual_review,
        "bus_mapping_case_studies.csv": case_studies,
    }
    for name, df in outputs.items():
        df.to_csv(out_dir / name, index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Assignment 3 diagnostics.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    outputs = run_diagnostics(args.repo_root)
    for name, df in outputs.items():
        print(f"{name}: {df.shape[0]:,} rows x {df.shape[1]:,} columns")


if __name__ == "__main__":
    main()
