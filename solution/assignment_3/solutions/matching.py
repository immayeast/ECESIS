"""End-to-end bus mapping pipeline for Assignment 3."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import (
    bus_diagnostics,
    coordinate_diagnostics,
    inspect_table,
    make_manual_review_sample,
    write_summary_report,
)
from graph_features import (
    add_graph_pair_features,
    build_graph_from_branches,
    endpoint_validity,
    graph_fingerprints,
)
from normalize import jaccard, name_tokens, numeric_tokens, standardize_bus_table
from propagation import build_propagated_results, iterative_seed_propagation
from spatial import distance_score, generate_spatial_candidates


RANDOM_STATE = 42


def name_similarity(left: str, right: str) -> float:
    """Fuzzy string similarity using the standard library."""
    return SequenceMatcher(None, left or "", right or "").ratio()


def same_voltage_band(left: object, right: object) -> bool:
    try:
        left = float(left)
        right = float(right)
    except (TypeError, ValueError):
        return False
    if pd.isna(left) or pd.isna(right):
        return False
    if left >= 300 and right >= 300:
        return True
    if 100 <= left < 300 and 100 <= right < 300:
        return True
    return left < 100 and right < 100


def add_attribute_features(candidates: pd.DataFrame, dayzer: pd.DataFrame, pano: pd.DataFrame, radius_km: float) -> pd.DataFrame:
    """Add name, voltage, geographic, and rank features to candidate pairs."""
    dz = dayzer[["dayzer_bus_id", "normalized_bus_name"]].rename(columns={"normalized_bus_name": "dayzer_normalized_name"})
    pn = pano[["pano_bus_id", "normalized_bus_name"]].rename(columns={"normalized_bus_name": "pano_normalized_name"})
    out = candidates.merge(dz, on="dayzer_bus_id", how="left").merge(pn, on="pano_bus_id", how="left")

    out["fuzzy_name_similarity"] = [
        name_similarity(l, r) for l, r in zip(out["dayzer_normalized_name"], out["pano_normalized_name"])
    ]
    out["token_overlap"] = [
        jaccard(name_tokens(l), name_tokens(r)) for l, r in zip(out["dayzer_normalized_name"], out["pano_normalized_name"])
    ]
    out["numeric_token_overlap"] = [
        jaccard(numeric_tokens(l), numeric_tokens(r))
        for l, r in zip(out["dayzer_normalized_name"], out["pano_normalized_name"])
    ]
    out["name_score"] = (
        0.60 * out["fuzzy_name_similarity"] + 0.25 * out["token_overlap"] + 0.15 * out["numeric_token_overlap"]
    )
    out["exact_voltage_match"] = np.isclose(out["dayzer_kv"], out["pano_kv"], equal_nan=False)
    out["voltage_difference"] = (out["dayzer_kv"] - out["pano_kv"]).abs()
    out["same_voltage_band"] = [same_voltage_band(l, r) for l, r in zip(out["dayzer_kv"], out["pano_kv"])]
    out["voltage_score"] = np.where(
        out["exact_voltage_match"],
        1.0,
        np.where(out["same_voltage_band"], 0.65, np.where(out["voltage_difference"].isna(), 0.25, 0.0)),
    )
    out["distance_score"] = out["distance_km"].map(lambda x: distance_score(x, radius_km=radius_km))
    out["name_rank"] = out.groupby("dayzer_bus_id")["name_score"].rank(method="first", ascending=False).astype(int)
    out["voltage_compatible_rank"] = (
        out.assign(voltage_compatible=out["voltage_score"] >= 0.65)
        .sort_values(["dayzer_bus_id", "voltage_compatible", "distance_rank"], ascending=[True, False, True])
        .groupby("dayzer_bus_id")
        .cumcount()
        + 1
    )
    return out


def add_scores(candidates: pd.DataFrame) -> pd.DataFrame:
    """Calculate composite scores and per-Dayzer ranking labels."""
    out = candidates.copy()
    out["composite_score"] = (
        0.30 * out["distance_score"].fillna(0)
        + 0.25 * out["voltage_score"].fillna(0)
        + 0.25 * out["name_score"].fillna(0)
        + 0.20 * out["topology_score"].fillna(0)
    )
    out = out.sort_values(["dayzer_bus_id", "composite_score", "voltage_score", "name_score"], ascending=[True, False, False, False])
    out["candidate_rank"] = out.groupby("dayzer_bus_id").cumcount() + 1
    next_score = out.groupby("dayzer_bus_id")["composite_score"].shift(-1)
    out["score_gap_to_second_best"] = np.where(out["candidate_rank"].eq(1), out["composite_score"] - next_score.fillna(0), np.nan)
    out["evidence_conflict_flag"] = (
        (out["voltage_score"] < 0.65)
        | ((out["distance_km"] > 50) & out["distance_km"].notna())
        | ((out["name_score"] < 0.25) & (out["topology_score"] < 0.45))
    )
    out["ambiguity_flag"] = np.where(
        out["candidate_rank"].eq(1),
        (out["score_gap_to_second_best"] < 0.05) | out["evidence_conflict_flag"],
        False,
    )
    out["confidence_label"] = "low"
    high_mask = (out["composite_score"] >= 0.85) & (out["score_gap_to_second_best"] >= 0.05) & out["candidate_rank"].eq(1)
    medium_mask = (out["composite_score"] >= 0.70) & out["candidate_rank"].eq(1)
    out.loc[medium_mask, "confidence_label"] = "medium"
    out.loc[high_mask, "confidence_label"] = "high"
    out.loc[out["ambiguity_flag"] & out["candidate_rank"].eq(1), "confidence_label"] = "ambiguous"
    return out


def select_matches(scored: pd.DataFrame, min_score: float = 0.45) -> pd.DataFrame:
    """Select one candidate per Dayzer bus and resolve duplicate Panorama claims."""
    top = scored.loc[scored["candidate_rank"].eq(1)].copy()
    top["pano_bus"] = top.get("pano_bus_name")
    top.loc[top["composite_score"] < min_score, ["pano_bus", "pano_bus_id"]] = [pd.NA, pd.NA]

    claimed = top[top["pano_bus_id"].notna()].copy()
    claimed["claim_rank"] = claimed.groupby("pano_bus_id")["composite_score"].rank(method="first", ascending=False)
    duplicate_losers = claimed.loc[claimed["claim_rank"] > 1, "dayzer_bus_id"]
    top.loc[top["dayzer_bus_id"].isin(duplicate_losers), "ambiguity_flag"] = True
    top.loc[top["dayzer_bus_id"].isin(duplicate_losers), "confidence_label"] = "ambiguous"
    top.loc[top["dayzer_bus_id"].isin(duplicate_losers), ["pano_bus", "pano_bus_id"]] = [pd.NA, pd.NA]

    result_cols = [
        "dayzer_bus_name",
        "pano_bus",
        "dayzer_bus_id",
        "pano_bus_id",
        "dayzer_kv",
        "pano_kv",
        "distance_km",
        "name_score",
        "voltage_score",
        "topology_score",
        "composite_score",
        "confidence_label",
        "ambiguity_flag",
        "score_gap_to_second_best",
        "degree_dayzer",
        "degree_pano",
        "neighbor_voltage_overlap",
        "neighbor_token_overlap",
    ]
    result = top[result_cols].rename(columns={"dayzer_bus_name": "dayzer_bus"})
    return result.sort_values(["confidence_label", "composite_score"], ascending=[True, False])


def run_pipeline(
    repo_root: Path,
    radius_km: float = 20.0,
    top_k: int = 10,
    min_score: float = 0.45,
    use_propagation: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the full Assignment 3 matching workflow."""
    data_dir = repo_root / "data" / "Assignment 3 - bus_mapping"
    out_dir = repo_root / "solution" / "assignment_3" / "outputs"
    report_dir = repo_root / "solution" / "assignment_3" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    dayzer_bus_raw = pd.read_csv(data_dir / "DAYZER_BUS_GEO.csv")
    pano_bus_raw = pd.read_csv(data_dir / "PANO_BUS_GEO.csv")
    dayzer_branch = pd.read_csv(data_dir / "DAYZER_BRANCH.csv")
    pano_branch = pd.read_csv(data_dir / "PANO_BRANCH.csv")

    dayzer_bus = standardize_bus_table(dayzer_bus_raw, "dayzer")
    pano_bus = standardize_bus_table(pano_bus_raw, "pano")

    candidates = generate_spatial_candidates(dayzer_bus, pano_bus, radius_km=radius_km, top_k=top_k)
    candidates.to_csv(out_dir / "bus_candidate_pairs.csv", index=False)

    attr = add_attribute_features(candidates, dayzer_bus, pano_bus, radius_km=radius_km)
    dayzer_graph = build_graph_from_branches(dayzer_branch, dayzer_bus, "dayzer")
    pano_graph = build_graph_from_branches(pano_branch, pano_bus, "pano")
    dayzer_graph_features = graph_fingerprints(dayzer_bus, dayzer_graph, "dayzer")
    pano_graph_features = graph_fingerprints(pano_bus, pano_graph, "pano")
    with_graph = add_graph_pair_features(attr, dayzer_graph_features, pano_graph_features)
    scored = add_scores(with_graph)
    scored.to_csv(out_dir / "bus_candidate_pairs_scored.csv", index=False)

    baseline_results = select_matches(scored, min_score=min_score)
    baseline_results.to_csv(out_dir / "bus_mapping_results_baseline.csv", index=False)

    if use_propagation:
        accepted_matches, propagated_candidates, propagation_log = iterative_seed_propagation(
            scored,
            dayzer_graph=dayzer_graph,
            pano_graph=pano_graph,
        )
        seed_matches = accepted_matches.loc[accepted_matches["match_stage"].eq("seed")].copy()
        seed_matches.to_csv(out_dir / "bus_mapping_seed_matches.csv", index=False)
        accepted_matches.to_csv(out_dir / "bus_mapping_accepted_graph_matches.csv", index=False)
        propagated_candidates.to_csv(out_dir / "bus_candidate_pairs_propagated.csv", index=False)
        propagation_log.to_csv(out_dir / "bus_mapping_propagation_log.csv", index=False)
        results = build_propagated_results(scored, accepted_matches, propagated_candidates, min_score=min_score)
    else:
        seed_matches = pd.DataFrame()
        accepted_matches = pd.DataFrame()
        propagated_candidates = scored.copy()
        propagation_log = pd.DataFrame()
        results = baseline_results

    results.to_csv(out_dir / "bus_mapping_results.csv", index=False)

    review = make_manual_review_sample(results, random_state=RANDOM_STATE)
    review.to_csv(out_dir / "bus_mapping_manual_review.csv", index=False)

    diagnostics = pd.DataFrame(
        [
            inspect_table(dayzer_bus_raw, "DAYZER_BUS_GEO"),
            inspect_table(pano_bus_raw, "PANO_BUS_GEO"),
            inspect_table(dayzer_branch, "DAYZER_BRANCH"),
            inspect_table(pano_branch, "PANO_BRANCH"),
        ]
    )
    diagnostics.to_csv(out_dir / "input_table_diagnostics.csv", index=False)
    pd.DataFrame([bus_diagnostics(dayzer_bus, "dayzer"), bus_diagnostics(pano_bus, "pano")]).to_csv(
        out_dir / "bus_diagnostics.csv", index=False
    )
    pd.DataFrame(
        [
            coordinate_diagnostics(dayzer_bus_raw, "dayzer"),
            coordinate_diagnostics(pano_bus_raw, "pano"),
        ]
    ).to_csv(out_dir / "coordinate_diagnostics.csv", index=False)

    endpoint_summary = {
        "dayzer": endpoint_validity(dayzer_branch, dayzer_bus, "dayzer"),
        "pano": endpoint_validity(pano_branch, pano_bus, "pano"),
    }
    pd.DataFrame(endpoint_summary).T.to_csv(out_dir / "branch_endpoint_validity.csv")

    write_summary_report(
        report_dir / "bus_mapping_auto_summary.md",
        dayzer_bus_count=len(dayzer_bus),
        pano_bus_count=len(pano_bus),
        results=results,
        endpoint_summary=endpoint_summary,
    )

    return {
        "dayzer_bus": dayzer_bus,
        "pano_bus": pano_bus,
        "candidates": candidates,
        "scored_candidates": scored,
        "baseline_results": baseline_results,
        "seed_matches": seed_matches,
        "accepted_matches": accepted_matches,
        "propagated_candidates": propagated_candidates,
        "propagation_log": propagation_log,
        "results": results,
        "manual_review": review,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Assignment 3 bus mapping pipeline.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--radius-km", type=float, default=20.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--no-propagation", action="store_true", help="Disable seed-and-propagate matching.")
    args = parser.parse_args()
    outputs = run_pipeline(
        args.repo_root,
        radius_km=args.radius_km,
        top_k=args.top_k,
        min_score=args.min_score,
        use_propagation=not args.no_propagation,
    )
    print(f"Wrote {len(outputs['results']):,} proposed bus mappings.")
    if not outputs["propagation_log"].empty:
        accepted = int(outputs["propagation_log"]["total_accepted"].max())
        print(f"Accepted {accepted:,} seed/propagated one-to-one matches.")


if __name__ == "__main__":
    main()
