"""Topology-aware seed-and-propagate matching for Assignment 3."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _top_candidate_table(scored: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Return candidates ranked by a chosen score."""
    ranked = scored.sort_values(
        ["dayzer_bus_id", score_col, "voltage_score", "name_score", "topology_score"],
        ascending=[True, False, False, False, False],
    ).copy()
    ranked["propagation_candidate_rank"] = ranked.groupby("dayzer_bus_id").cumcount() + 1
    next_score = ranked.groupby("dayzer_bus_id")[score_col].shift(-1)
    ranked["propagation_score_gap"] = np.where(
        ranked["propagation_candidate_rank"].eq(1),
        ranked[score_col] - next_score.fillna(0),
        np.nan,
    )
    return ranked


def initialize_seed_matches(
    scored: pd.DataFrame,
    seed_score_threshold: float = 0.85,
    seed_gap_threshold: float = 0.05,
    max_seed_distance_km: float = 10.0,
) -> pd.DataFrame:
    """Select strict one-to-one seed matches from the baseline scored candidates."""
    top = scored.loc[scored["candidate_rank"].eq(1)].copy()
    seed_mask = (
        (top["composite_score"] >= seed_score_threshold)
        & (top["score_gap_to_second_best"] >= seed_gap_threshold)
        & (top["voltage_score"] >= 0.95)
        & (top["distance_km"].fillna(np.inf) <= max_seed_distance_km)
        & (~top["ambiguity_flag"])
    )
    seeds = top.loc[seed_mask].copy()
    if seeds.empty:
        return seeds

    seeds = seeds.sort_values("composite_score", ascending=False)
    seeds = seeds.drop_duplicates("pano_bus_id", keep="first")
    seeds["match_stage"] = "seed"
    seeds["propagation_iteration"] = 0
    seeds["propagation_support_count"] = 0
    seeds["propagation_support_ratio"] = 0.0
    seeds["propagation_score"] = seeds["composite_score"]
    return seeds


def add_propagation_features(
    candidates: pd.DataFrame,
    accepted_map: dict[str, str],
    dayzer_graph: dict[str, set[str]],
    pano_graph: dict[str, set[str]],
) -> pd.DataFrame:
    """Score candidate pairs by consistency with already accepted neighbor matches."""
    reverse_map = {pano_id: dayzer_id for dayzer_id, pano_id in accepted_map.items()}
    support_counts = []
    support_ratios = []
    reverse_support_counts = []
    reverse_support_ratios = []

    for row in candidates.itertuples(index=False):
        dz_id = row.dayzer_bus_id
        pn_id = row.pano_bus_id
        dz_neighbors = dayzer_graph.get(dz_id, set())
        pn_neighbors = pano_graph.get(pn_id, set())

        mapped_neighbor_panos = [accepted_map[n] for n in dz_neighbors if n in accepted_map]
        support = sum(1 for mapped_pano in mapped_neighbor_panos if mapped_pano in pn_neighbors)
        support_total = len(mapped_neighbor_panos)

        mapped_neighbor_dayzers = [reverse_map[n] for n in pn_neighbors if n in reverse_map]
        reverse_support = sum(1 for mapped_dayzer in mapped_neighbor_dayzers if mapped_dayzer in dz_neighbors)
        reverse_total = len(mapped_neighbor_dayzers)

        support_counts.append(support)
        support_ratios.append(support / support_total if support_total else 0.0)
        reverse_support_counts.append(reverse_support)
        reverse_support_ratios.append(reverse_support / reverse_total if reverse_total else 0.0)

    out = candidates.copy()
    out["propagation_support_count"] = support_counts
    out["propagation_support_ratio"] = support_ratios
    out["reverse_propagation_support_count"] = reverse_support_counts
    out["reverse_propagation_support_ratio"] = reverse_support_ratios
    out["propagation_consistency_score"] = (
        0.65 * out["propagation_support_ratio"] + 0.35 * out["reverse_propagation_support_ratio"]
    )
    support_boost = np.where(out["propagation_support_count"] >= 2, 0.10, np.where(out["propagation_support_count"] == 1, 0.05, 0.0))
    out["propagation_score"] = (
        0.72 * out["composite_score"].fillna(0)
        + 0.28 * out["propagation_consistency_score"].fillna(0)
        + support_boost
    ).clip(0, 1)
    return out


def iterative_seed_propagation(
    scored: pd.DataFrame,
    dayzer_graph: dict[str, set[str]],
    pano_graph: dict[str, set[str]],
    max_iterations: int = 8,
    seed_score_threshold: float = 0.85,
    propagation_threshold: float = 0.78,
    propagation_gap_threshold: float = 0.03,
    min_support_count: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run topology-aware seed-and-propagate matching.

    Returns accepted matches, all candidates with final propagation features, and
    an iteration log.
    """
    seeds = initialize_seed_matches(scored, seed_score_threshold=seed_score_threshold)
    accepted = seeds.copy()
    accepted_map = dict(zip(accepted["dayzer_bus_id"], accepted["pano_bus_id"])) if not accepted.empty else {}
    used_pano = set(accepted_map.values())
    log_rows = [
        {
            "iteration": 0,
            "accepted_this_iteration": int(len(accepted)),
            "total_accepted": int(len(accepted)),
            "stage": "seed",
        }
    ]

    final_candidates = add_propagation_features(scored, accepted_map, dayzer_graph, pano_graph)

    for iteration in range(1, max_iterations + 1):
        unresolved = final_candidates[
            ~final_candidates["dayzer_bus_id"].isin(accepted_map)
            & ~final_candidates["pano_bus_id"].isin(used_pano)
            & (final_candidates["voltage_score"] >= 0.65)
        ].copy()
        if unresolved.empty:
            break

        with_features = add_propagation_features(unresolved, accepted_map, dayzer_graph, pano_graph)
        ranked = _top_candidate_table(with_features, "propagation_score")
        proposed = ranked.loc[ranked["propagation_candidate_rank"].eq(1)].copy()
        accept_mask = (
            (proposed["propagation_score"] >= propagation_threshold)
            & (proposed["propagation_score_gap"] >= propagation_gap_threshold)
            & (proposed["propagation_support_count"] >= min_support_count)
            & (proposed["voltage_score"] >= 0.65)
        )
        additions = proposed.loc[accept_mask].sort_values("propagation_score", ascending=False)
        additions = additions.drop_duplicates("pano_bus_id", keep="first")
        if additions.empty:
            log_rows.append(
                {
                    "iteration": iteration,
                    "accepted_this_iteration": 0,
                    "total_accepted": int(len(accepted_map)),
                    "stage": "propagation",
                }
            )
            break

        additions["match_stage"] = "propagated"
        additions["propagation_iteration"] = iteration
        accepted = pd.concat([accepted, additions], ignore_index=True, sort=False)
        for row in additions.itertuples(index=False):
            accepted_map[row.dayzer_bus_id] = row.pano_bus_id
            used_pano.add(row.pano_bus_id)

        log_rows.append(
            {
                "iteration": iteration,
                "accepted_this_iteration": int(len(additions)),
                "total_accepted": int(len(accepted_map)),
                "stage": "propagation",
            }
        )
        final_candidates = add_propagation_features(scored, accepted_map, dayzer_graph, pano_graph)

    final_candidates = add_propagation_features(scored, accepted_map, dayzer_graph, pano_graph)
    final_ranked = _top_candidate_table(final_candidates, "propagation_score")
    log = pd.DataFrame(log_rows)
    return accepted, final_ranked, log


def build_propagated_results(
    scored: pd.DataFrame,
    accepted: pd.DataFrame,
    final_ranked: pd.DataFrame,
    min_score: float = 0.45,
) -> pd.DataFrame:
    """Create one final output row per Dayzer bus after propagation."""
    top = final_ranked.loc[final_ranked["propagation_candidate_rank"].eq(1)].copy()
    accepted_keys = accepted[["dayzer_bus_id", "pano_bus_id", "match_stage", "propagation_iteration"]].drop_duplicates()
    top = top.merge(accepted_keys, on=["dayzer_bus_id", "pano_bus_id"], how="left", suffixes=("", "_accepted"))

    top["pano_bus"] = top["pano_bus_name"]
    accepted_pair = top["match_stage"].notna()
    low_score = top["propagation_score"] < min_score
    top.loc[~accepted_pair & low_score, ["pano_bus", "pano_bus_id"]] = [pd.NA, pd.NA]
    fallback_stage = pd.Series(
        np.where(top["pano_bus"].notna(), "top_ranked_unpropagated", "unmatched"),
        index=top.index,
    )
    top["match_stage"] = top["match_stage"].fillna(fallback_stage)
    top["propagation_iteration"] = top["propagation_iteration"].fillna(-1).astype(int)
    top["accepted_graph_match"] = top["match_stage"].isin(["seed", "propagated"])

    top["one_to_one_conflict_flag"] = False
    claimants = top[top["pano_bus_id"].notna()].copy()
    if not claimants.empty:
        claimants = claimants.sort_values(
            ["pano_bus_id", "accepted_graph_match", "propagation_score", "composite_score"],
            ascending=[True, False, False, False],
        )
        claimants["claim_rank"] = claimants.groupby("pano_bus_id").cumcount() + 1
        loser_ids = set(claimants.loc[claimants["claim_rank"] > 1, "dayzer_bus_id"])
        top.loc[top["dayzer_bus_id"].isin(loser_ids), "one_to_one_conflict_flag"] = True
        top.loc[top["dayzer_bus_id"].isin(loser_ids), ["pano_bus", "pano_bus_id"]] = [pd.NA, pd.NA]
        top.loc[top["dayzer_bus_id"].isin(loser_ids), "match_stage"] = "one_to_one_conflict"

    top["ambiguity_flag"] = (
        (top["propagation_score_gap"] < 0.05)
        | top["evidence_conflict_flag"]
        | (top["match_stage"].eq("unmatched"))
        | top["one_to_one_conflict_flag"]
    )
    top["confidence_label"] = "low"
    top.loc[top["propagation_score"] >= 0.70, "confidence_label"] = "medium"
    top.loc[
        (top["propagation_score"] >= 0.85)
        & (top["propagation_score_gap"] >= 0.05)
        & (top["voltage_score"] >= 0.95)
        & (~top["ambiguity_flag"]),
        "confidence_label",
    ] = "high"
    top.loc[top["ambiguity_flag"], "confidence_label"] = "ambiguous"

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
        "propagation_score",
        "propagation_consistency_score",
        "propagation_support_count",
        "propagation_support_ratio",
        "propagation_score_gap",
        "match_stage",
        "propagation_iteration",
        "accepted_graph_match",
        "one_to_one_conflict_flag",
        "confidence_label",
        "ambiguity_flag",
        "score_gap_to_second_best",
        "degree_dayzer",
        "degree_pano",
        "neighbor_voltage_overlap",
        "neighbor_token_overlap",
    ]
    return top[result_cols].rename(columns={"dayzer_bus_name": "dayzer_bus"}).sort_values(
        ["match_stage", "propagation_score"], ascending=[True, False]
    )
