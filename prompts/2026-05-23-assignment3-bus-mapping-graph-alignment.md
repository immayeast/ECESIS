# 2026-05-23 - Assignment 3 Bus Mapping Graph Alignment

## Context

Assignment 3 asks for a mapping between Dayzer buses and Panorama buses using bus metadata and branch connectivity. The user initially understood the problem as a more granular fuzzy-matching task and proposed using latitude/longitude to create a city/area map, then searching within a spherical neighborhood or k-nearest-neighbor region to rank plausible matches.

## Key Prompts and Requests

1. Discuss whether the longitude/latitude neighborhood method is valid and what alternatives might be better.
2. Compare the spatial method with a graph-first or seed-and-propagate alternative.
3. Implement Assignment 3 in the existing Summer2026/ECESIS solution repository without overwriting Assignments 1 or 2.
4. Create:
   - `assignment_3/notebooks/01_bus_mapping_graph_alignment.ipynb`
   - `assignment_3/src/normalize.py`
   - `assignment_3/src/spatial.py`
   - `assignment_3/src/graph_features.py`
   - `assignment_3/src/matching.py`
   - `assignment_3/src/evaluation.py`
   - output CSVs
   - summary report
5. Add a coordinate EDA step before matching to confirm latitude/longitude coverage and missingness.
6. Draft a report explaining the purpose of the longitude/latitude method and the need to explore a better missing-value approach.
7. Rewrite `bus_mapping_summary.md` as a polished research-style infrastructure graph/entity-resolution report.
8. Update `ai_journal`, `decisions.md`, `progress.md`, and prompt documentation to follow the repository's documentation policy.
9. Refine the final strategy as hybrid graph-aware bus entity resolution, emphasizing high-confidence seeds, topology-aware propagation, geographic validation, voltage feasibility, one-to-one conflict resolution, ambiguity handling, and manual review.
10. Implement the final topology-aware seed-and-propagate model, not only the report narrative.
11. Extend Assignment 3 with no-ground-truth evaluation diagnostics, ablation study, sensitivity analysis, extended manual review, case studies, one-to-one diagnostics, topology contribution analysis, and report polish.

## AI Suggestions

- Treat bus mapping as graph-aware entity resolution.
- Use longitude/latitude as the first-stage candidate-generation layer, not as the full solution.
- Preserve name, voltage, distance, topology, and ambiguity features for auditability.
- Build separate Dayzer and Panorama graphs from branch lists.
- Use local graph fingerprints to refine candidate rankings.
- Use coordinate EDA to document missing-coordinate risk before relying on spatial matching.
- Treat topology-aware seed propagation as the next extension for missing-coordinate or weak-name cases.
- Present topology-aware propagation as the strongest conceptual differentiator because electrical neighborhood consistency is central to the assignment.
- Implement propagation as a second-stage matcher using already-accepted neighbor matches as structural evidence.
- Use internal consistency diagnostics rather than true-accuracy claims because no labeled match ground truth is available.

## Accepted Direction

The user chose to proceed with the original longitude/latitude-first solution, while preserving graph-aware features and documenting graph-based alternatives for later exploration.

## Important Design Outcome

The final Assignment 3 implementation uses a hybrid approach:

```text
high-confidence seed generation
  -> graph construction
  -> topology-aware propagation / local topology scoring
  -> geographic validation
  -> composite scoring
  -> ambiguity and one-to-one conflict handling
  -> manual review outputs
```

The final report frames the assignment as infrastructure graph alignment rather than a generic fuzzy-matching exercise.

## Implemented Propagation Stage

The final implementation adds:

- strict seed match selection,
- iterative neighbor-consistency propagation,
- propagation support counts and ratios,
- accepted graph-match outputs,
- propagation iteration logs,
- baseline output preservation,
- one-to-one conflict resolution with duplicate Panorama claims removed from final nonblank matches.

## Implemented Evaluation Extension

The diagnostics layer adds:

- baseline-vs-graph-aware ablation results,
- sensitivity analysis over radius, topology weight, and confidence threshold,
- one-to-one conflict diagnostics,
- topology contribution summary,
- distance and voltage diagnostics,
- extended 300-row manual review sample,
- five actual case studies drawn from output tables.
