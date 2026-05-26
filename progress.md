# Progress

## 2026-05-19

- Imported source data from `https://github.com/ecesisllc/Summer2026`.
- Renamed the local project folder to `ECESIS`.
- Added AI project documentation policy files and repository documentation scaffold.
- Added a `solution/` workspace for Assignments 1, 2, and 3 with `notebooks/`, `execution_outputs/`, `solutions/`, and `summary_reports/` subfolders.
- Created the Assignment 1 starter notebook to load and inspect the Dayzer, Market, and Pano constraint CSV files.
- Renamed assignment-level `outputs/` folders to `execution_outputs/` and updated the Assignment 1 notebook to save generated tables there.
- Added Assignment 1 dataset inspection notes covering DAYZER, PJMISO, PANO, facilities, constraints, and contingencies.
- Expanded the Assignment 1 notebook with EDA for loading files, showing samples, and counting unique values.
- Added Assignment 1 missing-value and anomaly checks to the EDA notebook.
- Added Assignment 1 normalization, feature extraction, fuzzy matching, confidence scoring, ambiguity flags, and manual-review outputs.

## 2026-05-20

- Added an advanced Assignment 1 matching notebook with top-k candidate generation, multi-signal scoring, strategic review queue generation, manual-label templates, weak-supervision classifier fallback logic, and refined match outputs.
- Generated advanced Assignment 1 execution outputs, including candidate pairs, review queue, manual label template, refined matches, refined support rows, and validation summary.
- Added a 300-row manual-label random selection for hand grading and raised classifier training requirements to at least 100 `match` and 100 `not_match` labels.
- Added source-specific `rule_score_v2` weights while preserving `rule_score_v1`, and kept the manual-label random selection balanced at 100 rows per confidence bucket.

## 2026-05-22

- Added a supervised refinement stage that trains from reviewed manual labels, excludes uncertain labels, predicts match probabilities, and writes separate supervised refined outputs without overwriting the original matcher outputs.
- Added the Assignment 1 summary report and copied the final supervised match output into `solution/assignment_1/solutions/`.
- Created the Assignment 2 EDA/data-structure notebook with data discovery, timestamp handling, missingness, bus/zone count checks, seasonality plots, and explicit time-series leakage cautions.
- Updated and ran the Assignment 2 EDA notebook against the downloaded Parquet files using metadata, full zone reads, and deterministic bus samples to avoid loading all 331M bus rows into memory.
- Expanded Assignment 2 EDA with individual per-file profiling so each yearly bus and zone Parquet file is inspected separately, with bus and zone treated as different granularities.
- Implemented the Assignment 2 leakage-safe dual-path forecasting pipeline with source modules, baseline/model/evaluation notebooks, direct bus forecasts, zone-allocation forecasts, baseline forecasts, walk-forward fold definitions, and evaluation outputs.
- Extended Assignment 2 with cyclical temporal encodings, KMeans/GMM bus load-shape clustering, clustering-enhanced XGBoost experiments, advanced model comparison outputs, and visualization/report artifacts.
- Re-ran the Assignment 2 advanced experiments with CatBoost available and added SHAP summary outputs for the XGBoost + KMeans model.
- Added Assignment 2 research forecasting extensions covering Ridge, ElasticNet, RandomForest, LightGBM, LightGBM quantile intervals, lightweight TCN/GRU/LSTM sequence models, next-week research metrics, and research-style reporting.
- Added a focused Assignment 2 final model-selection layer that tests retained candidate models on 2023/2024 validation and a 2025 holdout subset, preserving deferred experimental models for reporting/future work.
- Drafted the Assignment 2 summary report covering EDA findings, leakage-safe validation, feature/model trials, retained/deferred decisions, focused 2025 holdout findings, and next steps.
- Added a verified Assignment 2 final recommendation update explaining that the prototype winner did not generalize best to 2025 and clarifying scenario-specific model choices.
- Finalized the Assignment 2 deliverables checklist confirming required final outputs and report coverage.
- Added an operational Assignment 2 CLI wrapper with YAML configs, train/predict/run scripts, output reports, and an optional Streamlit forecast viewer.

## 2026-05-23

- Discussed Assignment 3 as a graph-aware bus entity-resolution problem rather than simple fuzzy matching.
- Selected a first implementation strategy that uses latitude/longitude for spatial candidate generation, then ranks candidates with name, voltage, geographic, and topology features.
- Created `solution/assignment_3/` with notebooks, source modules, outputs, and reports for the bus-mapping workflow.
- Added reusable Assignment 3 modules for name normalization, haversine spatial candidate generation, graph fingerprint features, composite scoring, match selection, diagnostics, manual review sampling, and summary reporting.
- Generated `bus_candidate_pairs.csv`, `bus_candidate_pairs_scored.csv`, `bus_mapping_results.csv`, `bus_mapping_manual_review.csv`, input diagnostics, branch endpoint validity diagnostics, and coordinate diagnostics.
- Added and executed `01_bus_mapping_graph_alignment.ipynb`, including data inspection, coordinate EDA, normalization, spatial candidate generation, graph features, composite scoring, visualizations, and case studies.
- Added a coordinate EDA step before matching to verify latitude/longitude coverage and document missing-coordinate risk.
- Drafted `latitude_longitude_matching_plan.md` to document the initial spatial-matching idea, missing-coordinate concern, and rationale for exploring graph-aware alternatives.
- Rewrote `bus_mapping_summary.md` as a concise research-style infrastructure graph/entity-resolution study with methodology, results, examples, discussion, limitations, future work, and conclusion.
- Refined the final Assignment 3 strategy narrative around hybrid graph-aware bus entity resolution: high-confidence seeds, graph construction, topology-aware propagation, geographic validation, composite scoring, one-to-one conflict resolution, ambiguity handling, and manual review.
- Implemented the topology-aware seed-and-propagate model in `propagation.py`, including strict seed selection, neighbor-consistency scoring, iterative propagation, accepted graph-match outputs, propagation logs, and final one-to-one conflict handling.
- Regenerated Assignment 3 outputs with propagation enabled: 49 seed matches, 1,010 propagated graph matches, 1,059 accepted graph matches total, and zero duplicate nonblank Panorama IDs in the final mapping.
- Extended Assignment 3 with no-ground-truth evaluation diagnostics, ablation comparison, threshold sensitivity analysis, extended manual review sampling, one-to-one diagnostics, topology contribution analysis, distance/voltage diagnostics, and actual case studies.
- Added `diagnostics.py` and generated `bus_mapping_ablation_results.csv`, `bus_mapping_sensitivity_results.csv`, `bus_mapping_manual_review_extended.csv`, `one_to_one_resolution_diagnostics.csv`, `topology_contribution_summary.csv`, `distance_voltage_diagnostics.csv`, and `bus_mapping_case_studies.csv`.
- Updated `bus_mapping_summary.md` with baseline-vs-graph-aware diagnostics, ablation/sensitivity interpretation, manual review design, limitations, and final no-ground-truth conclusion.
- Verified the Assignment 3 pipeline and notebook end to end and confirmed source modules compile.
