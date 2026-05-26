# Decisions

## 2026-05-19: Repository Name

Decision: Name the working repository `ECESIS`.

Rationale: The user requested this repository name.

Alternatives considered:
- `Summer2026-work`: initial temporary folder name used during import.

Verification:
- Local folder renamed to `ECESIS`.

## 2026-05-19: Solution Workspace Structure

Decision: Use `solution/assignment_1`, `solution/assignment_2`, and `solution/assignment_3`, each with `notebooks/`, `execution_outputs/`, `solutions/`, and `summary_reports/`.

Rationale: The user requested a solution folder split by assignment and deliverable type, then clarified that notebook-generated graphs and tables should live in `execution_outputs/`. Snake_case assignment folder names keep paths simple for command-line tools, notebooks, and scripts.

Alternatives considered:
- Human-readable folder names such as `Assignment 1`: closer to the source data naming, but less convenient in scripts because of spaces.
- Folder name `outputs/`: shorter, but less specific than `execution_outputs/` for notebook-generated artifacts.
- A deliverable-first layout such as `solution/notebooks/assignment_1`: less aligned with the user's request to contain Assignment 1, 2, and 3.

Verification:
- Created all requested assignment and deliverable folders.
- Renamed assignment output folders to `execution_outputs/`.

## 2026-05-19: Assignment 1 Matching Reference

Decision: Use PJMISO/Market as the reference dataset for constraint matching, then select best-scoring PANO and DAYZER matches for each PJMISO record.

Rationale: PJMISO has the richest descriptive fields among the Assignment 1 files. The user specified the scoring formula and requested PJMISO as the record reference. `TOZONE` is missing for 19.64% of PJMISO rows, so zone/token overlap is retained as a weak 10% feature.

Alternatives considered:
- Use PANO as the reference because it has more rows. This would produce many unmatched PJMISO records and make the final output less aligned with the requested PJMISO-centered workflow.
- Use DAYZER as the reference. This is weaker because DAYZER only has `CID` and `NAME`.

Verification:
- Generated 5,230 PJMISO-centered match rows.
- Generated 10,460 support-score rows, one PANO and one DAYZER score per PJMISO record.
- Generated manual-review samples for 20 highest-confidence, 20 medium-confidence, and 20 ambiguous matches.

## 2026-05-20: Advanced Matching Review Strategy

Decision: Add a separate advanced matching notebook that generates top-k candidate pairs, broader review queues, manual-label templates, and optional classifier-based refinement.

Rationale: Reviewing only 60 sampled rows can miss likely errors. The advanced workflow reviews selected matches that are ambiguous, low-scoring, or close-margin, plus a random high-confidence audit sample. Manual labels are stored in CSV so the decisions can be audited and reused for model training.

Alternatives considered:
- Keep only the 60-row sample: simpler, but less defensible for ambiguous and low-confidence matches.
- Put all advanced logic into the existing EDA notebook: fewer files, but the notebook would become too long and harder to review.
- Require embeddings immediately: potentially useful, but unnecessary for the first implementation and adds dependency/API risk.

Verification:
- Generated 258,383 advanced candidate pairs.
- Generated an 8,797-row advanced review queue and manual label template.
- Generated 5,230 refined PJMISO match rows and 10,460 refined support rows.
- Validation showed every PJMISO row and PJMISO/source pair is present, zero below-threshold matches were accepted, and zero flagged selected matches are missing from the review queue.

## 2026-05-20: Manual Labeling Threshold

Decision: Use a 300-row hand-grading sample and require at least 100 `match` plus 100 `not_match` labels before classifier training.

Rationale: The user wants the more reliable 200-300 label strategy from the original recommendation. A 100/100 minimum reduces the risk of overfitting compared with the earlier 30/30 threshold.

Alternatives considered:
- Keep the 30/30 training gate: faster to start, but less defensible.
- Require all 300 labels before training: more rigorous, but unnecessarily blocks early classifier diagnostics if 200 balanced labels are already available.

Verification:
- Generated `manual_labels_random_selection.csv` with 100 high-confidence, 100 medium-confidence, and 100 low/ambiguous rows.

## 2026-05-20: Source-Specific Rule Score

Decision: Preserve the original shared score as `rule_score_v1` and use a source-specific `rule_score_v2` for ranking.

Rationale: PANO has both facility and contingency fields, so contingency similarity should remain important. DAYZER only has `CID` and `NAME`, so its score should rely more heavily on facility/interface similarity and avoid punishing the missing contingency signal.

Alternatives considered:
- Keep one shared formula for both sources: simpler, but unfair to DAYZER.
- Use only the future classifier: not available until enough manual labels exist.

Verification:
- Regenerated advanced outputs with both `rule_score_v1` and `rule_score_v2`.
- Confirmed refined outputs still include all 5,230 PJMISO rows and 10,460 PJMISO/source support rows.
- Confirmed manual-label random selection remains 100 high-confidence, 100 medium-confidence, and 100 low/ambiguous rows.

## 2026-05-22: Supervised Refinement Outputs

Decision: Add supervised refinement as a separate stage and write `supervised_*` outputs rather than overwriting the original refined match files.

Rationale: The user requested the original matching workflow remain intact. Separate supervised outputs preserve auditability and make it easy to compare rule-based confidence with classifier-refined confidence.

Alternatives considered:
- Overwrite `constraint_matches_refined.csv` and `constraint_match_support_refined.csv`: simpler filenames, but it would remove the original rule-based outputs.
- Train only after 100/100 balanced labels: stricter, but the user explicitly asked to use the reviewed labels now.

Verification:
- Trained Logistic Regression from 299 usable labels after excluding one uncertain label.
- Generated 5,230 supervised refined match rows and 10,460 supervised support rows.
- Generated diagnostics, logistic coefficients, candidate probabilities, and validation summary.
- Confirmed no accepted supervised match has probability below 0.50.

## 2026-05-22: Assignment 2 Time-Series Split Policy

Decision: Treat Assignment 2 as strict time-series data and avoid random shuffling for sampling, validation, or splitting.

Rationale: Forecasting tasks must avoid future-data leakage. The 2025 target period must remain held out for later next-day and next-month evaluation.

Alternatives considered:
- Random train/test split: inappropriate because it leaks future patterns into training.
- Random row sampling for inspection: avoided in favor of deterministic time windows or aggregate summaries.

Verification:
- Added the policy to the Assignment 2 EDA notebook.
- Recorded 2022-2024 as training/history and 2025 as held-out target period in the notebook split-policy cell.

## 2026-05-22: Assignment 2 Large-Data EDA Strategy

Decision: Use Parquet metadata, full zone-level reads, and bounded deterministic bus samples for Assignment 2 EDA.

Rationale: The bus-level files contain 331,157,941 total rows across 2022-2025. Full in-memory bus EDA is slow and unnecessary for initial structure inspection. Zone files are small enough to load fully and are suitable for the initial seasonality and distribution plots.

Alternatives considered:
- Full bus data load: possible but too heavy for iterative notebook EDA.
- Random bus samples: avoided because this is strict time-series data.

Verification:
- Generated Assignment 2 metadata, structure, missingness, count, seasonality, and plot outputs.
- Confirmed zone-level EDA used 315,153 zone rows.
- Confirmed deterministic bus sample used 800,000 rows.

## 2026-05-22: Assignment 2 Individual File Profiling

Decision: Profile each yearly bus and zone Parquet file individually, while keeping bus and zone outputs separate because they describe different levels of the system.

Rationale: Matching years do not mean matching data meaning. Zone files are aggregate time-series records and can be read fully. Bus files are much larger bus-level records, so per-file EDA uses Parquet metadata plus deterministic first-batch samples for structure, examples, and sample missingness.

Alternatives considered:
- Combine bus and zone by year: simpler, but it hides the granularity difference.
- Fully load every bus file for per-file EDA: more exhaustive, but too heavy for iterative notebook work.

Verification:
- Generated `assignment2_individual_file_profiles.csv` with 8 file-level profiles.
- Generated `assignment2_individual_file_missingness.csv` and `assignment2_individual_file_examples.csv`.
- Re-ran the notebook successfully and saved it without embedded outputs.

## 2026-05-22: Assignment 2 Forecasting Pipeline Prototype

Decision: Implement the forecasting workflow as reusable `src/` modules plus notebooks, and execute the first walk-forward fold on a deterministic high-volume bus subset.

Rationale: The full bus dataset contains hundreds of millions of rows, so the implementation needs to be scalable without forcing every notebook run to process all data. A bounded prototype validates the leakage-safe logic, output schemas, and model comparison workflow before scaling to longer folds and the 2025 holdout.

Alternatives considered:
- Train immediately on all bus rows from 2022-2024: closer to final production scale, but too heavy for iterative notebook development.
- Use random bus samples: avoided because the assignment is strict time series and the user requested no random sampling/splitting.

Verification:
- Created `load_data.py`, `features.py`, `baselines.py`, `models.py`, `evaluation.py`, and `pipeline.py`.
- Created notebooks `02_baselines_and_features.ipynb`, `03_walk_forward_models.ipynb`, and `04_evaluation_and_report.ipynb`.
- Generated `bus_forecast_direct.csv`, `bus_forecast_zone_allocated.csv`, `baseline_forecasts.csv`, `evaluation_summary.csv`, and `walk_forward_results.csv`.
- Confirmed all Assignment 2 notebooks execute successfully and are saved without embedded outputs.

## 2026-05-22: Assignment 2 Advanced Temporal and Clustering Features

Decision: Add cyclical encodings and training-window load-shape clustering as optional feature enhancements, then compare raw-time, cyclical, clustered, baseline, and hierarchical model variants.

Rationale: EDA showed strong periodic demand behavior by hour, weekday, and season. Sine/cosine encodings preserve adjacency across cycle boundaries. Bus load-shape clusters provide structural context so models can share information across buses with similar demand patterns without aggregating away bus-level targets.

Alternatives considered:
- Use only raw integer time fields: simpler, but creates artificial distance between adjacent cycle endpoints.
- Use clustering as a standalone forecast: rejected because clustering is better suited here as contextual feature engineering.
- Require CatBoost: skipped because CatBoost is not installed locally; the experiment log records it as unavailable.

Verification:
- Generated `bus_clusters.csv`, cluster stats, cluster diagnostics, KMeans centers, and GMM distribution outputs.
- Generated `advanced_model_predictions.csv`, `advanced_evaluation_summary.csv`, `advanced_model_experiment_log.csv`, and `advanced_comparative_summary.csv`.
- Added `05_advanced_temporal_clustering_models.ipynb` and confirmed it executes without embedded outputs.

## 2026-05-22: Assignment 2 CatBoost and SHAP Follow-Up

Decision: Re-run the advanced model comparison after confirming CatBoost is available, and run SHAP only for the XGBoost + KMeans clustered variant.

Rationale: CatBoost was previously logged as unavailable, but the package is now importable in the active Python environment. SHAP is most useful on the strongest clustered boosting model and is kept bounded to avoid expensive attribution runs as the workflow scales.

Alternatives considered:
- Run SHAP for every model: more exhaustive, but unnecessary and computationally heavier.
- Leave CatBoost as skipped: inaccurate after confirming the package is available.

Verification:
- `direct_catboost_cyclical_kmeans` completed and appears in `advanced_model_experiment_log.csv`.
- Generated `shap_summary_direct_xgb_cyclical_kmeans.csv` and `.png`.
- Re-ran `05_advanced_temporal_clustering_models.ipynb` successfully and saved it without embedded outputs.

## 2026-05-22: Assignment 2 Research Forecasting Extensions

Decision: Add a research-oriented extension layer with classical models, probabilistic quantile forecasts, lightweight sequence models, and graph/spatial future-work notes.

Rationale: The assignment is moving beyond a single benchmark toward a comparative forecasting framework. Ridge and ElasticNet provide interpretable coefficient baselines; RandomForest and LightGBM provide additional nonlinear baselines; quantile forecasts add operational uncertainty; small TCN/GRU/LSTM experiments expose sequence-model tradeoffs without requiring a full deep-learning training pipeline.

Alternatives considered:
- Fully tune deep learning models: deferred because the current goal is exploratory and compute-aware.
- Implement graph neural networks immediately: deferred because bus topology data are not available in Assignment 2.
- Rerun heavy research training on every notebook execution: avoided by making the research notebook read generated artifacts by default.

Verification:
- Generated research classical predictions, evaluation, runtime logs, and linear coefficients.
- Generated LightGBM quantile P10/P50/P90 forecasts and interval visualizations.
- Generated lightweight TCN/GRU/LSTM exploratory day-ahead outputs.
- Added `06_research_forecasting_extensions.ipynb` and confirmed it executes in artifact-reading mode without embedded outputs.

## 2026-05-22: Assignment 2 Focused Final Model Set

Decision: Narrow expanded validation and 2025 holdout testing to `baseline_lag_168h`, `baseline_historical_mean`, `direct_xgb_cyclical_kmeans`, `research_lightgbm_cyclical_kmeans`, `research_random_forest_cyclical_kmeans`, and `zone_allocated_hgb`.

Rationale: Prototype results should narrow the candidate set, not decide the final model. The retained set covers the strongest sanity baseline, a stable interpretable baseline, two scalable nonlinear direct models, a RandomForest robustness check, and the hierarchical zone-allocation strategy. Ridge/ElasticNet, CatBoost, direct HGB, deep sequence models, GMM-required variants, and quantile models remain documented but deferred from expensive final point-forecast testing.

Alternatives considered:
- Test every experimental model through 2025: too expensive and strategically noisy.
- Declare the prototype winner final: rejected because the prototype only used 2022 Jan-Mar to 2022 Apr.

Verification:
- Generated `final_model_validation_results.csv`, `final_2025_holdout_results.csv`, and `final_model_selection_summary.md`.
- Added per-zone, per-cluster, per-bus error distribution, runtime, retained/deferred rationale, and evaluation-scope outputs.
- Added `07_final_model_selection.ipynb` and confirmed it executes in artifact-reading mode without embedded outputs.

## 2026-05-23: Assignment 3 Hybrid Bus-Mapping Strategy

Decision: Implement Assignment 3 as a hybrid graph-aware entity-resolution workflow: use latitude/longitude for first-stage candidate generation, then rank candidates with name similarity, voltage compatibility, geographic distance, and local topology features.

Rationale: The user first proposed a spatial-neighborhood method, using longitude and latitude to build a local map and rank nearby Panorama candidates for each Dayzer bus. This is a valid and interpretable first layer because buses represent physical infrastructure. However, the assignment emphasizes network structure, and geography alone can fail in dense substations, missing-coordinate cases, and situations where two systems model buses at different granularity. Adding graph features makes the approach better aligned with the assignment objective.

Alternatives considered:
- Pure fuzzy name matching: simpler, but too sensitive to abbreviations and naming conventions across systems.
- Pure latitude/longitude nearest-neighbor matching: strong candidate filter, but insufficient when coordinates are missing, noisy, or shared by nearby buses.
- Full global graph alignment as the first implementation: conceptually attractive, but too heavy for an open-ended assignment without labels and less interpretable as a first deliverable.
- Seed-and-propagate graph matching only: useful future direction, but best explored after creating auditable spatial/name/voltage candidates.

Verification:
- Generated 892,106 candidate pairs.
- Generated one proposed result row for each of 10,799 Dayzer buses.
- Preserved component scores and graph features in `bus_candidate_pairs_scored.csv`.
- Exported final mappings, candidate support, and manual review samples under `solution/assignment_3/outputs/`.

## 2026-05-23: Assignment 3 Coordinate EDA Before Spatial Matching

Decision: Add a pre-matching coordinate EDA step before relying on latitude and longitude for candidate generation.

Rationale: Spatial matching depends on coordinate availability and plausibility. The user specifically wanted to make sure there were no major missing latitude/longitude issues before matching. The EDA also documents where missing coordinates may create harder matching conditions and motivate topology-aware fallback methods.

Alternatives considered:
- Trust the coordinate fields without EDA: faster, but less defensible because missing coordinates directly affect candidate generation.
- Drop missing-coordinate buses: rejected because each Dayzer bus still needs a proposed match status or review path.
- Use name and voltage only for all records: avoids coordinate missingness, but discards strong physical evidence for the majority of buses.

Verification:
- Created `coordinate_diagnostics.csv`.
- Confirmed Dayzer has 681 rows missing latitude/longitude, with 93.69% coordinate coverage.
- Confirmed Panorama has 100.00% coordinate coverage.
- Confirmed no invalid non-missing coordinate ranges in either bus list.
- Re-executed the Assignment 3 notebook after adding the EDA section.

## 2026-05-23: Assignment 3 Report Framing

Decision: Rewrite the Assignment 3 summary report as a concise infrastructure graph/entity-resolution study rather than a generic notebook dump.

Rationale: The assignment is open-ended and evaluates reasoning as much as accuracy. A research-style report better explains why each signal matters, how ambiguity is handled, and why the result should be interpreted as an auditable matching framework rather than perfect ground truth.

Alternatives considered:
- Keep a short generated metrics report: concise, but too thin to communicate the methodology and design trade-offs.
- Write a long exploratory notebook narrative only: useful for process, but less polished as a final deliverable.
- Claim exact correctness from the scores: rejected because no labeled ground truth was provided.

Verification:
- Rewrote `solution/assignment_3/reports/bus_mapping_summary.md`.
- Included actual run metrics, coordinate coverage, branch endpoint validity, and three evidence-based case studies.
- Kept claims bounded to observed evidence and emphasized limitations, ambiguity, and future work.

## 2026-05-23: Assignment 3 Final Strategy Narrative

Decision: Frame the strongest Assignment 3 strategy as hybrid graph-aware bus entity resolution with seven stages: high-confidence seed generation, graph construction, topology-aware propagation, geographic validation, composite scoring, one-to-one conflict resolution, and ambiguity handling with manual review.

Rationale: This narrative best reflects the assignment's central hint that electrical neighborhood structure can reveal identity even when names differ. It also clarifies the role of each signal: seeds provide reliable anchors, topology becomes the main network-aware refinement signal, geography validates and filters candidates, voltage enforces physical feasibility, and one-to-one conflict handling prevents unrealistic many-to-one collapse.

Alternatives considered:
- Present the solution as primarily geospatial matching: rejected because coordinates are useful but not sufficient and some Dayzer coordinates are missing.
- Present the solution as primarily fuzzy matching: rejected because names vary across systems and do not capture electrical neighborhood consistency.
- Present topology as only another score component: revised because the strongest conceptual strategy makes topology-aware propagation the central differentiator.

Verification:
- Updated `bus_mapping_summary.md` to explicitly state the seven-stage hybrid strategy.
- Added the final narrative explaining seed anchors, matched-neighbor propagation, geography as validation, voltage as feasibility, and one-to-one realism.
- Preserved careful wording that the current implementation computes local topology scores and establishes outputs needed for a fuller iterative seed-and-propagate extension.

## 2026-05-23: Assignment 3 Topology-Aware Propagation Implementation

Decision: Implement the seed-and-propagate graph matching model as a real pipeline stage, not only as report framing.

Rationale: The user clarified that the final strategy should be implemented, not just described. The implemented model starts with strict high-confidence seeds, then iteratively accepts unresolved Dayzer-Panorama pairs when already-matched neighboring buses provide structural support. This directly operationalizes the assignment hint that electrical neighborhoods can reveal identity.

Alternatives considered:
- Leave topology as a static score only: rejected because it does not fully implement the final strategy.
- Accept all top-ranked candidates after propagation scoring: rejected because it weakens one-to-one realism and may preserve duplicate Panorama claims.
- Require every final match to be seed or propagated only: stricter, but too conservative for an exploratory assignment; the final output preserves weaker top-ranked candidates separately as `top_ranked_unpropagated` and flags conflicts/ambiguity for review.

Verification:
- Added `solution/assignment_3/src/propagation.py`.
- Updated `matching.py` to run propagation by default and preserve baseline results separately.
- Generated `bus_mapping_seed_matches.csv`, `bus_mapping_accepted_graph_matches.csv`, `bus_mapping_propagation_log.csv`, `bus_candidate_pairs_propagated.csv`, and `bus_mapping_results_baseline.csv`.
- Confirmed 49 strict seed matches expanded to 1,059 accepted graph matches over 8 propagation iterations.
- Confirmed final output has zero duplicate nonblank Panorama IDs after one-to-one conflict handling.
- Re-executed the Assignment 3 notebook end to end.

## 2026-05-25: Assignment 3 No-Ground-Truth Evaluation Diagnostics

Decision: Extend Assignment 3 with internal consistency diagnostics instead of claiming true accuracy.

Rationale: No labeled ground truth is available for bus matches. The most defensible evaluation is therefore based on operational plausibility: duplicate Panorama assignment rate, voltage compatibility, distance plausibility, topology support, confidence/ambiguity behavior, and manual review sampling. This directly answers whether graph-aware matching is more defensible than pure latitude/longitude nearest matching without overstating accuracy.

Alternatives considered:
- Report only match counts: rejected because higher match count can hide many bad assignments.
- Add heavy graph embeddings, GNNs, or global graph matching: deferred because the current need is evaluation and defensibility, not another complex model.
- Treat pure lat/lon nearest as the default answer: rejected because it produced 7,102 duplicate Panorama assignments and a 42.2% voltage conflict rate above 1 kV.

Verification:
- Added `diagnostics.py`.
- Generated ablation, sensitivity, one-to-one, topology contribution, distance/voltage, extended manual review, and case-study outputs.
- Confirmed `bus_mapping_manual_review_extended.csv` has 300 rows, with 50 rows in each requested bucket.
- Confirmed final graph-aware output still has zero duplicate nonblank Panorama IDs.
- Re-executed the Assignment 3 notebook end to end after adding diagnostics.
