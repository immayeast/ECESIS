# Supervised Refinement Prompt - 2026-05-22

## User Prompt Summary

Add a human-in-the-loop supervised refinement stage after manual review labels. Use the reviewed label file, exclude uncertain labels, train Logistic Regression with Random Forest fallback, predict match probabilities, re-rank PANO and DAYZER candidates, write separate supervised refined outputs, and produce diagnostics.

Draft an Assignment 1 markdown report with results, diagnosis evaluation, current label-imbalance caveat, and final match output placement in the Assignment 1 solutions folder.

Create an Assignment 2 EDA notebook for bus/zone load data structure, missingness, seasonality, plots, and leakage-safe time-series split cautions.

Run the Assignment 2 EDA after the Parquet data was downloaded locally, while avoiding full in-memory bus-data loads.

Expand Assignment 2 EDA to inspect each individual bus and zone file separately because yearly bus and zone files represent different information and granularity.

## High-Impact AI Output Summary

- Added supervised refinement cells to the advanced matching notebook.
- Trained Logistic Regression from reviewed labels.
- Generated supervised candidate predictions, supervised refined matches, supervised support rows, diagnostics, validation summary, and logistic coefficients.
- Preserved the original rule-based matching outputs.
- Added the Assignment 1 report and copied final supervised matches into `solution/assignment_1/solutions/`.
- Added the Assignment 2 EDA/data-structure notebook and recorded the no-random-shuffle, 2025-holdout policy.
- Updated the EDA notebook for large Parquet data and generated Assignment 2 EDA tables and plots.
- Added per-file Assignment 2 EDA outputs for individual file profiles, missingness, and example values.
