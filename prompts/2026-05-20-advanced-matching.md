# Advanced Matching Prompt - 2026-05-20

## User Prompt Summary

Implement the advanced Assignment 1 matching plan as a new notebook with PJMISO/Market as the reference dataset, top-k PANO and DAYZER candidates, multi-signal scoring, strategic review queues, CSV manual labels, optional classifier refinement, refined outputs, and validation checks.

Add a 200-300 row manual-label sample for immediate hand grading and keep Random Forest, XGBoost, and LightGBM as possible follow-up model comparisons after Logistic Regression.

Accept source-specific alternative rule scoring while keeping equal-sized manual-label selection buckets.

## High-Impact AI Output Summary

- Created the advanced matching and weak-supervision notebook.
- Generated candidate-pair, review-queue, manual-label, prediction, refined-match, refined-support, and validation CSV outputs.
- Generated a 300-row manual-label random selection with high, medium, and low/ambiguous buckets.
- Added `rule_score_v1` for the original shared formula and `rule_score_v2` for source-specific ranking.
- Updated project documentation and inspection notes to explain the broader review strategy.
