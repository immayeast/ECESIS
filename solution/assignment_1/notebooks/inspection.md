# Dataset Inspection

## DAYZER

Columns contain only:

- `CID`
- `NAME`

Issue: abbreviated and simplified names. To compare this dataset, it might be easier to find one dataset to compare first, then use that comparison to approach another dataset.

## PJMISO

Columns:

- `CONSTRAINT`
- `CONTINGENCY`
- `REPORTEDNAME`
- `TOZONE`

Pros: most descriptive data. It might need standardization.

PJMISO contains missing values for `TOZONE`.

| dataset | column | row_count | missing_count | missing_percent | blank_string_count |
|---|---|---:|---:|---:|---:|
| market | TOZONE | 5230 | 1027 | 19.64 | 0 |

Since 19.64% of PJMISO `TOZONE` is missing, zone/token overlap should be a weak feature only.

## PANO

Columns:

- `MONITORED FACILITY`
- `CONTINGENCY NAME`

Pros: also descriptive data.

Cons: formatted differently than PJMISO.

PANO contains much more data than PJMISO, meaning many PANO records may be unmatched.

## Facility and Constraints

Constraints are policies or rules to limit or guide the power grid.

Contingency would be the possible events that would likely trigger the contingencies documented.

Constraints are not always active.

Because PANO and DAYZER contain much more data than PJMISO, many records in those datasets will not have a corresponding PJMISO match.

## Advanced Review Policy

The original 60-row manual review sample is useful, but it is not sufficient by itself because many matching mistakes are most likely to appear in low-confidence or ambiguous rows.

The advanced matching workflow creates a broader review queue that includes:

- selected matches with ambiguity flags
- selected matches below the 0.75 review threshold
- selected matches where the best and second-best scores are within 0.05
- a random audit sample of high-confidence matches

For immediate hand grading, use `manual_labels_random_selection.csv`, which samples 100 high-confidence, 100 medium-confidence, and 100 low/ambiguous candidate pairs.

Manual labels should be copied into `manual_labels.csv` using `match`, `not_match`, or `unsure`. Once there are at least 100 `match` and 100 `not_match` labels, the weak-supervision notebook can train a logistic regression classifier to refine match probabilities. Until then, the workflow falls back to the transparent rule score.

If logistic regression metrics are weak after reliable labels exist, compare Random Forest next, then XGBoost or LightGBM if those dependencies are available.

The advanced notebook keeps the original shared weighted score as `rule_score_v1` and uses the accepted source-specific score as `rule_score_v2` for ranking:

- PANO: 45% facility, 30% contingency, 15% voltage, 5% token overlap, 5% zone overlap.
- DAYZER: 65% facility, 15% voltage, 10% token overlap, 10% interface keyword match.

The 100/100/100 manual-label selection remains balanced across high-confidence, medium-confidence, and low/ambiguous buckets.
