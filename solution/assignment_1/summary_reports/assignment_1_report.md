# Assignment 1 Report: Constraint Matching

## Objective

Assignment 1 matches PJMISO market constraints to related records in the DAYZER and PANO datasets. PJMISO/Market is chosen as the reference dataset because it contains the richest descriptive fields, including constraint, contingency, reported name, zone, and ID fields. 

## Data Inspection

The three datasets have different levels of detail:

- DAYZER contains `CID` and `NAME`, so matching depends mostly on facility and interface terms. The Dataset also requires string splitting.
- PJMISO contains `CONSTRAINT`, `CONTINGENCY`, `REPORTEDNAME`, `TOZONE`, `CONSTRAINTID`, and `CONTINGENCYID`. Richest dataset and can be compared to the other two datasets.
- PANO contains monitored facility, contingency name, and date coverage fields.

The main missing-value issue is in PJMISO `TOZONE`:

| dataset | column | row_count | missing_count | missing_percent | blank_string_count |
|---|---|---:|---:|---:|---:|
| market | TOZONE | 5,230 | 1,027 | 19.64 | 0 |

Because 19.64% of PJMISO `TOZONE` is missing, zone/token overlap is treated as a weak supporting feature rather than a primary matching signal. The meaning of missing value is also not taken as zero but Missing Completely At Random (MCAR) in this case.

## Matching Workflow

The matching pipeline is PJMISO-centered:

1. Normalize raw text by lowercasing, removing punctuation, replacing separators with spaces, and standardizing common terms.
2. Extract facility tokens, contingency tokens, voltage level, zone/utility terms, and interface keywords.
3. Generate top-k PANO and DAYZER candidate pairs for each PJMISO record.
4. Score candidate pairs using multiple signals.
5. Use manual review labels to train a supervised refinement model.
6. Re-rank PANO and DAYZER candidates by supervised match probability.
7. Leave a source unmatched when the best probability is below 0.50.

The final match file is stored at:

`solution/assignment_1/solutions/constraint_matches_refined.csv`

## Supervised Refinement

Manual review labels use this convention:

- `1` = match
- `0` = unmatch / not a valid match
- `-1` = uncertain / cannot determine (although there only appeared to be one where I manually reviewed and yet was clueless about the match)

Rows labeled `-1` were excluded from training. The supervised model uses engineered matching features such as weighted score, facility score, contingency score, voltage match, zone/token overlap, ambiguity flag, confidence encoding, and score gap to the second-best candidate.

Logistic Regression was used as the supervised refinement model because it is lightweight and interpretable. Random Forest remains available as a fallback if Logistic Regression fails due to missing or constant features.

## Diagnosis Evaluation

The reviewed label file contains 300 rows:

- 299 rows used for training after excluding uncertain labels.
- 275 positive labels.
- 24 negative labels.
- 1 uncertain label excluded.

Cross-validation diagnostics:

| metric | value |
|---|---:|
| accuracy | 0.766 |
| precision | 0.968 |
| recall | 0.771 |
| F1 | 0.858 |

The supervised refinement stage prioritized precision over recall to minimize incorrect constraint mappings. High-confidence matches achieved strong precision (0.968), while uncertain or weakly supported matches were conservatively rejected rather than force-matched.

The current evaluation is label-imbalanced because the reviewed labels contain many more matches than non-matches. A future improvement is to label more `not_match` examples so the model can be evaluated on a more balanced decision boundary. If I have more time I would either reselect more data and rerun this manual review on a larger data selection, or I would exclude the already-reviewed data of this 300 selections and draw more from the dataset.

## Final Outputs

The final supervised match output contains 5,230 PJMISO rows:

- `market_constraint`
- `dayzer_constraint`
- `pano_constraint`

The supervised support output contains 10,460 PJMISO/source rows and preserves audit fields:

- original weighted score
- match probability
- refined confidence label
- facility score
- contingency score
- voltage match
- zone/token overlap
- ambiguity flag
- manual label when available

Validation summary:

| check | result |
|---|---:|
| reviewed rows | 300 |
| labeled rows used | 299 |
| supervised match rows | 5,230 |
| supervised support rows | 10,460 |
| every PJMISO row present | true |
| expected support rows present | true |
| accepted matches below 0.50 | 0 |

## Limitations and Next Steps

The classifier diagnostics should be interpreted as preliminary because the training labels are imbalanced. The next review pass should intentionally add more `not_match` examples, especially from low-confidence and ambiguous rows.