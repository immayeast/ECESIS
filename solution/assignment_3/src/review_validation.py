"""Manual-review validation layer for Assignment 3.

This script consumes the reviewed manual sample and produces validation outputs
without claiming full-population accuracy. Labels are interpreted as:

1  = likely/correct match
0  = incorrect match
-1 = uncertain / excluded from precision denominators
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_COL = "review_label"


def find_review_file(data_dir: Path) -> Path:
    """Find reviewed manual-review file, accepting the user's typo."""
    candidates = [
        data_dir / "bus_mapping_manual_reviewed_extended.csv",
        data_dir / "bus_mapping_manual_reveiwed_extended.csv",
        data_dir / "bus_mapping_manual_review_extended_reviewed.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    recursive_patterns = [
        "*manual*reviewed*extended*.csv",
        "*manual*reveiwed*extended*.csv",
        "*manual*review*extended*reviewed*.csv",
    ]
    search_roots = [data_dir, data_dir.parent, data_dir.parents[2] if len(data_dir.parents) > 2 else data_dir.parent]
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in recursive_patterns:
            found = sorted(root.rglob(pattern))
            if found:
                return found[0]
    raise FileNotFoundError(
        "Could not find reviewed manual file. Expected one of: "
        + ", ".join(path.name for path in candidates)
    )


def clean_labels(review: pd.DataFrame) -> pd.DataFrame:
    """Normalize review labels and add validation flags."""
    if LABEL_COL not in review.columns:
        raise ValueError(f"Reviewed file must contain a `{LABEL_COL}` column.")
    out = review.copy()
    out[LABEL_COL] = pd.to_numeric(out[LABEL_COL], errors="coerce")
    valid_values = {1, 0, -1}
    invalid = out[LABEL_COL].notna() & ~out[LABEL_COL].isin(valid_values)
    if invalid.any():
        bad = sorted(out.loc[invalid, LABEL_COL].dropna().unique())
        raise ValueError(f"Unexpected review labels found: {bad}. Use 1, 0, or -1.")
    out["is_reviewed"] = out[LABEL_COL].isin([1, 0, -1])
    out["is_scored_review"] = out[LABEL_COL].isin([1, 0])
    out["is_likely_correct"] = out[LABEL_COL].eq(1)
    out["is_incorrect"] = out[LABEL_COL].eq(0)
    out["is_uncertain"] = out[LABEL_COL].eq(-1)
    return out


def precision_summary(review: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Compute reviewed precision summary overall or by groups."""
    group_cols = group_cols or []
    data = review.copy()
    if group_cols:
        grouped = data.groupby(group_cols, dropna=False)
    else:
        data["_overall"] = "overall"
        grouped = data.groupby("_overall", dropna=False)
        group_cols = ["_overall"]

    rows = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        scored = group[group["is_scored_review"]]
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(
            {
                "rows_in_sample": int(len(group)),
                "reviewed_rows": int(group["is_reviewed"].sum()),
                "scored_rows_excluding_uncertain": int(len(scored)),
                "likely_correct_count": int(scored["is_likely_correct"].sum()),
                "incorrect_count": int(scored["is_incorrect"].sum()),
                "uncertain_count": int(group["is_uncertain"].sum()),
                "reviewed_precision": float(scored["is_likely_correct"].mean()) if len(scored) else np.nan,
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if "_overall" in out.columns:
        out = out.drop(columns=["_overall"])
    return out


def merge_review_labels(final_results: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    """Attach manual labels to full mapping output where sample rows match."""
    key_cols = ["dayzer_bus"]
    review_cols = [
        "dayzer_bus",
        "graph_aware_pano_bus",
        "review_bucket",
        "review_label",
        "review_notes",
        "is_reviewed",
        "is_scored_review",
        "is_likely_correct",
        "is_incorrect",
        "is_uncertain",
    ]
    available = [col for col in review_cols if col in review.columns]
    sample = review[available].copy()
    if "graph_aware_pano_bus" in sample.columns:
        sample = sample.rename(columns={"graph_aware_pano_bus": "reviewed_graph_aware_pano_bus"})
    sample = sample.drop_duplicates(key_cols, keep="first")
    merged = final_results.merge(sample, on=key_cols, how="left")
    merged["validation_status"] = "not_sampled"
    merged.loc[merged["review_label"].eq(1), "validation_status"] = "review_likely_correct"
    merged.loc[merged["review_label"].eq(0), "validation_status"] = "review_incorrect"
    merged.loc[merged["review_label"].eq(-1), "validation_status"] = "review_uncertain"
    return merged


def write_validation_report(report_path: Path, summaries: dict[str, pd.DataFrame], review_file: Path) -> None:
    """Write a concise reviewed-validation markdown report."""
    overall = summaries["overall"].iloc[0]
    by_bucket = summaries["by_bucket"]
    by_confidence = summaries["by_confidence"]
    by_stage = summaries["by_stage"]

    def table(df: pd.DataFrame) -> str:
        return df.to_markdown(index=False)

    text = f"""# Assignment 3 Manual Review Validation

## Reviewed File

Reviewed sample file: `{review_file.name}`

Manual label convention:

- `1` = likely/correct match
- `0` = incorrect match
- `-1` = uncertain

## Evaluation Scope

This validation estimates quality on the manually reviewed sample only. It does not create full-population ground truth for all Dayzer buses.

## Overall Reviewed Precision

- Rows in sample: {int(overall['rows_in_sample'])}
- Reviewed rows: {int(overall['reviewed_rows'])}
- Scored rows excluding uncertain: {int(overall['scored_rows_excluding_uncertain'])}
- Likely/correct labels: {int(overall['likely_correct_count'])}
- Incorrect labels: {int(overall['incorrect_count'])}
- Uncertain labels: {int(overall['uncertain_count'])}
- Reviewed precision: {overall['reviewed_precision']:.3f}

## Precision by Review Bucket

{table(by_bucket)}

## Precision by Confidence Label

{table(by_confidence)}

## Precision by Match Stage

{table(by_stage)}

## Interpretation

The reviewed sample converts Assignment 3 from a purely unsupervised/internal-consistency evaluation into a manually checked validation sample. Precision should still be interpreted as sample-based, because the reviewed rows were intentionally stratified across confidence and risk buckets rather than randomly sampled from the entire population.
"""
    report_path.write_text(text)


def run_review_validation(
    repo_root: Path,
    data_dir_override: Path | None = None,
    review_file_override: Path | None = None,
) -> dict[str, pd.DataFrame]:
    assignment_dir = repo_root / "solution" / "assignment_3"
    data_dir = data_dir_override or assignment_dir / "execution_outputs"
    if not data_dir.exists():
        data_dir = assignment_dir / "outputs"
    output_dir = assignment_dir / "execution_outputs"
    solution_dir = assignment_dir / "solutions"
    report_dir = assignment_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    solution_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    review_file = review_file_override or find_review_file(data_dir)
    review = clean_labels(pd.read_csv(review_file))
    final_results_path = data_dir / "bus_mapping_results.csv"
    if not final_results_path.exists():
        raise FileNotFoundError(f"Could not find final mapping file: {final_results_path}")
    final_results = pd.read_csv(final_results_path)

    summaries = {
        "overall": precision_summary(review),
        "by_bucket": precision_summary(review, ["review_bucket"]),
        "by_confidence": precision_summary(review, ["confidence_label"]),
        "by_stage": precision_summary(review, ["match_stage"]) if "match_stage" in review.columns else pd.DataFrame(),
    }
    if "accepted_graph_match" in review.columns:
        summaries["by_graph_acceptance"] = precision_summary(review, ["accepted_graph_match"])

    validated_rows = review.copy()
    full_validated = merge_review_labels(final_results, review)

    outputs = {
        "manual_review_validation_overall.csv": summaries["overall"],
        "manual_review_validation_by_bucket.csv": summaries["by_bucket"],
        "manual_review_validation_by_confidence.csv": summaries["by_confidence"],
        "manual_review_validation_by_stage.csv": summaries["by_stage"],
        "manual_review_validated_rows.csv": validated_rows,
    }
    if "by_graph_acceptance" in summaries:
        outputs["manual_review_validation_by_graph_acceptance.csv"] = summaries["by_graph_acceptance"]

    for name, df in outputs.items():
        df.to_csv(output_dir / name, index=False)

    solution_path = solution_dir / "bus_mapping_results_review_validated.csv"
    full_validated.to_csv(solution_path, index=False)
    write_validation_report(report_dir / "manual_review_validation_summary.md", summaries, review_file)

    return {**outputs, "bus_mapping_results_review_validated.csv": full_validated}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Assignment 3 manual review validation.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing bus_mapping_results.csv and reviewed manual CSV.")
    parser.add_argument("--review-file", type=Path, default=None, help="Explicit path to reviewed manual CSV.")
    args = parser.parse_args()
    outputs = run_review_validation(args.repo_root, data_dir_override=args.data_dir, review_file_override=args.review_file)
    for name, df in outputs.items():
        print(f"{name}: {df.shape[0]:,} rows x {df.shape[1]:,} columns")


if __name__ == "__main__":
    main()
