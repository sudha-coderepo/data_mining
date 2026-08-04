"""
Merge validated human labels into data/reviews_human_gold.csv.

Runs validation first; merge only proceeds if validation passes.

Usage:
    python scripts/merge_human_labels.py labeling/gold_labeling_completed.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from split_utils import HUMAN_GOLD_CSV, ensure_dirs, load_gold_indices, load_manifest
from validate_human_labels import validate_human_labels, write_report


def merge_human_labels(completed_path: str, min_rows: int) -> pd.DataFrame:
    manifest = load_manifest()
    gold_indices = load_gold_indices()
    expected_ids = manifest.loc[gold_indices, "review_id"].astype(str).tolist()

    completed = pd.read_csv(completed_path, low_memory=False)
    completed["review_id"] = completed["review_id"].astype(str).str.strip()
    completed = completed.drop_duplicates(subset=["review_id"], keep="last")

    gold = completed[completed["review_id"].isin(expected_ids)].copy()
    gold = gold.sort_values("review_id")

    merged = manifest.loc[gold_indices].merge(
        gold[
            [
                "review_id",
                "human_sentiment",
                "human_category",
                "annotator",
                "notes",
            ]
        ],
        on="review_id",
        how="left",
    )

    ensure_dirs()
    merged.to_csv(HUMAN_GOLD_CSV, index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge validated human gold labels.")
    parser.add_argument("completed_csv", help="Path to completed labeling CSV")
    parser.add_argument(
        "--min-rows",
        type=int,
        default=100,
        help="Minimum complete rows required (default: 100)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Merge even if validation fails (not recommended).",
    )
    args = parser.parse_args()

    result = validate_human_labels(args.completed_csv, min_rows=args.min_rows)
    write_report(result, args.completed_csv)

    if not result.passed and not args.force:
        print("\nMerge aborted. Fix validation errors and re-run.", file=sys.stderr)
        sys.exit(1)

    merged = merge_human_labels(args.completed_csv, args.min_rows)
    print(f"\nMerged {len(merged)} gold reviews → {HUMAN_GOLD_CSV}")
    print(
        "\nHuman gold set is ready. Next (automated when you return):\n"
        "  python scripts/run_post_labeling_pipeline.py"
    )


if __name__ == "__main__":
    main()
