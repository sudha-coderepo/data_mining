"""
Combine two completed labeling parts into one file for validation/merge.

Usage:
    python scripts/combine_human_labels.py \
        labeling/gold_labeling_completed_part1.csv \
        labeling/gold_labeling_completed_part2.csv \
        --output labeling/gold_labeling_completed.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def combine_parts(paths: list[str], output: str) -> pd.DataFrame:
    frames = [pd.read_csv(p, low_memory=False) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined["review_id"] = combined["review_id"].astype(str).str.strip()

    if combined["review_id"].duplicated().any():
        dupes = combined.loc[combined["review_id"].duplicated(), "review_id"].tolist()
        raise ValueError(f"Duplicate review_id after combine: {dupes[:5]}")

    combined.to_csv(output, index=False)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine split human label files.")
    parser.add_argument("part_files", nargs="+", help="Completed part CSV files")
    parser.add_argument(
        "--output",
        default="labeling/gold_labeling_completed.csv",
        help="Combined output path",
    )
    args = parser.parse_args()

    combined = combine_parts(args.part_files, args.output)
    print(f"Combined {len(combined)} reviews → {args.output}")
    print(
        "\nNext:\n"
        f"  python scripts/validate_human_labels.py {args.output}\n"
        f"  python scripts/merge_human_labels.py {args.output}"
    )


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
