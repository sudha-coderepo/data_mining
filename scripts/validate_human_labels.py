"""
Validate a completed human labeling file before merge.

Usage:
    python scripts/validate_human_labels.py labeling/gold_labeling_completed.csv
    python scripts/validate_human_labels.py path/to/your_labels.csv --min-rows 100

Exit code 0 = passed (safe to merge)
Exit code 1 = failed (fix issues and re-run)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import pandas as pd

from split_utils import (
    GOLD_SIZE,
    HUMAN_GOLD_CSV,
    OUTPUTS_DIR,
    VALIDATION_REPORT,
    VALID_CATEGORIES,
    VALID_SENTIMENTS,
    ensure_dirs,
    load_gold_indices,
    load_manifest,
)


@dataclass
class ValidationResult:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.passed = False
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def _normalize_label(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def validate_human_labels(
    completed_path: str,
    min_rows: int = GOLD_SIZE,
    strict_gold_ids: bool = True,
) -> ValidationResult:
    result = ValidationResult()
    ensure_dirs()

    try:
        completed = pd.read_csv(completed_path, low_memory=False)
    except FileNotFoundError:
        result.fail(f"File not found: {completed_path}")
        return result

    required_columns = {"review_id", "human_sentiment", "human_category"}
    missing = required_columns - set(completed.columns)
    if missing:
        result.fail(f"Missing required columns: {sorted(missing)}")
        return result

    completed["review_id"] = completed["review_id"].astype(str).str.strip()
    completed["human_sentiment"] = completed["human_sentiment"].map(_normalize_label)
    completed["human_category"] = completed["human_category"].map(_normalize_label)

    if completed["review_id"].duplicated().any():
        dupes = completed.loc[completed["review_id"].duplicated(), "review_id"].tolist()
        result.fail(f"Duplicate review_id values: {dupes[:5]}")

    manifest = load_manifest()
    manifest_ids = set(manifest["review_id"].astype(str))
    unknown_ids = set(completed["review_id"]) - manifest_ids
    if unknown_ids:
        result.fail(
            f"{len(unknown_ids)} review_id(s) not in manifest (e.g. {sorted(unknown_ids)[:3]})"
        )

    if strict_gold_ids:
        gold_indices = load_gold_indices()
        expected_ids = set(manifest.loc[gold_indices, "review_id"].astype(str))
        submitted_ids = set(completed["review_id"])
        extra = submitted_ids - expected_ids
        missing_gold = expected_ids - submitted_ids
        if extra:
            result.warn(
                f"{len(extra)} review_id(s) not in gold holdout (will ignore extras on merge)."
            )
        if missing_gold:
            result.fail(
                f"Missing {len(missing_gold)} gold review_id(s). "
                f"Examples: {sorted(missing_gold)[:5]}"
            )

    labeled = completed[
        (completed["human_sentiment"] != "") & (completed["human_category"] != "")
    ]
    blank_sent = int((completed["human_sentiment"] == "").sum())
    blank_cat = int((completed["human_category"] == "").sum())
    if blank_sent or blank_cat:
        result.fail(
            f"Incomplete labels: {blank_sent} blank human_sentiment, "
            f"{blank_cat} blank human_category."
        )

    if len(labeled) < min_rows:
        result.fail(
            f"Only {len(labeled)} complete rows; need at least {min_rows}."
        )

    bad_sent = set(labeled["human_sentiment"]) - VALID_SENTIMENTS
    if bad_sent:
        result.fail(f"Invalid human_sentiment values: {sorted(bad_sent)}")

    bad_cat = set(labeled["human_category"]) - VALID_CATEGORIES
    if bad_cat:
        result.fail(f"Invalid human_category values: {sorted(bad_cat)}")

    # Informational comparisons (not pass/fail)
    merged = labeled.merge(
        manifest[["review_id", "sentiment", "llm_sentiment", "llm_category"]],
        on="review_id",
        how="left",
    )
    agree_stars = (merged["human_sentiment"] == merged["sentiment"]).mean()
    agree_llm_sent = (merged["human_sentiment"] == merged["llm_sentiment"]).mean()
    agree_llm_cat = (merged["human_category"] == merged["llm_category"]).mean()

    result.stats = {
        "submitted_rows": int(len(completed)),
        "complete_rows": int(len(labeled)),
        "human_sentiment_counts": labeled["human_sentiment"].value_counts().to_dict(),
        "human_category_counts": labeled["human_category"].value_counts().to_dict(),
        "agreement_human_vs_star_sentiment": round(float(agree_stars), 4),
        "agreement_human_vs_llm_sentiment": round(float(agree_llm_sent), 4),
        "agreement_human_vs_llm_category": round(float(agree_llm_cat), 4),
    }

    rare_cats = {"Delivery issue", "Customer service issue", "Price complaint"}
    for cat in rare_cats:
        if cat not in labeled["human_category"].values:
            result.warn(f"No human labels for rare category: {cat}")

    return result


def write_report(result: ValidationResult, completed_path: str) -> None:
    lines = [
        "Human Label Validation Report",
        "==============================",
        f"Input file: {completed_path}",
        f"Status: {'PASSED' if result.passed else 'FAILED'}",
        "",
    ]
    if result.stats:
        lines.append("Statistics:")
        for key, value in result.stats.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
    if result.errors:
        lines.append("Errors (must fix):")
        for err in result.errors:
            lines.append(f"  - {err}")
        lines.append("")
    if result.warnings:
        lines.append("Warnings:")
        for warn in result.warnings:
            lines.append(f"  - {warn}")
        lines.append("")
    if result.passed:
        lines.append("Next step:")
        lines.append(f"  python scripts/merge_human_labels.py {completed_path}")

    report_text = "\n".join(lines)
    VALIDATION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_REPORT.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\nReport saved: {VALIDATION_REPORT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate completed human labels.")
    parser.add_argument("completed_csv", help="Path to completed labeling CSV")
    parser.add_argument(
        "--min-rows",
        type=int,
        default=GOLD_SIZE,
        help=f"Minimum complete rows required (default: {GOLD_SIZE})",
    )
    parser.add_argument(
        "--allow-extra-ids",
        action="store_true",
        help="Do not fail if gold holdout IDs are missing (not recommended).",
    )
    args = parser.parse_args()

    result = validate_human_labels(
        args.completed_csv,
        min_rows=args.min_rows,
        strict_gold_ids=not args.allow_extra_ids,
    )
    write_report(result, args.completed_csv)
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
