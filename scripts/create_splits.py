"""
Create train/val/gold splits and export the human labeling template.

Usage:
    python scripts/create_splits.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from split_utils import (
    DELIVERY_KEYWORDS,
    GOLD_CATEGORY_TARGETS,
    GOLD_INDICES_PKL,
    GOLD_SIZE,
    GOLD_TEMPLATE_CSV,
    GOLD_TEMPLATE_XLSX,
    MANIFEST_CSV,
    RANDOM_STATE,
    SPLIT_SUMMARY_JSON,
    TRAIN_INDICES_PKL,
    VAL_FRACTION,
    VAL_INDICES_PKL,
    ensure_dirs,
    load_labeled_reviews,
    save_pickle,
    write_split_summary,
)


def _text_has_delivery_signal(text: str) -> bool:
    lowered = str(text).lower()
    return any(keyword in lowered for keyword in DELIVERY_KEYWORDS)


def sample_gold_indices(df: pd.DataFrame, gold_size: int = GOLD_SIZE) -> list[int]:
    """Stratified gold holdout; fills Delivery from keyword matches when LLM has none."""
    rng = np.random.default_rng(RANDOM_STATE)
    selected: list[int] = []
    selected_set: set[int] = set()

    def take_from_pool(pool: pd.DataFrame, n: int) -> None:
        if n <= 0 or pool.empty:
            return
        n = min(n, len(pool))
        picks = pool.sample(n=n, random_state=RANDOM_STATE).index.tolist()
        for idx in picks:
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)

    # Priority: sentiment disagreements (informative for validation)
    disagree = df[df["sentiment"] != df["llm_sentiment"]]
    take_from_pool(disagree, min(26, len(disagree)))

    # Category-stratified picks from LLM labels
    for category, target in GOLD_CATEGORY_TARGETS.items():
        if category == "Delivery issue":
            delivery_pool = df[
                (df["llm_category"] == "Delivery issue")
                | (
                    (df["llm_category"] == "Other")
                    & df["reviews.text"].map(_text_has_delivery_signal)
                )
            ]
            take_from_pool(delivery_pool[~delivery_pool.index.isin(selected_set)], target)
        else:
            pool = df[df["llm_category"] == category]
            take_from_pool(pool[~pool.index.isin(selected_set)], target)

    # Fill to gold_size from remaining rows, preferring disagreements then diversity
    remaining = df[~df.index.isin(selected_set)]
    if len(selected) < gold_size and not remaining.empty:
        need = gold_size - len(selected)
        # Prefer more disagreement cases first
        remaining = remaining.assign(
            _priority=(remaining["sentiment"] != remaining["llm_sentiment"]).astype(int)
        ).sort_values("_priority", ascending=False)
        fill = remaining.head(need).index.tolist()
        selected.extend(fill)
        selected_set.update(fill)

    if len(selected) < gold_size:
        print(
            f"Warning: only sampled {len(selected)} gold reviews (target {gold_size}).",
            file=sys.stderr,
        )

    rng.shuffle(selected)
    return selected[:gold_size]


def export_labeling_template(df: pd.DataFrame, gold_indices: list[int]) -> None:
    gold = df.loc[gold_indices].copy()
    template = pd.DataFrame(
        {
            "review_id": gold["review_id"],
            "reviews.title": gold["reviews.title"],
            "reviews.text": gold["reviews.text"],
            "reviews.rating": gold["reviews.rating"],
            "sentiment_rating": gold["sentiment"],
            "human_sentiment": "",
            "human_category": "",
            "annotator": "",
            "notes": "",
        }
    )
    template.to_csv(GOLD_TEMPLATE_CSV, index=False, encoding="utf-8")

    try:
        template.to_excel(GOLD_TEMPLATE_XLSX, index=False, engine="openpyxl")
        xlsx_msg = str(GOLD_TEMPLATE_XLSX)
    except ImportError:
        xlsx_msg = "(skipped — pip install openpyxl for .xlsx export)"

    print(f"Exported CSV template:  {GOLD_TEMPLATE_CSV}")
    print(f"Exported Excel template: {xlsx_msg}")
    print(
        "\nLabeling instructions: fill human_sentiment and human_category only.\n"
        "Allowed sentiment: Positive, Neutral, Negative\n"
        "Allowed category: Delivery issue, Product quality issue, Price complaint,\n"
        "                  Customer service issue, Feature request, Other\n"
        "When done, run: python scripts/validate_human_labels.py labeling/gold_labeling_completed.csv"
    )


def main() -> None:
    ensure_dirs()
    df = load_labeled_reviews()
    print(f"Loaded {len(df)} reviews with stable review_id (R00000–R{len(df)-1:05d}).")

    gold_indices = sample_gold_indices(df, GOLD_SIZE)
    remaining = df.index.difference(gold_indices)

    train_indices, val_indices = train_test_split(
        remaining,
        test_size=VAL_FRACTION,
        random_state=RANDOM_STATE,
        stratify=df.loc[remaining, "llm_category"],
    )

    manifest = df.copy()
    manifest["split"] = "train"
    manifest.loc[val_indices, "split"] = "val"
    manifest.loc[gold_indices, "split"] = "gold"
    manifest.to_csv(MANIFEST_CSV, index=False)

    save_pickle(GOLD_INDICES_PKL, list(gold_indices))
    save_pickle(TRAIN_INDICES_PKL, list(train_indices))
    save_pickle(VAL_INDICES_PKL, list(val_indices))

    summary = {
        "total_reviews": int(len(df)),
        "gold_size": len(gold_indices),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "gold_llm_category_counts": df.loc[gold_indices, "llm_category"]
        .value_counts()
        .to_dict(),
        "gold_sentiment_disagreement_count": int(
            (df.loc[gold_indices, "sentiment"] != df.loc[gold_indices, "llm_sentiment"]).sum()
        ),
        "random_state": RANDOM_STATE,
    }
    write_split_summary(summary)
    export_labeling_template(df, gold_indices)

    print(f"\nSaved manifest:       {MANIFEST_CSV}")
    print(f"Saved gold indices:   {GOLD_INDICES_PKL} ({len(gold_indices)} reviews)")
    print(f"Saved train indices:  {TRAIN_INDICES_PKL} ({len(train_indices)} reviews)")
    print(f"Saved val indices:    {VAL_INDICES_PKL} ({len(val_indices)} reviews)")
    print(f"Saved split summary:  {SPLIT_SUMMARY_JSON}")
    print("\ncreate_splits.py completed successfully.")


if __name__ == "__main__":
    main()
