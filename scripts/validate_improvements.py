"""
Compare metrics before vs after Phase 1 improvements.

Usage:
    python scripts/validate_improvements.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "outputs" / "metrics"

# Pre-improvement Track A baselines (from Report §5.2)
BEFORE = {
    ("Sentiment", "Logistic Regression"): {"Accuracy": 0.74, "Macro_F1": 0.5903},
    ("Sentiment", "Random Forest"): {"Accuracy": 0.72, "Macro_F1": 0.4484},
    ("Theme", "Logistic Regression"): {"Accuracy": 0.59, "Macro_F1": 0.2977},
    ("Theme", "LLM (Zero-Shot)"): {"Accuracy": 0.56, "Macro_F1": 0.4373},
}


def main() -> None:
    sent = pd.read_csv(METRICS / "track_a_sentiment_gold.csv")
    cat = pd.read_csv(METRICS / "track_a_category_gold.csv")
    cv = pd.read_csv(METRICS / "cv_human_gold.csv")

    rows = []
    for task, df in [("Sentiment", sent), ("Theme", cat)]:
        for _, r in df.iterrows():
            model = r["Model"]
            key = (task, model)
            before = BEFORE.get(key, {})
            rows.append(
                {
                    "Task": task,
                    "Model": model,
                    "Before_Acc": before.get("Accuracy"),
                    "After_Acc": r["Accuracy"],
                    "Before_Macro_F1": before.get("Macro_F1"),
                    "After_Macro_F1": r["Macro_F1"],
                    "Acc_Delta": r["Accuracy"] - before.get("Accuracy", r["Accuracy"]) if before else None,
                    "F1_Delta": r["Macro_F1"] - before.get("Macro_F1", r["Macro_F1"]) if before else None,
                }
            )

    report = pd.DataFrame(rows)
    out = METRICS / "improvement_validation.csv"
    report.to_csv(out, index=False)

    print("=" * 60)
    print(" IMPROVEMENT VALIDATION — Track A (Human Gold)")
    print("=" * 60)
    show = report[report["Before_Macro_F1"].notna()][
        ["Task", "Model", "Before_Macro_F1", "After_Macro_F1", "F1_Delta"]
    ]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n5-Fold CV (human-label training target, honest estimate):")
    print(cv[["Task", "Model", "Macro_F1_mean", "Macro_F1_std"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    lr_sent = sent.loc[sent["Model"] == "Logistic Regression (Tuned)", "Macro_F1"].iloc[0]
    before_lr = BEFORE[("Sentiment", "Logistic Regression")]["Macro_F1"]
    print(f"\nSentiment LR (Tuned) Macro F1: {before_lr:.2%} → {lr_sent:.2%} (+{(lr_sent - before_lr) * 100:.1f} pts)")
    print(f"Saved: {out}")

    if lr_sent <= before_lr:
        sys.exit(1)


if __name__ == "__main__":
    main()
