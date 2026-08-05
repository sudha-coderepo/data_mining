"""
5-fold cross-validation on human gold labels (Phase 1.1).

Trains on human_sentiment / human_category within each fold — no leakage across folds.

Usage:
    python scripts/train_human_gold_cv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import load_sentiment_matrix, load_text_matrix
from split_utils import HUMAN_GOLD_CSV, OUTPUTS_DIR


def cv_task(X, y, task_name: str, n_splits: int = 5) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []
    for model_name, model in [
        ("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
    ]:
        fold_metrics = []
        for train_i, test_i in skf.split(X, y):
            m = clone(model)
            m.fit(X[train_i], y[train_i])
            pred = m.predict(X[test_i])
            p, r, f1, _ = precision_recall_fscore_support(
                y[test_i], pred, average="macro", zero_division=0
            )
            fold_metrics.append(
                {
                    "accuracy": accuracy_score(y[test_i], pred),
                    "macro_f1": f1,
                }
            )
        rows.append(
            {
                "Task": task_name,
                "Model": model_name,
                "Accuracy_mean": np.mean([m["accuracy"] for m in fold_metrics]),
                "Accuracy_std": np.std([m["accuracy"] for m in fold_metrics]),
                "Macro_F1_mean": np.mean([m["macro_f1"] for m in fold_metrics]),
                "Macro_F1_std": np.std([m["macro_f1"] for m in fold_metrics]),
                "Folds": n_splits,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if not HUMAN_GOLD_CSV.exists():
        raise FileNotFoundError(f"Missing {HUMAN_GOLD_CSV}")

    df = pd.read_csv(ROOT / "preprocessed_reviews.csv", low_memory=False)
    gold_df = pd.read_csv(HUMAN_GOLD_CSV, low_memory=False)
    X_text = load_text_matrix(ROOT)
    X_sent = load_sentiment_matrix(ROOT)

    rid_to_idx = {str(r): i for i, r in enumerate(df["review_id"].astype(str))}
    gold_idx = np.array([rid_to_idx[str(r)] for r in gold_df["review_id"].astype(str)])

    y_sent = gold_df["human_sentiment"].values
    y_cat = gold_df["human_category"].values

    sent_cv = cv_task(X_sent[gold_idx], y_sent, "Sentiment (human labels)")
    cat_cv = cv_task(X_text[gold_idx], y_cat, "Theme (human labels)")

    out = pd.concat([sent_cv, cat_cv], ignore_index=True)
    metrics_dir = OUTPUTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "cv_human_gold.csv"
    out.to_csv(out_path, index=False)

    print("\n5-Fold CV on Human Gold (100 reviews)")
    print(out.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
