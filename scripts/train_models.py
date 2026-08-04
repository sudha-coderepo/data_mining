"""
Train classifiers on train split only (gold excluded).

Usage:
    python scripts/train_models.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from split_utils import load_pickle, TRAIN_INDICES_PKL, VAL_INDICES_PKL


def train_task(X_train, y_train, X_val, y_val, prefix: str, task_name: str) -> None:
    print(f"\n{'=' * 50}\n TRAINING: {task_name}\n{'=' * 50}")
    models = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        ),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        print(f"\n--- {name.replace('_', ' ').title()} (validation) ---")
        print(classification_report(y_val, y_pred, zero_division=0))
        path = ROOT / f"{prefix}_{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved: {path.name}")


def main() -> None:
    df = pd.read_csv(ROOT / "preprocessed_reviews.csv", low_memory=False)
    with open(ROOT / "tfidf_matrix.pkl", "rb") as f:
        X = pickle.load(f)

    train_idx = list(load_pickle(TRAIN_INDICES_PKL))
    val_idx = list(load_pickle(VAL_INDICES_PKL))

    X_train, X_val = X[train_idx], X[val_idx]

    train_task(
        X_train, df.loc[train_idx, "sentiment"], X_val, df.loc[val_idx, "sentiment"],
        "sentiment", "Sentiment (target: star-mapped sentiment)",
    )
    train_task(
        X_train, df.loc[train_idx, "llm_category"], X_val, df.loc[val_idx, "llm_category"],
        "category", "Theme (target: LLM silver labels)",
    )
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
