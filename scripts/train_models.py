"""
Train classifiers on train split only (gold excluded).

Sentiment: text + star rating features, target = star-mapped sentiment.
Theme: text features, target = LLM silver labels (Track C / deployment).
Also saves human-target theme models trained on gold for CV comparison script.

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
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import load_sentiment_matrix, load_text_matrix
from split_utils import load_pickle, TRAIN_INDICES_PKL, VAL_INDICES_PKL


def get_models():
    return {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        ),
    }


def train_task(X_train, y_train, X_val, y_val, prefix: str, task_name: str) -> None:
    print(f"\n{'=' * 50}\n TRAINING: {task_name}\n{'=' * 50}")
    for name, model in get_models().items():
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
    X_text = load_text_matrix(ROOT)
    X_sent = load_sentiment_matrix(ROOT)

    train_idx = list(load_pickle(TRAIN_INDICES_PKL))
    val_idx = list(load_pickle(VAL_INDICES_PKL))

    train_task(
        X_sent[train_idx],
        df.loc[train_idx, "sentiment"],
        X_sent[val_idx],
        df.loc[val_idx, "sentiment"],
        "sentiment",
        "Sentiment (target: star-mapped, features: text+rating)",
    )
    train_task(
        X_text[train_idx],
        df.loc[train_idx, "llm_category"],
        X_text[val_idx],
        df.loc[val_idx, "llm_category"],
        "category",
        "Theme (target: LLM silver labels, features: text)",
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
