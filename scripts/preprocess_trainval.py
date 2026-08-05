"""
Preprocess text and fit word+char TF-IDF on train split only (no gold leakage).
Builds separate feature matrices for theme (text) and sentiment (text + rating).

Usage:
    python scripts/preprocess_trainval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import fit_transform_text, save_features, stack_sentiment_features
from split_utils import load_gold_indices, load_labeled_reviews, load_pickle, save_pickle, TRAIN_INDICES_PKL

PREPROCESSED_CSV = ROOT / "preprocessed_reviews.csv"


def main() -> None:
    if not TRAIN_INDICES_PKL.exists():
        raise FileNotFoundError("Missing splits. Run: python scripts/create_splits.py")

    df = load_labeled_reviews()
    train_idx = load_pickle(TRAIN_INDICES_PKL)
    gold_idx = set(load_gold_indices())

    vectorizer, text_matrix, combined = fit_transform_text(df, train_idx)
    df["cleaned_text"] = combined
    ratings = df["reviews.rating"].astype(float).values
    sentiment_matrix = stack_sentiment_features(text_matrix, ratings)

    df.to_csv(PREPROCESSED_CSV, index=False)
    save_features(ROOT, vectorizer, text_matrix, sentiment_matrix)

    print(f"Rows: {len(df)} | Train fit: {len(train_idx)} | Gold held out: {len(gold_idx)}")
    print(f"Text TF-IDF shape: {text_matrix.shape}")
    print(f"Sentiment feature shape: {sentiment_matrix.shape}")
    print("Saved: feature_vectorizer.pkl, tfidf_matrix.pkl, sentiment_feature_matrix.pkl")


if __name__ == "__main__":
    main()
