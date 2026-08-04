"""
Preprocess text and fit TF-IDF on train split only (no gold leakage).

Usage:
    python scripts/preprocess_trainval.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from split_utils import (
    LABELED_CSV,
    load_gold_indices,
    load_labeled_reviews,
    load_pickle,
    save_pickle,
    TRAIN_INDICES_PKL,
)
from text_utils import clean_review_text

PREPROCESSED_CSV = ROOT / "preprocessed_reviews.csv"
VECTORIZER_PKL = ROOT / "tfidf_vectorizer.pkl"
MATRIX_PKL = ROOT / "tfidf_matrix.pkl"


def main() -> None:
    if not TRAIN_INDICES_PKL.exists():
        raise FileNotFoundError("Missing splits. Run: python scripts/create_splits.py")

    df = load_labeled_reviews()
    train_idx = load_pickle(TRAIN_INDICES_PKL)
    gold_idx = set(load_gold_indices())

    df["cleaned_text"] = df["reviews.text"].apply(clean_review_text)

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=5000)
    vectorizer.fit(df.loc[train_idx, "cleaned_text"])
    tfidf_matrix = vectorizer.transform(df["cleaned_text"])

    df.to_csv(PREPROCESSED_CSV, index=False)
    save_pickle(VECTORIZER_PKL, vectorizer)
    save_pickle(MATRIX_PKL, tfidf_matrix)

    print(f"Rows: {len(df)} | Train fit: {len(train_idx)} | Gold held out: {len(gold_idx)}")
    print(f"TF-IDF shape: {tfidf_matrix.shape}")
    print(f"Saved: {PREPROCESSED_CSV.name}, {VECTORIZER_PKL.name}, {MATRIX_PKL.name}")


if __name__ == "__main__":
    main()
