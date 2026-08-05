"""Feature engineering: word+char TF-IDF, title+body text, star rating."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from text_utils import clean_review_text

FEATURE_VECTORIZER_PKL = "feature_vectorizer.pkl"
TEXT_MATRIX_PKL = "tfidf_matrix.pkl"
SENTIMENT_MATRIX_PKL = "sentiment_feature_matrix.pkl"


def build_combined_text(df: pd.DataFrame) -> pd.Series:
    title = df.get("reviews.title", pd.Series("", index=df.index)).fillna("").astype(str)
    body = df["reviews.text"].fillna("").astype(str)
    return (title + " " + body).apply(clean_review_text)


def build_text_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=8000,
                    min_df=2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                    max_features=5000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def rating_matrix(ratings: np.ndarray) -> csr_matrix:
    norm = (ratings.astype(float) - 1.0) / 4.0
    return csr_matrix(norm.reshape(-1, 1))


def stack_sentiment_features(text_matrix, ratings: np.ndarray):
    return hstack([text_matrix, rating_matrix(ratings)])


def fit_transform_text(df: pd.DataFrame, train_idx: list[int]):
    combined = build_combined_text(df)
    vectorizer = build_text_vectorizer()
    vectorizer.fit(combined.iloc[train_idx])
    text_matrix = vectorizer.transform(combined)
    return vectorizer, text_matrix, combined


def save_features(root: Path, vectorizer, text_matrix, sentiment_matrix) -> None:
    with open(root / FEATURE_VECTORIZER_PKL, "wb") as f:
        pickle.dump(vectorizer, f)
    # Backward-compatible alias for dashboard / older scripts
    with open(root / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(root / TEXT_MATRIX_PKL, "wb") as f:
        pickle.dump(text_matrix, f)
    with open(root / SENTIMENT_MATRIX_PKL, "wb") as f:
        pickle.dump(sentiment_matrix, f)


def load_text_matrix(root: Path):
    with open(root / TEXT_MATRIX_PKL, "rb") as f:
        return pickle.load(f)


def load_sentiment_matrix(root: Path):
    with open(root / SENTIMENT_MATRIX_PKL, "rb") as f:
        return pickle.load(f)


def load_vectorizer(root: Path):
    path = root / FEATURE_VECTORIZER_PKL
    if not path.exists():
        path = root / "tfidf_vectorizer.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def transform_live(text: str, title: str, rating: float, vectorizer):
    combined = clean_review_text(f"{title} {text}".strip())
    text_x = vectorizer.transform([combined])
    sent_x = hstack([text_x, rating_matrix(np.array([rating]))])
    return text_x, sent_x
