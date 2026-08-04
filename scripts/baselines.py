"""Baseline predictors for sentiment and theme classification."""

from __future__ import annotations

import numpy as np
import pandas as pd

RULES = {
    "Delivery issue": [
        "shipping", "delivery", "delivered", "package", "arrived", "late", "delayed",
    ],
    "Product quality issue": [
        "broken", "defect", "quality", "stopped working", "battery", "screen", "dead",
    ],
    "Price complaint": [
        "expensive", "overpriced", "price", "worth", "money", "cost", "pricy",
    ],
    "Customer service issue": [
        "support", "return", "refund", "service", "warranty", "billing",
    ],
    "Feature request": [
        "wish", "would be nice", "should add", "feature", "update", "hope",
    ],
}


def majority_sentiment_predict(y_train: pd.Series) -> str:
    return y_train.value_counts().idxmax()


def majority_category_predict(y_train: pd.Series) -> str:
    return y_train.value_counts().idxmax()


def predict_majority_sentiment(n: int, label: str) -> list[str]:
    return [label] * n


def predict_majority_category(n: int, label: str) -> list[str]:
    return [label] * n


def predict_keyword_category(texts: pd.Series) -> list[str]:
    preds = []
    for text in texts.astype(str):
        lowered = text.lower()
        matched = None
        for category, keywords in RULES.items():
            if any(kw in lowered for kw in keywords):
                matched = category
                break
        preds.append(matched or "Other")
    return preds
