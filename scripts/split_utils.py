"""Shared paths, label schema, and split helpers for the gold-labeling pipeline."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import pandas as pd

# Repository root (data_mining/)
ROOT = Path(__file__).resolve().parent.parent

LABELED_CSV = ROOT / "reviews_labeled_llm.csv"
SPLITS_DIR = ROOT / "splits"
LABELING_DIR = ROOT / "labeling"
OUTPUTS_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"

MANIFEST_CSV = SPLITS_DIR / "reviews_manifest.csv"
GOLD_INDICES_PKL = SPLITS_DIR / "gold_indices.pkl"
TRAIN_INDICES_PKL = SPLITS_DIR / "train_indices.pkl"
VAL_INDICES_PKL = SPLITS_DIR / "val_indices.pkl"
SPLIT_SUMMARY_JSON = SPLITS_DIR / "split_summary.json"

GOLD_TEMPLATE_CSV = LABELING_DIR / "gold_labeling_template.csv"
GOLD_TEMPLATE_XLSX = LABELING_DIR / "gold_labeling_template.xlsx"
HUMAN_GOLD_CSV = DATA_DIR / "reviews_human_gold.csv"
VALIDATION_REPORT = OUTPUTS_DIR / "human_label_validation_report.txt"

RANDOM_STATE = 42
GOLD_SIZE = 100
VAL_FRACTION = 0.2

VALID_SENTIMENTS = {"Positive", "Neutral", "Negative"}
VALID_CATEGORIES = {
    "Delivery issue",
    "Product quality issue",
    "Price complaint",
    "Customer service issue",
    "Feature request",
    "Other",
}

# Target counts for stratified gold sample (100 total)
GOLD_CATEGORY_TARGETS = {
    "Delivery issue": 12,
    "Product quality issue": 12,
    "Price complaint": 10,
    "Customer service issue": 10,
    "Feature request": 12,
    "Other": 20,
}

DELIVERY_KEYWORDS = (
    "shipping",
    "delivery",
    "delivered",
    "package",
    "arrived",
    "late",
    "delayed",
    "carrier",
    "fedex",
    "ups",
    "usps",
)


def ensure_dirs() -> None:
    for path in (SPLITS_DIR, LABELING_DIR, OUTPUTS_DIR, DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_labeled_reviews() -> pd.DataFrame:
    if not LABELED_CSV.exists():
        raise FileNotFoundError(f"Missing labeled dataset: {LABELED_CSV}")
    df = pd.read_csv(LABELED_CSV, low_memory=False)
    df = df.reset_index(drop=True)
    df["review_id"] = df.index.map(lambda i: f"R{i:05d}")
    if "sentiment" not in df.columns:
        raise ValueError("Expected 'sentiment' column from star-rating mapping.")
    return df


def save_pickle(path: Path, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_gold_indices() -> list[int]:
    if not GOLD_INDICES_PKL.exists():
        raise FileNotFoundError(
            f"Gold split not found. Run: python scripts/create_splits.py"
        )
    return list(load_pickle(GOLD_INDICES_PKL))


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(
            f"Manifest not found. Run: python scripts/create_splits.py"
        )
    return pd.read_csv(MANIFEST_CSV)


def write_split_summary(summary: dict) -> None:
    with open(SPLIT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
