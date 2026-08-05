"""
Rank unlabeled train-pool reviews for the next human labeling batch (Phase 4).

Usage:
    python scripts/select_labeling_candidates.py --task theme --top-n 50
    python scripts/select_labeling_candidates.py --task sentiment --top-n 50
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from split_utils import LABELING_DIR, load_gold_indices, load_pickle, TRAIN_INDICES_PKL

RARE_THEMES = {
    "Delivery issue",
    "Customer service issue",
    "Price complaint",
}


def parse_args():
    p = argparse.ArgumentParser(description="Select active learning labeling candidates")
    p.add_argument("--task", choices=["sentiment", "theme"], default="theme")
    p.add_argument("--top-n", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(ROOT / "preprocessed_reviews.csv", low_memory=False)
    train_idx = set(load_pickle(TRAIN_INDICES_PKL))
    gold_idx = set(load_gold_indices())
    assert train_idx.isdisjoint(gold_idx), "Gold rows must not be in train pool for export"

    prefix = "sentiment" if args.task == "sentiment" else "category"
    matrix_file = "sentiment_feature_matrix.pkl" if args.task == "sentiment" else "tfidf_matrix.pkl"
    with open(ROOT / matrix_file, "rb") as f:
        X = pickle.load(f)
    with open(ROOT / f"{prefix}_logistic_regression.pkl", "rb") as f:
        model = pickle.load(f)

    pool = sorted(train_idx)
    proba = model.predict_proba(X[pool])
    classes = list(model.classes_)
    uncertainty = 1.0 - proba.max(axis=1)
    sorted_p = np.sort(proba, axis=1)
    margin = 1.0 - (sorted_p[:, -1] - sorted_p[:, -2])
    pred = model.predict(X[pool])

    rows = []
    for i, idx in enumerate(pool):
        row = df.loc[idx]
        llm_col = "llm_sentiment" if args.task == "sentiment" else "llm_category"
        llm_val = row[llm_col]
        disagreement = float(pred[i] != llm_val)
        rare_boost = 0.2 if args.task == "theme" and llm_val in RARE_THEMES else 0.0
        final_score = 0.5 * uncertainty[i] + 0.3 * disagreement + 0.2 * margin[i] + rare_boost
        rows.append(
            {
                "review_id": row["review_id"],
                "final_score": round(final_score, 4),
                "uncertainty": round(float(uncertainty[i]), 4),
                "margin": round(float(margin[i]), 4),
                "disagreement": disagreement,
                "model_pred": pred[i],
                llm_col: llm_val,
                "reviews.rating": row["reviews.rating"],
                "snippet": str(row["reviews.text"])[:120] + "...",
            }
        )

    out = pd.DataFrame(rows).sort_values("final_score", ascending=False).head(args.top_n)
    LABELING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LABELING_DIR / f"active_learning_batch_{args.task}_{date.today().isoformat()}.csv"
    out.to_csv(out_path, index=False)
    print(f"Exported top {len(out)} candidates to {out_path}")
    print(out[["review_id", "final_score", "model_pred", llm_col]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
