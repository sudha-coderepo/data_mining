"""
Evaluate models on human gold (Track A), validation stars (Track B), LLM distillation (Track C).

Usage:
    python scripts/evaluation.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import (
    majority_category_predict,
    majority_sentiment_predict,
    predict_keyword_category,
    predict_majority_category,
    predict_majority_sentiment,
)
from split_utils import HUMAN_GOLD_CSV, OUTPUTS_DIR, load_gold_indices, load_pickle, TRAIN_INDICES_PKL, VAL_INDICES_PKL


def calc_metrics(y_true, y_pred, model_name: str) -> dict:
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_Precision": p_m,
        "Macro_Recall": r_m,
        "Macro_F1": f1_m,
        "Weighted_Precision": p_w,
        "Weighted_Recall": r_w,
        "Weighted_F1": f1_w,
    }


def plot_confusion(y_true, y_pred, title: str, out_path: Path) -> None:
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_comparison(df: pd.DataFrame, title: str, out_path: Path) -> None:
    melted = df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Macro_F1", "Weighted_F1"],
        var_name="Metric",
        value_name="Score",
    )
    plt.figure(figsize=(11, 6))
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette=["#2b7bba", "#e68422", "#4caf50"])
    plt.title(title, fontsize=13, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def row_indices_for_gold(df: pd.DataFrame, gold_df: pd.DataFrame) -> list[int]:
    rid_to_idx = {str(r): i for i, r in enumerate(df["review_id"].astype(str))}
    return [rid_to_idx[str(rid)] for rid in gold_df["review_id"].astype(str)]


def evaluate_track(
    df: pd.DataFrame,
    X,
    indices,
    y_true_col: str,
    task: str,
    prefix: str,
    y_train_sentiment,
    y_train_category,
    include_llm: bool = False,
) -> pd.DataFrame:
    results = []
    y_true = df.loc[indices, y_true_col]
    X_sub = X[indices]
    texts = df.loc[indices, "reviews.text"]

    maj_sent = majority_sentiment_predict(y_train_sentiment)
    maj_cat = majority_category_predict(y_train_category)

    if task == "sentiment":
        results.append(calc_metrics(y_true, predict_majority_sentiment(len(y_true), maj_sent), "Majority Class"))
    else:
        results.append(calc_metrics(y_true, predict_majority_category(len(y_true), maj_cat), "Majority Class"))
        results.append(calc_metrics(y_true, predict_keyword_category(texts), "Keyword Rules"))

    for name in ["naive_bayes", "logistic_regression", "random_forest"]:
        with open(ROOT / f"{prefix}_{name}.pkl", "rb") as f:
            model = pickle.load(f)
        display = name.replace("_", " ").title()
        results.append(calc_metrics(y_true, model.predict(X_sub), display))

    if include_llm:
        llm_col = "llm_sentiment" if task == "sentiment" else "llm_category"
        results.append(calc_metrics(y_true, df.loc[indices, llm_col], "LLM (Zero-Shot)"))

    return pd.DataFrame(results)


def main() -> None:
    metrics_dir = OUTPUTS_DIR / "metrics"
    figures_dir = OUTPUTS_DIR / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not HUMAN_GOLD_CSV.exists():
        raise FileNotFoundError(f"Missing {HUMAN_GOLD_CSV}. Run merge_human_labels.py first.")

    df = pd.read_csv(ROOT / "preprocessed_reviews.csv", low_memory=False)
    gold_df = pd.read_csv(HUMAN_GOLD_CSV, low_memory=False)
    with open(ROOT / "tfidf_matrix.pkl", "rb") as f:
        X = pickle.load(f)

    train_idx = list(load_pickle(TRAIN_INDICES_PKL))
    val_idx = list(load_pickle(VAL_INDICES_PKL))
    gold_idx = row_indices_for_gold(df, gold_df)

    y_train_sent = df.loc[train_idx, "sentiment"]
    y_train_cat = df.loc[train_idx, "llm_category"]

    gold_map = gold_df.set_index(gold_df["review_id"].astype(str))
    for idx in gold_idx:
        rid = str(df.loc[idx, "review_id"])
        df.loc[idx, "human_sentiment"] = gold_map.loc[rid, "human_sentiment"]
        df.loc[idx, "human_category"] = gold_map.loc[rid, "human_category"]

    print("\n" + "=" * 60)
    print(" TRACK A — Human Gold (PRIMARY, n=100)")
    print("=" * 60)

    sent_a = evaluate_track(
        df, X, gold_idx, "human_sentiment", "sentiment", "sentiment",
        y_train_sent, y_train_cat, include_llm=True,
    )
    cat_a = evaluate_track(
        df, X, gold_idx, "human_category", "category", "category",
        y_train_sent, y_train_cat, include_llm=True,
    )
    print("\nSentiment (human gold):")
    print(sent_a[["Model", "Accuracy", "Macro_F1", "Weighted_F1"]].to_string(index=False))
    print("\nTheme (human gold):")
    print(cat_a[["Model", "Accuracy", "Macro_F1", "Weighted_F1"]].to_string(index=False))

    sent_a.to_csv(metrics_dir / "track_a_sentiment_gold.csv", index=False)
    cat_a.to_csv(metrics_dir / "track_a_category_gold.csv", index=False)
    sent_a.to_csv(ROOT / "sentiment_metrics_comparison.csv", index=False)
    cat_a.to_csv(ROOT / "category_metrics_comparison.csv", index=False)

    # Confusion matrices — best classical = Random Forest
    with open(ROOT / "sentiment_random_forest.pkl", "rb") as f:
        rf_sent = pickle.load(f)
    with open(ROOT / "category_random_forest.pkl", "rb") as f:
        rf_cat = pickle.load(f)

    y_true_s = df.loc[gold_idx, "human_sentiment"]
    y_pred_s = rf_sent.predict(X[gold_idx])
    y_true_c = df.loc[gold_idx, "human_category"]
    y_pred_c = rf_cat.predict(X[gold_idx])

    plot_confusion(
        y_true_s, y_pred_s,
        "Sentiment — Random Forest vs Human Gold",
        figures_dir / "sentiment_confusion_gold.png",
    )
    plot_confusion(
        y_true_c, y_pred_c,
        "Theme — Random Forest vs Human Gold",
        figures_dir / "category_confusion_gold.png",
    )
    plot_confusion(y_true_s, y_pred_s, "Sentiment — Random Forest vs Human Gold", ROOT / "sentiment_confusion.png")
    plot_confusion(y_true_c, y_pred_c, "Theme — Random Forest vs Human Gold", ROOT / "category_confusion.png")

    plot_comparison(
        sent_a, "Sentiment Model Comparison (Human Gold Ground Truth)",
        figures_dir / "sentiment_comparison_gold.png",
    )
    plot_comparison(
        cat_a, "Theme Model Comparison (Human Gold Ground Truth)",
        figures_dir / "category_comparison_gold.png",
    )
    plot_comparison(sent_a, "Sentiment Model Comparison (Human Gold Ground Truth)", ROOT / "sentiment_comparison.png")
    plot_comparison(cat_a, "Theme Model Comparison (Human Gold Ground Truth)", ROOT / "category_comparison.png")

    print("\n" + "=" * 60)
    print(" TRACK B — Validation vs Star Sentiment (n=216)")
    print("=" * 60)
    sent_b = evaluate_track(
        df, X, val_idx, "sentiment", "sentiment", "sentiment",
        y_train_sent, y_train_cat, include_llm=True,
    )
    print(sent_b[["Model", "Accuracy", "Macro_F1", "Weighted_F1"]].to_string(index=False))
    sent_b.to_csv(metrics_dir / "track_b_sentiment_stars.csv", index=False)

    print("\n" + "=" * 60)
    print(" TRACK C — Validation vs LLM Labels (distillation, n=216)")
    print("=" * 60)
    sent_c = evaluate_track(
        df, X, val_idx, "llm_sentiment", "sentiment", "sentiment",
        y_train_sent, y_train_cat,
    )
    cat_c = evaluate_track(
        df, X, val_idx, "llm_category", "category", "category",
        y_train_sent, y_train_cat,
    )
    print("\nSentiment vs LLM:")
    print(sent_c[["Model", "Accuracy", "Macro_F1", "Weighted_F1"]].to_string(index=False))
    print("\nTheme vs LLM:")
    print(cat_c[["Model", "Accuracy", "Macro_F1", "Weighted_F1"]].to_string(index=False))
    sent_c.to_csv(metrics_dir / "track_c_sentiment_llm.csv", index=False)
    cat_c.to_csv(metrics_dir / "track_c_category_llm.csv", index=False)

    print("\nEvaluation complete. Primary results: outputs/metrics/track_a_*_gold.csv")


if __name__ == "__main__":
    main()
