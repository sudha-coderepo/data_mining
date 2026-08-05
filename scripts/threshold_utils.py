"""Per-class threshold tuning for multiclass Macro F1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def tune_multiclass_thresholds(y_true, proba, classes, grid=None) -> dict[str, float]:
    """One-vs-rest threshold per class; pick threshold maximizing binary F1 on validation."""
    if grid is None:
        grid = np.linspace(0.2, 0.8, 13)

    y_true = np.asarray(y_true)
    thresholds = {}
    for i, cls in enumerate(classes):
        binary_true = (y_true == cls).astype(int)
        if binary_true.sum() == 0:
            thresholds[str(cls)] = 0.5
            continue
        best_t, best_f1 = 0.5, -1.0
        col = proba[:, i]
        for t in grid:
            pred = (col >= t).astype(int)
            f1 = f1_score(binary_true, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[str(cls)] = best_t
    return thresholds


def predict_with_thresholds(proba, classes, thresholds: dict) -> np.ndarray:
    adjusted = proba.copy()
    for i, cls in enumerate(classes):
        t = thresholds.get(str(cls), 0.5)
        adjusted[:, i] = proba[:, i] - t
    return np.array([classes[i] for i in adjusted.argmax(axis=1)])


def save_thresholds(path: Path, task: str, thresholds: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"task": task, "thresholds": thresholds}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    data[task] = thresholds
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
