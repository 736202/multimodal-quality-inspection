from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    labels: list[float],
    probabilities: list[float],
    threshold: float = 0.5,
) -> dict[str, float | list]:
    """Compute a standard set of binary classification metrics.

    Parameters
    ----------
    labels:
        Ground-truth binary labels (0 or 1).
    probabilities:
        Predicted positive-class probabilities in [0, 1].
    threshold:
        Decision boundary; samples with probability ≥ threshold are predicted
        positive.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1``, ``auc_roc``,
        ``confusion_matrix``, ``threshold``.
    """
    y_true = np.array(labels, dtype=np.int32)
    y_prob = np.array(probabilities, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.int32)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": float(threshold),
    }
    return metrics


def curve_payload(labels: list[float], probabilities: list[float]) -> dict[str, list[float]]:
    """Compute ROC and precision-recall curve data for plotting.

    Parameters
    ----------
    labels:
        Ground-truth binary labels.
    probabilities:
        Predicted positive-class probabilities.

    Returns
    -------
    dict
        ``roc_curve`` and ``pr_curve`` sub-dicts, each containing the arrays
        needed to draw the respective curve.
    """
    y_true = np.array(labels, dtype=np.int32)
    y_prob = np.array(probabilities, dtype=np.float32)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    return {
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
        "pr_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
    }


def select_best_threshold(
    labels: list[float],
    probabilities: list[float],
) -> dict[str, float | dict]:
    """Search for the decision threshold that maximises F1 on the validation set.

    The search grid spans [0.05, 0.95] with 181 equally-spaced candidates.
    Ties in F1 are broken by recall (higher is better), then by accuracy.
    This criterion reflects the industrial priority of avoiding missed defects
    (false negatives are costlier than false positives in quality control).

    Parameters
    ----------
    labels:
        Validation ground-truth labels.
    probabilities:
        Validation predicted probabilities.

    Returns
    -------
    dict
        ``selected_threshold``, ``metrics_at_selected_threshold``, and
        ``metrics_at_default_threshold`` (at 0.5 for comparison).
    """
    candidate_thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_metrics = compute_classification_metrics(labels, probabilities, threshold=0.5)
    best_score = (best_metrics["f1"], best_metrics["recall"], best_metrics["accuracy"])

    for threshold in candidate_thresholds:
        metrics = compute_classification_metrics(labels, probabilities, threshold=float(threshold))
        score = (metrics["f1"], metrics["recall"], metrics["accuracy"])
        if score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score

    return {
        "selected_threshold": best_threshold,
        "metrics_at_selected_threshold": best_metrics,
        "metrics_at_default_threshold": compute_classification_metrics(labels, probabilities, threshold=0.5),
    }


def save_json(payload: dict, output_path: Path) -> None:
    """Serialise a dict to a JSON file, creating parent directories as needed.

    Parameters
    ----------
    payload:
        JSON-serialisable dictionary.
    output_path:
        Destination file path (extension ``.json`` recommended).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
