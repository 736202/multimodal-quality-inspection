from __future__ import annotations

import pytest

from mqi.training.metrics import (
    compute_classification_metrics,
    curve_payload,
    select_best_threshold,
)


# ── compute_classification_metrics ───────────────────────────────────────────

def test_perfect_predictions():
    labels = [0, 0, 1, 1]
    probs  = [0.1, 0.2, 0.8, 0.9]
    m = compute_classification_metrics(labels, probs)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["recall"]   == pytest.approx(1.0)
    assert m["f1"]       == pytest.approx(1.0)


def test_all_wrong_predictions():
    labels = [0, 0, 1, 1]
    probs  = [0.9, 0.9, 0.1, 0.1]
    m = compute_classification_metrics(labels, probs)
    assert m["accuracy"] == pytest.approx(0.0)
    assert m["recall"]   == pytest.approx(0.0)


def test_threshold_affects_predictions():
    labels = [1, 1, 0, 0]
    probs  = [0.4, 0.4, 0.3, 0.3]
    m_low  = compute_classification_metrics(labels, probs, threshold=0.35)
    m_high = compute_classification_metrics(labels, probs, threshold=0.45)
    # Low threshold: both positives are caught
    assert m_low["recall"] == pytest.approx(1.0)
    # High threshold: positives are missed
    assert m_high["recall"] == pytest.approx(0.0)


def test_confusion_matrix_shape():
    labels = [0, 1, 0, 1]
    probs  = [0.2, 0.8, 0.3, 0.7]
    m = compute_classification_metrics(labels, probs)
    cm = m["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2


def test_threshold_in_output():
    labels = [0, 1]
    probs  = [0.3, 0.7]
    m = compute_classification_metrics(labels, probs, threshold=0.6)
    assert m["threshold"] == pytest.approx(0.6)


# ── curve_payload ─────────────────────────────────────────────────────────────

def test_curve_payload_keys():
    labels = [0, 0, 1, 1]
    probs  = [0.1, 0.3, 0.6, 0.9]
    payload = curve_payload(labels, probs)
    assert "roc_curve" in payload
    assert "pr_curve" in payload
    assert "fpr" in payload["roc_curve"]
    assert "tpr" in payload["roc_curve"]


# ── select_best_threshold ─────────────────────────────────────────────────────

def test_select_best_threshold_returns_valid_threshold():
    labels = [0, 0, 1, 1, 0, 1]
    probs  = [0.1, 0.2, 0.7, 0.8, 0.3, 0.6]
    result = select_best_threshold(labels, probs)
    t = result["selected_threshold"]
    assert 0.0 <= t <= 1.0


def test_select_best_threshold_recall_priority():
    """A threshold that maximises recall should be preferred over plain accuracy."""
    labels = [1] * 10 + [0] * 2
    probs  = [0.4] * 10 + [0.1, 0.1]
    result = select_best_threshold(labels, probs)
    m = result["metrics_at_selected_threshold"]
    # With threshold ≤ 0.4, all positives are captured
    assert m["recall"] == pytest.approx(1.0)
