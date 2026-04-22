"""Statistical validation: bootstrap confidence intervals + McNemar tests.

Loads saved predictions from outputs/{mode}/test_metrics.json and computes:

bootstrap_ci
    95 % CI for F1, AUC-ROC, Recall, Precision for all three models
    (1 000 bootstrap resamples, no retraining required).

mcnemar
    McNemar chi-squared test (with continuity correction) for all three
    model pairs: image vs sensor, image vs multimodal, sensor vs multimodal.
    Tests whether the two classifiers make significantly different errors.

Results saved to outputs/final_report/statistical_validation.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap CI and McNemar tests.")
    parser.add_argument(
        "--outputs-dir", type=Path,
        default=PROJECT_ROOT / "outputs",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_predictions(outputs_dir: Path, mode: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (labels, probabilities, threshold) from saved test_metrics.json."""
    path = outputs_dir / mode / "test_metrics.json"
    data = json.loads(path.read_text())
    labels = np.array(data["labels"], dtype=np.float32)
    probs  = np.array(data["probabilities"], dtype=np.float32)
    try:
        thr = data["threshold_selection"]["selected_threshold"]
    except (KeyError, TypeError):
        thr = data.get("metrics", {}).get("threshold", 0.5)
    return labels, probs, float(thr)


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Compute 95 % bootstrap CI for F1, AUC, Recall, Precision.

    Each bootstrap iteration resamples the test set with replacement and
    computes all four metrics.  The CI is the [alpha/2, 1-alpha/2] percentile
    interval of the bootstrap distribution.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    preds = (probs >= threshold).astype(int)

    metrics = {"f1": [], "auc_roc": [], "recall": [], "precision": []}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_true = labels[idx]
        y_prob = probs[idx]
        y_pred = preds[idx]

        if len(np.unique(y_true)) < 2:
            continue

        metrics["f1"].append(f1_score(y_true, y_pred, zero_division=0))
        metrics["auc_roc"].append(roc_auc_score(y_true, y_prob))
        metrics["recall"].append(recall_score(y_true, y_pred, zero_division=0))
        metrics["precision"].append(precision_score(y_true, y_pred, zero_division=0))

    lo, hi = (alpha / 2) * 100, (1 - alpha / 2) * 100
    result = {}
    for metric, values in metrics.items():
        arr = np.array(values)
        result[metric] = {
            "mean":  round(float(arr.mean()), 4),
            "ci_lo": round(float(np.percentile(arr, lo)), 4),
            "ci_hi": round(float(np.percentile(arr, hi)), 4),
            "std":   round(float(arr.std()), 4),
        }
    return result


# ── McNemar test ──────────────────────────────────────────────────────────────

def mcnemar_test(
    labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> dict:
    """McNemar test with continuity correction (Fleiss, 1981).

    Contingency table:
        b = #{A correct, B wrong}
        c = #{A wrong,   B correct}

    H0: the two classifiers make the same error rate.
    Statistic: chi2 = (|b - c| - 1)^2 / (b + c)   (continuity correction)
    Distribution: chi2(1) under H0.
    """
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    b = int(( correct_a & ~correct_b).sum())   # A right, B wrong
    c = int((~correct_a &  correct_b).sum())   # A wrong, B right

    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0,
                "significant_at_05": False, "significant_at_01": False,
                "interpretation": "Accord parfait (b+c=0) — test non applicable",
                "note": "b+c=0, no discordant pairs"}

    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)

    # p-value from chi2(1) survival function (1 - CDF)
    # Approximation via regularised incomplete gamma:
    # P(chi2(1) > x) = 1 - erf(sqrt(x/2))
    p_value = 1.0 - math.erf(math.sqrt(chi2 / 2.0))

    return {
        "b": b,
        "c": c,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 4),
        "significant_at_05": bool(p_value < 0.05),
        "significant_at_01": bool(p_value < 0.01),
        "interpretation": (
            "Différence significative (p < 0.05)" if p_value < 0.05
            else "Différence non significative (p ≥ 0.05)"
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("Loading predictions …")
    labels_img, probs_img, thr_img = load_predictions(args.outputs_dir, "image")
    labels_sen, probs_sen, thr_sen = load_predictions(args.outputs_dir, "sensor")
    labels_mm,  probs_mm,  thr_mm  = load_predictions(args.outputs_dir, "multimodal")

    # All models must share the same test set
    assert np.array_equal(labels_img, labels_sen), "Test label mismatch img/sen"
    assert np.array_equal(labels_img, labels_mm),  "Test label mismatch img/mm"
    labels = labels_img
    n_test = len(labels)
    print(f"  n_test={n_test}, n_defects={int(labels.sum())}, n_ok={int((labels==0).sum())}")

    preds_img = (probs_img >= thr_img).astype(int)
    preds_sen = (probs_sen >= thr_sen).astype(int)
    preds_mm  = (probs_mm  >= thr_mm ).astype(int)

    # ── Bootstrap CI ─────────────────────────────────────────────────────────
    print(f"\nBootstrap CI ({args.n_bootstrap} resamples) …")
    ci_img = bootstrap_ci(labels, probs_img, thr_img, args.n_bootstrap, args.seed)
    ci_sen = bootstrap_ci(labels, probs_sen, thr_sen, args.n_bootstrap, args.seed + 1)
    ci_mm  = bootstrap_ci(labels, probs_mm,  thr_mm,  args.n_bootstrap, args.seed + 2)

    for name, ci in [("image", ci_img), ("sensor", ci_sen), ("multimodal", ci_mm)]:
        f1 = ci["f1"]
        print(f"  {name}: F1 = {f1['mean']:.4f}  [95% CI: {f1['ci_lo']:.4f} – {f1['ci_hi']:.4f}]")

    # ── McNemar tests ─────────────────────────────────────────────────────────
    print("\nMcNemar tests …")
    mc_img_sen = mcnemar_test(labels, preds_img, preds_sen)
    mc_img_mm  = mcnemar_test(labels, preds_img, preds_mm)
    mc_sen_mm  = mcnemar_test(labels, preds_sen, preds_mm)

    for pair, result in [
        ("image vs sensor",     mc_img_sen),
        ("image vs multimodal", mc_img_mm),
        ("sensor vs multimodal",mc_sen_mm),
    ]:
        print(f"  {pair}: chi2={result['chi2']:.4f}, p={result['p_value']:.4f} — {result['interpretation']}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = {
        "n_test": n_test,
        "n_bootstrap": args.n_bootstrap,
        "thresholds": {"image": thr_img, "sensor": thr_sen, "multimodal": thr_mm},
        "bootstrap_ci": {
            "image":      ci_img,
            "sensor":     ci_sen,
            "multimodal": ci_mm,
        },
        "mcnemar": {
            "image_vs_sensor":     mc_img_sen,
            "image_vs_multimodal": mc_img_mm,
            "sensor_vs_multimodal":mc_sen_mm,
        },
    }
    out_path = args.outputs_dir / "final_report" / "statistical_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
