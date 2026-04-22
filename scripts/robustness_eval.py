"""Robustness and ablation experiments for the multimodal casting inspection system.

Produces ``outputs/final_report/robustness_results.json`` with four experiment blocks:

degradation
    F1, AUC-ROC, recall for {image, multimodal} under {none, moderate, severe}
    image corruption.

sensor_missing
    F1, AUC-ROC, recall for {sensor, multimodal} at missing rates
    {0, 0.10, 0.20, 0.30, 0.40}.

sensor_ablation
    AUC-ROC of the sensor model when each of the 6 features is zeroed
    independently.  Baseline (no zeroing) is included for reference.

cost_analysis
    Total asymmetric cost and optimal threshold under C_FN / C_FP = 10
    for all three models, sweeping thresholds in [0.05, 0.95].

inference_time
    Mean and std latency (ms / sample) for all three models over the test set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from mqi.data.catalog import build_catalog, stratified_split
from mqi.data.datasets import (
    build_image_transform,
    fit_sensor_scaler,
    split_records,
)
from mqi.data.degradation import DEGRADATION_LEVELS, corrupt_image, mask_sensors
from mqi.data.synthetic_sensors import SENSOR_COLUMNS, generate_sensor_table
from mqi.models.image import ImageClassifier
from mqi.models.sensors import SensorClassifier
from mqi.models.multimodal import MultimodalClassifier
from mqi.utils.repro import seed_everything

from PIL import Image as PILImage


# ── Cost-analysis constants ────────────────────────────────────────────────────
# In industrial quality control a false negative (defective part shipped) carries
# a cost roughly 10× that of a false positive (good part scrapped).
C_FN = 10.0
C_FP = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness and ablation experiments.")
    parser.add_argument(
        "--outputs-dir", type=Path,
        default=PROJECT_ROOT / "outputs",
    )
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_cfg(outputs_dir: Path, mode: str) -> dict:
    return json.loads((outputs_dir / mode / "config.json").read_text())


def _load_image_model(outputs_dir: Path, device: torch.device) -> ImageClassifier:
    cfg = _load_cfg(outputs_dir, "image")
    model = ImageClassifier(pretrained=False)
    model.load_state_dict(
        torch.load(outputs_dir / "image" / "best_model.pt",
                   map_location=device, weights_only=True)
    )
    model.eval()
    return model.to(device)


def _load_sensor_model(outputs_dir: Path, device: torch.device) -> SensorClassifier:
    cfg = _load_cfg(outputs_dir, "sensor")
    model = SensorClassifier(dropout=cfg.get("dropout_sensor", 0.3))
    model.load_state_dict(
        torch.load(outputs_dir / "sensor" / "best_model.pt",
                   map_location=device, weights_only=True)
    )
    model.eval()
    return model.to(device)


def _load_mm_model(outputs_dir: Path, device: torch.device) -> MultimodalClassifier:
    cfg = _load_cfg(outputs_dir, "multimodal")
    model = MultimodalClassifier(
        pretrained=False,
        sensor_dropout=cfg.get("dropout_sensor", 0.3),
        fusion_dropout_1=cfg.get("dropout_fusion_1", 0.4),
        fusion_dropout_2=cfg.get("dropout_fusion_2", 0.3),
    )
    model.load_state_dict(
        torch.load(outputs_dir / "multimodal" / "best_model.pt",
                   map_location=device, weights_only=True)
    )
    model.eval()
    return model.to(device)


def _load_threshold(outputs_dir: Path, mode: str) -> float:
    metrics = json.loads((outputs_dir / mode / "test_metrics.json").read_text())
    # Try threshold_selection first, fall back to metrics.threshold
    try:
        return metrics["threshold_selection"]["selected_threshold"]
    except (KeyError, TypeError):
        return metrics.get("metrics", {}).get("threshold", 0.5)


def _metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy  = (tp + tn) / max(len(labels), 1)
    auc       = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0
    return {
        "f1": round(f1, 4), "recall": round(recall, 4),
        "precision": round(precision, 4), "accuracy": round(accuracy, 4),
        "auc_roc": round(auc, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _optimal_cost_threshold(
    labels: np.ndarray, probs: np.ndarray, c_fn: float = C_FN, c_fp: float = C_FP
) -> tuple[float, float]:
    """Return (optimal_threshold, minimum_cost)."""
    thresholds = np.linspace(0.01, 0.99, 199)
    best_thr, best_cost = 0.5, float("inf")
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        fn = int(((preds == 0) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        cost = c_fn * fn + c_fp * fp
        if cost < best_cost:
            best_cost, best_thr = cost, float(thr)
    return best_thr, best_cost


# ── Image inference ───────────────────────────────────────────────────────────

def _image_probs(
    model: ImageClassifier,
    records,
    transform,
    device: torch.device,
    degradation: str = "none",
    batch_size: int = 32,
) -> np.ndarray:
    probs = []
    for start in range(0, len(records), batch_size):
        batch_recs = records[start:start + batch_size]
        tensors = []
        for rec in batch_recs:
            with PILImage.open(rec.image_path) as img:
                t = transform(img)
            tensors.append(t)
        batch = torch.stack(tensors).to(device)
        if degradation != "none":
            batch = corrupt_image(batch, level=degradation, seed=42)
        with torch.no_grad():
            logits = model(batch)
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.array(probs, dtype=np.float32)


def _sensor_probs(
    model: SensorClassifier,
    records,
    X: np.ndarray,
    device: torch.device,
    missing_rate: float = 0.0,
    zeroed_feature: int | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    X_use = X.copy()
    if zeroed_feature is not None:
        X_use[:, zeroed_feature] = 0.0
    probs = []
    for start in range(0, len(records), batch_size):
        batch = torch.tensor(X_use[start:start + batch_size], dtype=torch.float32)
        if missing_rate > 0.0:
            batch = mask_sensors(batch, missing_rate=missing_rate, seed=42 + start)
        batch = batch.to(device)
        with torch.no_grad():
            logits = model(batch)
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.array(probs, dtype=np.float32)


def _mm_probs(
    model: MultimodalClassifier,
    records,
    transform,
    X: np.ndarray,
    device: torch.device,
    degradation: str = "none",
    missing_rate: float = 0.0,
    batch_size: int = 32,
) -> np.ndarray:
    probs = []
    for start in range(0, len(records), batch_size):
        batch_recs = records[start:start + batch_size]
        n = len(batch_recs)
        tensors = []
        for rec in batch_recs:
            with PILImage.open(rec.image_path) as img:
                t = transform(img)
            tensors.append(t)
        img_batch = torch.stack(tensors).to(device)
        if degradation != "none":
            img_batch = corrupt_image(img_batch, level=degradation, seed=42)
        sen_batch = torch.tensor(X[start:start + n], dtype=torch.float32)
        if missing_rate > 0.0:
            sen_batch = mask_sensors(sen_batch, missing_rate=missing_rate, seed=42 + start)
        sen_batch = sen_batch.to(device)
        with torch.no_grad():
            logits = model(img_batch, sen_batch)
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.array(probs, dtype=np.float32)


# ── Experiment runners ────────────────────────────────────────────────────────

def exp_degradation(
    img_model, mm_model, records, transform, X_test, labels,
    img_thr, mm_thr, device
) -> dict:
    results = {}
    for level in DEGRADATION_LEVELS:
        img_probs = _image_probs(img_model, records, transform, device, degradation=level)
        mm_p      = _mm_probs(mm_model, records, transform, X_test, device, degradation=level)
        results[level] = {
            "image":      _metrics(labels, img_probs, img_thr),
            "multimodal": _metrics(labels, mm_p,      mm_thr),
        }
        print(f"  degradation={level}: image F1={results[level]['image']['f1']:.4f}, "
              f"fusion F1={results[level]['multimodal']['f1']:.4f}")
    return results


def exp_sensor_missing(
    sen_model, mm_model, records, transform, X_test, labels,
    sen_thr, mm_thr, device
) -> dict:
    rates = [0.0, 0.10, 0.20, 0.30, 0.40]
    results = {}
    for rate in rates:
        sen_p = _sensor_probs(sen_model, records, X_test, device, missing_rate=rate)
        mm_p  = _mm_probs(mm_model, records, transform, X_test, device, missing_rate=rate)
        key = f"{int(rate * 100)}pct"
        results[key] = {
            "missing_rate": rate,
            "sensor":     _metrics(labels, sen_p, sen_thr),
            "multimodal": _metrics(labels, mm_p,  mm_thr),
        }
        print(f"  missing={rate:.0%}: sensor F1={results[key]['sensor']['f1']:.4f}, "
              f"fusion F1={results[key]['multimodal']['f1']:.4f}")
    return results


def exp_sensor_ablation(
    sen_model, records, X_test, labels, sen_thr, device
) -> dict:
    # Baseline
    base_probs = _sensor_probs(sen_model, records, X_test, device)
    base_auc   = float(roc_auc_score(labels, base_probs))
    results = {"baseline": {"auc_roc": round(base_auc, 4), "feature_zeroed": None}}
    for fi, col in enumerate(SENSOR_COLUMNS):
        probs = _sensor_probs(sen_model, records, X_test, device, zeroed_feature=fi)
        auc   = float(roc_auc_score(labels, probs))
        drop  = base_auc - auc
        results[col] = {
            "auc_roc": round(auc, 4),
            "auc_drop": round(drop, 4),
            "feature_zeroed": col,
        }
        print(f"  ablation zero({col}): AUC={auc:.4f}  drop={drop:+.4f}")
    return results


def exp_cost_analysis(
    img_model, sen_model, mm_model,
    records, transform, X_test, labels,
    device
) -> dict:
    img_probs = _image_probs(img_model, records, transform, device)
    sen_probs = _sensor_probs(sen_model, records, X_test, device)
    mm_probs  = _mm_probs(mm_model, records, transform, X_test, device)

    results = {}
    for name, probs in [("image", img_probs), ("sensor", sen_probs), ("multimodal", mm_probs)]:
        opt_thr, opt_cost = _optimal_cost_threshold(labels, probs)
        # Cost curve (for plotting)
        thresholds = [round(t, 2) for t in np.linspace(0.05, 0.95, 91).tolist()]
        costs = []
        for thr in thresholds:
            preds = (probs >= thr).astype(int)
            fn = int(((preds == 0) & (labels == 1)).sum())
            fp = int(((preds == 1) & (labels == 0)).sum())
            costs.append(C_FN * fn + C_FP * fp)
        results[name] = {
            "optimal_threshold": round(opt_thr, 3),
            "optimal_cost": int(opt_cost),
            "cost_fn_ratio": C_FN,
            "cost_fp_ratio": C_FP,
            "curve_thresholds": thresholds,
            "curve_costs": costs,
        }
        print(f"  cost({name}): opt_thr={opt_thr:.3f}  min_cost={opt_cost:.0f}")
    return results


def exp_inference_time(
    img_model, sen_model, mm_model,
    records, transform, X_test, device,
    n_warmup: int = 10, n_repeat: int = 5,
) -> dict:
    """Measure per-sample inference latency (ms) over n_repeat full passes."""
    n = len(records)

    def time_model(fn) -> tuple[float, float]:
        # warm up
        for _ in range(n_warmup):
            fn()
        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0 / n)  # ms/sample
        return float(np.mean(times)), float(np.std(times))

    # Pre-build batches to avoid I/O in timing
    imgs = []
    for rec in records:
        with PILImage.open(rec.image_path) as img:
            imgs.append(transform(img))
    img_batch = torch.stack(imgs).to(device)
    sen_batch = torch.tensor(X_test, dtype=torch.float32).to(device)

    def run_img():
        with torch.no_grad():
            img_model(img_batch)

    def run_sen():
        with torch.no_grad():
            sen_model(sen_batch)

    def run_mm():
        with torch.no_grad():
            mm_model(img_batch, sen_batch)

    mean_img, std_img = time_model(run_img)
    mean_sen, std_sen = time_model(run_sen)
    mean_mm,  std_mm  = time_model(run_mm)

    results = {
        "image":      {"mean_ms_per_sample": round(mean_img, 4), "std_ms": round(std_img, 4)},
        "sensor":     {"mean_ms_per_sample": round(mean_sen, 4), "std_ms": round(std_sen, 4)},
        "multimodal": {"mean_ms_per_sample": round(mean_mm,  4), "std_ms": round(std_mm,  4)},
        "n_test_samples": n,
    }
    for name, r in results.items():
        if isinstance(r, dict) and "mean_ms_per_sample" in r:
            print(f"  latency({name}): {r['mean_ms_per_sample']:.4f} ± {r['std_ms']:.4f} ms/sample")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    # ── Load shared config (use image config as canonical) ───────────────────
    cfg = _load_cfg(args.outputs_dir, "image")
    seed_everything(cfg["seed"])

    print("Loading catalog and sensor table …")
    records      = build_catalog(Path(cfg["dataset_dir"]))
    records      = stratified_split(records, val_ratio=cfg["val_ratio"],
                                    test_ratio=cfg["test_ratio"], seed=cfg["seed"])
    sensor_table = generate_sensor_table(records, base_seed=cfg["seed"])
    train_recs, _, test_recs = split_records(records)
    scaler       = fit_sensor_scaler(train_recs, sensor_table)

    X_test = np.array(
        [scaler.transform(sensor_table[r.sample_id]) for r in test_recs],
        dtype=np.float32,
    )
    labels = np.array([r.label for r in test_recs], dtype=np.float32)
    transform = build_image_transform(train=False, image_size=cfg["image_size"])

    print("Loading models …")
    img_model = _load_image_model(args.outputs_dir, device)
    sen_model = _load_sensor_model(args.outputs_dir, device)
    mm_model  = _load_mm_model(args.outputs_dir, device)

    img_thr = _load_threshold(args.outputs_dir, "image")
    sen_thr = _load_threshold(args.outputs_dir, "sensor")
    mm_thr  = _load_threshold(args.outputs_dir, "multimodal")
    print(f"  thresholds: image={img_thr}, sensor={sen_thr}, multimodal={mm_thr}")

    results: dict = {}

    # ── Experiment 1 : image degradation ────────────────────────────────────
    print("\n[1/4] Image degradation experiment …")
    results["degradation"] = exp_degradation(
        img_model, mm_model, test_recs, transform, X_test, labels,
        img_thr, mm_thr, device,
    )

    # ── Experiment 2 : sensor missing values ────────────────────────────────
    print("\n[2/4] Sensor missing values experiment …")
    results["sensor_missing"] = exp_sensor_missing(
        sen_model, mm_model, test_recs, transform, X_test, labels,
        sen_thr, mm_thr, device,
    )

    # ── Experiment 3 : per-sensor ablation ──────────────────────────────────
    print("\n[3/4] Per-sensor ablation …")
    results["sensor_ablation"] = exp_sensor_ablation(
        sen_model, test_recs, X_test, labels, sen_thr, device,
    )

    # ── Experiment 4 : cost analysis ────────────────────────────────────────
    print("\n[4/4] Cost analysis …")
    results["cost_analysis"] = exp_cost_analysis(
        img_model, sen_model, mm_model,
        test_recs, transform, X_test, labels, device,
    )

    # ── Inference time ───────────────────────────────────────────────────────
    print("\n[+] Inference time measurement …")
    results["inference_time"] = exp_inference_time(
        img_model, sen_model, mm_model,
        test_recs, transform, X_test, device,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = args.outputs_dir / "final_report" / "robustness_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
