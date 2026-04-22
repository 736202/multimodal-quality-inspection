from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation plots for a training run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate Grad-CAM visualisations for a sample of test images (image/multimodal modes only).",
    )
    parser.add_argument("--gradcam-samples", type=int, default=8, help="Number of images to visualise with Grad-CAM.")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_training_curves(history: list[dict], output_dir: Path) -> None:
    """Plot loss and F1-score evolution over training epochs."""
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train"]["loss"] for entry in history]
    val_loss = [entry["val"]["loss"] for entry in history]
    train_f1 = [entry["train"]["f1"] for entry in history]
    val_f1 = [entry["val"]["f1"] for entry in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Perte par époque")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_f1, label="train")
    axes[1].plot(epochs, val_f1, label="val")
    axes[1].set_title("F1-score par époque")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("F1")
    axes[1].legend()

    fig.tight_layout()
    path = output_dir / "training_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    logger.info("Saved training curves → %s", path)


def plot_confusion_matrix(matrix: list[list[int]], output_dir: Path) -> None:
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Prédit OK", "Prédit Défaut"],
        yticklabels=["Réel OK", "Réel Défaut"],
    )
    ax.set_title("Matrice de confusion (jeu de test)")
    fig.tight_layout()
    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", path)


def plot_roc_curve(curves: dict, auc_value: float, output_dir: Path) -> None:
    """Plot the ROC curve with AUC annotation."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(curves["fpr"], curves["tpr"], label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aléatoire")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "roc_curve.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    logger.info("Saved ROC curve → %s", path)


def plot_probability_distribution(
    labels: list[float],
    probabilities: list[float],
    threshold: float,
    output_dir: Path,
) -> None:
    """Plot predicted probability distributions for the two classes."""
    labels_array = np.array(labels)
    probs_array = np.array(probabilities)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(probs_array[labels_array == 0], color="#1f77b4", label="ok", kde=True, stat="density", ax=ax, alpha=0.5)
    sns.histplot(probs_array[labels_array == 1], color="#d62728", label="défaut", kde=True, stat="density", ax=ax, alpha=0.5)
    ax.axvline(threshold, linestyle="--", color="black", label=f"seuil = {threshold:.2f}")
    ax.set_xlabel("Probabilité prédite")
    ax.set_title("Distribution des probabilités par classe")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "probability_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    logger.info("Saved probability distribution → %s", path)


# ── Grad-CAM ─────────────────────────────────────────────────────────────────

def _compute_gradcam(model, image_tensor, device):
    """Compute a Grad-CAM heatmap for a single image using the last conv layer.

    Returns a 2-D numpy array (H×W) normalised to [0, 1].
    """
    import torch

    activations = {}
    gradients = {}

    # Hook the last convolutional layer of ResNet18 (layer4)
    def forward_hook(module, inp, out):
        activations["value"] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    target_layer = model.encoder.backbone.layer4  # ImageClassifier path
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    model.eval()
    inp = image_tensor.unsqueeze(0).to(device)
    inp.requires_grad_(False)

    # Forward
    logit = model(inp)
    model.zero_grad()
    logit.backward()

    fh.remove()
    bh.remove()

    act = activations["value"].squeeze(0)   # (C, h, w)
    grad = gradients["value"].squeeze(0)    # (C, h, w)
    weights = grad.mean(dim=(1, 2))         # global average pooling
    cam = (weights[:, None, None] * act).sum(dim=0)
    cam = torch.clamp(cam, min=0)
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    return cam.cpu().numpy()


def _denorm_image(tensor):
    """Reverse ImageNet normalisation for display purposes."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)


def generate_gradcam_grid(
    run_dir: Path,
    n_samples: int,
    output_dir: Path,
) -> None:
    """Load the best model and produce a Grad-CAM grid on test images.

    Works only for ``image`` and ``multimodal`` modes (requires the image
    backbone).  Silently skips if the model file or sensor mode is not found.
    """
    import torch
    from mqi.data.catalog import build_catalog, stratified_split
    from mqi.data.datasets import build_image_transform, split_records, fit_sensor_scaler
    from mqi.data.synthetic_sensors import generate_sensor_table
    from mqi.models.image import ImageClassifier
    from mqi.utils.repro import seed_everything

    config_path = run_dir / "config.json"
    model_path = run_dir / "best_model.pt"
    if not config_path.exists() or not model_path.exists():
        logger.warning("Skipping Grad-CAM: config.json or best_model.pt not found in %s", run_dir)
        return

    with config_path.open() as f:
        cfg = json.load(f)

    mode = cfg.get("mode", "image")  # TrainingConfig doesn't store mode, infer from dir name
    mode = run_dir.name  # directory name is the mode

    if mode not in ("image", "multimodal"):
        logger.info("Grad-CAM skipped for mode=%s (no image backbone).", mode)
        return

    seed_everything(cfg.get("seed", 42))
    dataset_dir = Path(cfg["dataset_dir"])
    image_size = cfg.get("image_size", 224)

    records = build_catalog(dataset_dir)
    records = stratified_split(records, val_ratio=cfg["val_ratio"], test_ratio=cfg["test_ratio"], seed=cfg["seed"])
    sensor_table = generate_sensor_table(records, base_seed=cfg["seed"])
    _, _, test_records = split_records(records)

    transform = build_image_transform(train=False, image_size=image_size)

    device = torch.device("cpu")
    model = ImageClassifier(pretrained=False).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)

    # For multimodal runs the saved state has keys like "image_encoder.*"; remap to "encoder.*"
    # For image-only runs keys are already "encoder.*" and "head.*"
    if any(k.startswith("image_encoder.") for k in state):
        # Multimodal: extract image encoder weights and synthesise a dummy head
        remapped = {
            k.replace("image_encoder.", "encoder.", 1): v
            for k, v in state.items()
            if k.startswith("image_encoder.")
        }
        # Build a compatible head from the existing model's random head weights
        head_state = {k: v for k, v in model.state_dict().items() if k.startswith("head.")}
        try:
            model.load_state_dict({**remapped, **head_state})
        except RuntimeError:
            logger.warning("Grad-CAM: could not remap multimodal weights — skipping.")
            return
    else:
        try:
            model.load_state_dict(state)
        except RuntimeError:
            logger.warning("Grad-CAM: could not load weights into ImageClassifier — skipping.")
            return

    n_samples = min(n_samples, len(test_records))
    indices = np.random.choice(len(test_records), n_samples, replace=False)
    selected = [test_records[i] for i in sorted(indices)]

    from PIL import Image as PILImage

    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for idx, record in enumerate(selected):
        with PILImage.open(record.image_path) as pil_img:
            tensor = transform(pil_img)

        cam = _compute_gradcam(model, tensor, device)
        original = _denorm_image(tensor)
        cam_resized = np.array(PILImage.fromarray((cam * 255).astype(np.uint8)).resize(
            (original.shape[1], original.shape[0]), PILImage.BILINEAR
        )) / 255.0

        ax_img = axes[idx * 2]
        ax_cam = axes[idx * 2 + 1]

        ax_img.imshow(original)
        ax_img.axis("off")
        label_str = "Défaut" if record.label == 1 else "OK"
        ax_img.set_title(f"{label_str}", fontsize=8)

        ax_cam.imshow(original)
        ax_cam.imshow(cam_resized, alpha=0.5, cmap="jet")
        ax_cam.axis("off")
        ax_cam.set_title("Grad-CAM", fontsize=8)

    # Hide unused axes
    for ax in axes[n_samples * 2:]:
        ax.axis("off")

    fig.suptitle("Grad-CAM — activation regions for defect detection", fontsize=12)
    fig.tight_layout()
    path = output_dir / "gradcam_grid.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    logger.info("Saved Grad-CAM grid → %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating evaluation plots for %s", run_dir)

    history = read_json(run_dir / "training_history.json")
    test_metrics = read_json(run_dir / "test_metrics.json")

    sns.set_theme(style="whitegrid")

    plot_training_curves(history["history"], run_dir)
    plot_confusion_matrix(test_metrics["metrics"]["confusion_matrix"], run_dir)
    plot_roc_curve(test_metrics["curves"]["roc_curve"], test_metrics["metrics"]["auc_roc"], run_dir)
    plot_probability_distribution(
        test_metrics["labels"],
        test_metrics["probabilities"],
        float(test_metrics["metrics"]["threshold"]),
        run_dir,
    )

    if args.gradcam:
        generate_gradcam_grid(run_dir, n_samples=args.gradcam_samples, output_dir=run_dir)

    logger.info("All plots saved to %s", run_dir)


if __name__ == "__main__":
    main()
