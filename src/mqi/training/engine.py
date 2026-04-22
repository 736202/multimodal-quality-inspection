from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
import time

import torch
from torch import nn
from tqdm import tqdm

from mqi.training.metrics import compute_classification_metrics, curve_payload, save_json

logger = logging.getLogger(__name__)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Transfer all tensor values in a batch dict to the target device.

    Parameters
    ----------
    batch:
        Dictionary of tensors as returned by a PyTorch DataLoader.
    device:
        Target computation device.

    Returns
    -------
    dict
        New dict with the same keys; tensor values moved to ``device``,
        non-tensor values (e.g. sample IDs) left unchanged.
    """
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def forward_by_mode(model: nn.Module, batch: dict, mode: str) -> torch.Tensor:
    """Dispatch a forward pass to the correct model signature based on mode.

    Parameters
    ----------
    model:
        The model to call.
    batch:
        Device-side batch dictionary.
    mode:
        One of ``"image"``, ``"sensor"``, or ``"multimodal"``.

    Returns
    -------
    torch.Tensor
        Raw logits of shape ``(B,)``.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the three supported values.
    """
    if mode == "image":
        return model(batch["image"])
    if mode == "sensor":
        return model(batch["sensors"])
    if mode == "multimodal":
        return model(batch["image"], batch["sensors"])
    raise ValueError(f"Unsupported mode: {mode}")


def run_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    mode: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Run one full epoch (training or evaluation) and return aggregate metrics.

    Parameters
    ----------
    model:
        Model to evaluate or update.
    dataloader:
        DataLoader for the current partition.
    criterion:
        Loss function (``BCEWithLogitsLoss``).
    device:
        Computation device.
    mode:
        Modality mode forwarded to :func:`forward_by_mode`.
    optimizer:
        If provided, gradients are computed and weights updated (training mode).
        Pass ``None`` for validation/test (evaluation mode, no grad).

    Returns
    -------
    dict
        Metric dictionary including ``loss``, ``accuracy``, ``f1``, etc.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_count = 0
    all_labels: list[float] = []
    all_probabilities: list[float] = []

    for batch in tqdm(dataloader, leave=False):
        batch = move_batch_to_device(batch, device)
        labels = batch["label"]

        with torch.set_grad_enabled(is_train):
            logits = forward_by_mode(model, batch, mode)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        probabilities = torch.sigmoid(logits)
        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        all_labels.extend(labels.detach().cpu().tolist())
        all_probabilities.extend(probabilities.detach().cpu().tolist())

    metrics = compute_classification_metrics(all_labels, all_probabilities)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    mode: str,
    max_epochs: int,
    patience: int,
    output_dir: Path,
) -> dict:
    """Full training loop with early stopping and best-model checkpointing.

    At each epoch the model is evaluated on both training and validation sets.
    The best validation-loss checkpoint is saved to ``output_dir/best_model.pt``
    and restored at the end of training.  Training stops early when validation
    loss has not improved for ``patience`` consecutive epochs.

    Parameters
    ----------
    model:
        Uninitialised or partially pretrained model on ``device``.
    train_loader:
        DataLoader for the training partition.
    val_loader:
        DataLoader for the validation partition.
    criterion:
        Loss function.
    optimizer:
        Optimiser instance (e.g. Adam).
    scheduler:
        Learning-rate scheduler (e.g. ReduceLROnPlateau).
    device:
        Computation device.
    mode:
        Modality mode.
    max_epochs:
        Maximum number of epochs before forced termination.
    patience:
        Number of non-improving epochs before early stopping.
    output_dir:
        Directory where ``best_model.pt`` and ``training_history.json`` are saved.

    Returns
    -------
    dict
        Training report including ``mode``, ``epochs_ran``, ``best_val_loss``,
        ``duration_seconds``, and full per-epoch ``history``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict] = []
    started_at = time.time()

    for epoch in range(1, max_epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, mode, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, mode, optimizer=None)
        scheduler.step(val_metrics["loss"])

        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_metrics)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f val_loss=%.4f val_f1=%.4f",
            epoch,
            max_epochs,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["f1"],
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_state, output_dir / "best_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info("Early stopping triggered at epoch %d (patience=%d).", epoch, patience)
            break

    model.load_state_dict(best_state)

    report = {
        "mode": mode,
        "duration_seconds": round(time.time() - started_at, 2),
        "best_val_loss": best_val_loss,
        "epochs_ran": len(history),
        "history": history,
    }
    save_json(report, output_dir / "training_history.json")
    return report


@torch.no_grad()
def predict(
    model: nn.Module,
    dataloader,
    device: torch.device,
    mode: str,
    threshold: float = 0.5,
) -> dict:
    """Run inference on a DataLoader and collect predictions with metrics.

    Parameters
    ----------
    model:
        Trained model in evaluation mode.
    dataloader:
        DataLoader for the target partition.
    device:
        Computation device.
    mode:
        Modality mode.
    threshold:
        Decision threshold applied to sigmoid probabilities.

    Returns
    -------
    dict
        ``sample_ids``, ``labels``, ``probabilities``, ``metrics``, and
        ``curves`` (ROC + PR).
    """
    model.eval()
    all_labels: list[float] = []
    all_probabilities: list[float] = []
    all_sample_ids: list[str] = []

    for batch in tqdm(dataloader, leave=False):
        sample_ids = batch["sample_id"]
        batch = move_batch_to_device(batch, device)
        logits = forward_by_mode(model, batch, mode)
        probabilities = torch.sigmoid(logits)

        all_labels.extend(batch["label"].detach().cpu().tolist())
        all_probabilities.extend(probabilities.detach().cpu().tolist())
        all_sample_ids.extend(sample_ids)

    return {
        "sample_ids": all_sample_ids,
        "labels": all_labels,
        "probabilities": all_probabilities,
        "metrics": compute_classification_metrics(all_labels, all_probabilities, threshold=threshold),
        "curves": curve_payload(all_labels, all_probabilities),
    }
