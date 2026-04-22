from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from mqi.config import default_config
from mqi.data.catalog import build_catalog, stratified_split
from mqi.data.datasets import (
    ImageDataset,
    MultimodalDataset,
    SensorDataset,
    build_image_transform,
    class_weight,
    dataset_manifest,
    fit_sensor_scaler,
    split_records,
)
from mqi.data.synthetic_sensors import generate_sensor_table
from mqi.models.image import ImageClassifier
from mqi.models.multimodal import MultimodalClassifier
from mqi.models.sensors import SensorClassifier
from mqi.training.engine import predict, train_model
from mqi.training.metrics import save_json, select_best_threshold
from mqi.utils.repro import ensure_dir, seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MQI models.")
    parser.add_argument("--mode", choices=["image", "sensor", "multimodal"], default="multimodal")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def build_model(mode: str, pretrained: bool, dropout_sensor: float, dropout_f1: float, dropout_f2: float):
    """Instantiate the model matching the requested modality mode."""
    if mode == "image":
        return ImageClassifier(pretrained=pretrained)
    if mode == "sensor":
        return SensorClassifier(dropout=dropout_sensor)
    if mode == "multimodal":
        return MultimodalClassifier(
            pretrained=pretrained,
            sensor_dropout=dropout_sensor,
            fusion_dropout_1=dropout_f1,
            fusion_dropout_2=dropout_f2,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def build_dataloaders(mode: str, config, records, sensor_table):
    """Build train/val/test DataLoaders for the requested modality mode.

    The sensor scaler is fitted exclusively on training samples to prevent
    data leakage.
    """
    train_records, val_records, test_records = split_records(records)
    scaler = fit_sensor_scaler(train_records, sensor_table)

    train_transform = build_image_transform(train=True, image_size=config.image_size)
    eval_transform = build_image_transform(train=False, image_size=config.image_size)

    if mode == "image":
        train_dataset = ImageDataset(train_records, transform=train_transform)
        val_dataset = ImageDataset(val_records, transform=eval_transform)
        test_dataset = ImageDataset(test_records, transform=eval_transform)
    elif mode == "sensor":
        train_dataset = SensorDataset(train_records, sensor_table=sensor_table, scaler=scaler)
        val_dataset = SensorDataset(val_records, sensor_table=sensor_table, scaler=scaler)
        test_dataset = SensorDataset(test_records, sensor_table=sensor_table, scaler=scaler)
    else:
        train_dataset = MultimodalDataset(train_records, sensor_table=sensor_table, scaler=scaler, transform=train_transform)
        val_dataset = MultimodalDataset(val_records, sensor_table=sensor_table, scaler=scaler, transform=eval_transform)
        test_dataset = MultimodalDataset(test_records, sensor_table=sensor_table, scaler=scaler, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader, test_loader, train_records, scaler


def export_manifest(config, records, sensor_table) -> None:
    """Write the dataset manifest CSV to the output directory."""
    manifest_path = ensure_dir(config.output_dir) / "dataset_manifest.csv"
    rows = dataset_manifest(records, sensor_table)
    fieldnames = list(rows[0].keys())
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Manifest written to %s", manifest_path)


def run_training(args: argparse.Namespace) -> None:
    """End-to-end training run: load data, train, evaluate, export artefacts."""
    config = default_config(PROJECT_ROOT, args.mode)
    config.batch_size = args.batch_size
    config.max_epochs = args.epochs
    config.learning_rate = args.lr
    config.seed = args.seed
    config.num_workers = args.num_workers
    config.pretrained = not args.no_pretrained

    logger.info("Starting training | mode=%s seed=%d epochs=%d", args.mode, args.seed, args.epochs)
    seed_everything(config.seed)
    ensure_dir(config.output_dir)

    records = build_catalog(config.dataset_dir)
    logger.info("Catalogue: %d total images", len(records))
    records = stratified_split(records, val_ratio=config.val_ratio, test_ratio=config.test_ratio, seed=config.seed)
    sensor_table = generate_sensor_table(records, base_seed=config.seed)
    export_manifest(config, records, sensor_table)

    train_loader, val_loader, test_loader, train_records, _ = build_dataloaders(args.mode, config, records, sensor_table)
    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    logger.info("Device: %s", device)
    model = build_model(
        args.mode,
        pretrained=config.pretrained,
        dropout_sensor=config.dropout_sensor,
        dropout_f1=config.dropout_fusion_1,
        dropout_f2=config.dropout_fusion_2,
    ).to(device)

    positive_weight = class_weight(train_records)
    logger.info("Positive class weight: %.3f", positive_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], device=device))
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    training_report = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mode=args.mode,
        max_epochs=config.max_epochs,
        patience=config.patience,
        output_dir=config.output_dir,
    )
    logger.info("Training complete | epochs_ran=%d duration=%.1fs", training_report["epochs_ran"], training_report["duration_seconds"])

    val_report = predict(model, val_loader, device=device, mode=args.mode, threshold=0.5)
    threshold_report = select_best_threshold(val_report["labels"], val_report["probabilities"])
    selected_threshold = float(threshold_report["selected_threshold"])
    logger.info("Selected threshold: %.3f (val F1=%.4f)", selected_threshold, threshold_report["metrics_at_selected_threshold"]["f1"])

    test_report = predict(model, test_loader, device=device, mode=args.mode, threshold=selected_threshold)
    val_report["threshold_selection"] = threshold_report
    test_report["threshold_selection"] = threshold_report
    test_report["metrics_default_threshold"] = predict(model, test_loader, device=device, mode=args.mode, threshold=0.5)["metrics"]

    logger.info(
        "Test results | accuracy=%.4f recall=%.4f f1=%.4f auc_roc=%.4f",
        test_report["metrics"]["accuracy"],
        test_report["metrics"]["recall"],
        test_report["metrics"]["f1"],
        test_report["metrics"]["auc_roc"],
    )

    save_json(config.to_dict(), config.output_dir / "config.json")
    save_json(training_report, config.output_dir / "training_summary.json")
    save_json(val_report, config.output_dir / "val_metrics.json")
    save_json(test_report, config.output_dir / "test_metrics.json")
    logger.info("Artefacts saved to %s", config.output_dir)


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
