from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from mqi.config import TrainingConfig
from mqi.data.catalog import SampleRecord
from mqi.data.synthetic_sensors import SENSOR_COLUMNS


@dataclass(slots=True)
class SensorScaler:
    """Z-score normaliser fitted on training data only.

    Attributes
    ----------
    mean:
        Per-column mean computed on the training split.
    std:
        Per-column standard deviation (with epsilon for numerical stability).
    """

    mean: dict[str, float]
    std: dict[str, float]

    def transform(self, values: dict[str, float]) -> np.ndarray:
        """Apply standardisation and return a float32 NumPy array.

        Parameters
        ----------
        values:
            Raw sensor readings keyed by column name.

        Returns
        -------
        np.ndarray
            Shape ``(n_sensors,)`` normalised to zero mean and unit variance.
        """
        return np.array(
            [
                (values[column] - self.mean[column]) / self.std[column]
                for column in SENSOR_COLUMNS
            ],
            dtype=np.float32,
        )


def fit_sensor_scaler(
    records: list[SampleRecord],
    sensor_table: dict[str, dict[str, float]],
) -> SensorScaler:
    """Compute per-column statistics from the training split and return a scaler.

    Only training records should be passed to prevent data leakage into the
    validation and test sets.

    Parameters
    ----------
    records:
        Training records only.
    sensor_table:
        Full sensor table (all splits); only training samples will be read.

    Returns
    -------
    SensorScaler
        Fitted scaler ready to transform any split.
    """
    stats_mean: dict[str, float] = {}
    stats_std: dict[str, float] = {}
    for column in SENSOR_COLUMNS:
        values = np.array([sensor_table[item.sample_id][column] for item in records], dtype=np.float32)
        stats_mean[column] = float(values.mean())
        stats_std[column] = float(values.std() + 1e-6)
    return SensorScaler(stats_mean, stats_std)


def build_image_transform(train: bool, image_size: int) -> Callable:
    """Build a torchvision transform pipeline for image preprocessing.

    Training transforms include data augmentation (flips, rotation, colour
    jitter, Gaussian blur).  Evaluation transforms only resize and normalise.
    Both pipelines convert grayscale inputs to 3-channel tensors and apply
    ImageNet statistics normalisation, which is required by the pretrained
    ResNet18 backbone.

    Parameters
    ----------
    train:
        Whether to include data-augmentation operations.
    image_size:
        Target spatial resolution (both height and width).

    Returns
    -------
    Callable
        A ``torchvision.transforms.Compose`` pipeline.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


class ImageDataset(Dataset):
    """PyTorch Dataset for image-only samples.

    Each item is a dict with keys ``"image"`` (3×H×W float tensor),
    ``"label"`` (float scalar), and ``"sample_id"`` (str).
    """

    def __init__(self, records: list[SampleRecord], transform: Callable) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            tensor = self.transform(image)
        return {
            "image": tensor,
            "label": torch.tensor(record.label, dtype=torch.float32),
            "sample_id": record.sample_id,
        }


class SensorDataset(Dataset):
    """PyTorch Dataset for sensor-only (tabular) samples.

    Each item is a dict with keys ``"sensors"`` (float32 vector of length 6),
    ``"label"``, and ``"sample_id"``.
    """

    def __init__(
        self,
        records: list[SampleRecord],
        sensor_table: dict[str, dict[str, float]],
        scaler: SensorScaler,
    ) -> None:
        self.records = records
        self.sensor_table = sensor_table
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        sensors = self.scaler.transform(self.sensor_table[record.sample_id])
        return {
            "sensors": torch.tensor(sensors, dtype=torch.float32),
            "label": torch.tensor(record.label, dtype=torch.float32),
            "sample_id": record.sample_id,
        }


class MultimodalDataset(Dataset):
    """PyTorch Dataset combining image and sensor modalities.

    Each item is a dict with keys ``"image"``, ``"sensors"``, ``"label"``,
    and ``"sample_id"``.
    """

    def __init__(
        self,
        records: list[SampleRecord],
        sensor_table: dict[str, dict[str, float]],
        scaler: SensorScaler,
        transform: Callable,
    ) -> None:
        self.records = records
        self.sensor_table = sensor_table
        self.scaler = scaler
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            image_tensor = self.transform(image)
        sensor_tensor = self.scaler.transform(self.sensor_table[record.sample_id])
        return {
            "image": image_tensor,
            "sensors": torch.tensor(sensor_tensor, dtype=torch.float32),
            "label": torch.tensor(record.label, dtype=torch.float32),
            "sample_id": record.sample_id,
        }


def split_records(records: list[SampleRecord]) -> tuple[list[SampleRecord], list[SampleRecord], list[SampleRecord]]:
    """Partition a flat record list into (train, val, test) sublists.

    Parameters
    ----------
    records:
        Records with the ``split`` field already assigned.

    Returns
    -------
    tuple
        ``(train_records, val_records, test_records)``.
    """
    train_records = [item for item in records if item.split == "train"]
    val_records = [item for item in records if item.split == "val"]
    test_records = [item for item in records if item.split == "test"]
    return train_records, val_records, test_records


def class_weight(train_records: list[SampleRecord]) -> float:
    """Compute the positive-class weight for weighted BCE loss.

    Returns ``n_negatives / n_positives`` so that the minority class
    receives proportionally higher gradient signal.

    Parameters
    ----------
    train_records:
        Training records only (must not include val/test).

    Returns
    -------
    float
        Positive-class weight ≥ 1.
    """
    positives = sum(item.label for item in train_records)
    negatives = len(train_records) - positives
    return negatives / max(positives, 1)


def dataset_manifest(records: list[SampleRecord], sensor_table: dict[str, dict[str, float]]) -> list[dict]:
    """Build a full audit trail combining image metadata and sensor values.

    Parameters
    ----------
    records:
        All records across all splits.
    sensor_table:
        Synthetic sensor readings keyed by sample_id.

    Returns
    -------
    list[dict]
        One row per sample with columns: sample_id, image_path, label, split,
        and one column per sensor variable.
    """
    manifest = []
    for record in records:
        row = {
            "sample_id": record.sample_id,
            "image_path": str(record.image_path),
            "label": record.label,
            "split": record.split,
        }
        row.update(sensor_table[record.sample_id])
        manifest.append(row)
    return manifest
