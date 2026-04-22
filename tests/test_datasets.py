from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from mqi.data.catalog import SampleRecord
from mqi.data.datasets import (
    SensorScaler,
    fit_sensor_scaler,
    split_records,
    class_weight,
    dataset_manifest,
)
from mqi.data.synthetic_sensors import SENSOR_COLUMNS, generate_sensor_table


def _fake_records(n: int = 10) -> list[SampleRecord]:
    records = []
    for i in range(n):
        records.append(
            SampleRecord(
                sample_id=f"s{i:04d}",
                image_path=Path(f"/fake/{i}.jpeg"),
                label=i % 2,
                split="train",
            )
        )
    return records


def _fake_sensor_table(records: list[SampleRecord]) -> dict:
    return generate_sensor_table(records, base_seed=42)


# ── SensorScaler ──────────────────────────────────────────────────────────────

def test_sensor_scaler_transform_shape():
    records = _fake_records()
    table = _fake_sensor_table(records)
    scaler = fit_sensor_scaler(records, table)
    values = table[records[0].sample_id]
    arr = scaler.transform(values)
    assert arr.shape == (len(SENSOR_COLUMNS),)


def test_sensor_scaler_mean_zero_on_train():
    records = _fake_records(100)
    table = _fake_sensor_table(records)
    scaler = fit_sensor_scaler(records, table)
    transformed = np.stack([scaler.transform(table[r.sample_id]) for r in records])
    # Column means should be very close to 0
    assert np.abs(transformed.mean(axis=0)).max() < 0.05


def test_sensor_scaler_dtype():
    records = _fake_records()
    table = _fake_sensor_table(records)
    scaler = fit_sensor_scaler(records, table)
    arr = scaler.transform(table[records[0].sample_id])
    assert arr.dtype == np.float32


# ── split_records ─────────────────────────────────────────────────────────────

def test_split_records_partition():
    records = []
    for i, s in enumerate(["train"] * 5 + ["val"] * 2 + ["test"] * 3):
        records.append(SampleRecord(f"s{i}", Path(f"/f/{i}.jpeg"), 0, s))
    train, val, test = split_records(records)
    assert len(train) == 5
    assert len(val) == 2
    assert len(test) == 3


# ── class_weight ──────────────────────────────────────────────────────────────

def test_class_weight_balanced():
    records = [SampleRecord(f"s{i}", Path(f"/f/{i}.jpeg"), i % 2, "train") for i in range(10)]
    w = class_weight(records)
    assert w == pytest.approx(1.0)


def test_class_weight_imbalanced():
    records = (
        [SampleRecord(f"s{i}", Path(f"/f/{i}.jpeg"), 0, "train") for i in range(8)] +
        [SampleRecord(f"s{i}", Path(f"/f/{i}.jpeg"), 1, "train") for i in range(8, 10)]
    )
    w = class_weight(records)
    assert w == pytest.approx(4.0)


# ── dataset_manifest ──────────────────────────────────────────────────────────

def test_dataset_manifest_columns():
    records = _fake_records(5)
    table = _fake_sensor_table(records)
    manifest = dataset_manifest(records, table)
    assert len(manifest) == 5
    for row in manifest:
        assert "sample_id" in row
        assert "label" in row
        assert "split" in row
        for col in SENSOR_COLUMNS:
            assert col in row
