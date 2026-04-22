from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mqi.data.catalog import build_catalog, stratified_split, SampleRecord


def _make_fake_dataset(base: Path) -> Path:
    """Create a minimal fake dataset directory with two JPEG files per class."""
    for class_name in ("ok_front", "def_front"):
        class_dir = base / class_name
        class_dir.mkdir(parents=True)
        for i in range(4):
            (class_dir / f"img_{i:04d}.jpeg").touch()
    return base


def test_build_catalog_returns_all_images():
    with tempfile.TemporaryDirectory() as tmp:
        dataset_dir = _make_fake_dataset(Path(tmp) / "dataset")
        records = build_catalog(dataset_dir)
    assert len(records) == 8


def test_build_catalog_labels_correct():
    with tempfile.TemporaryDirectory() as tmp:
        dataset_dir = _make_fake_dataset(Path(tmp) / "dataset")
        records = build_catalog(dataset_dir)
    labels = sorted(set(r.label for r in records))
    assert labels == [0, 1]


def test_build_catalog_missing_dir_raises():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            build_catalog(Path(tmp) / "nonexistent")


def test_build_catalog_empty_dir_raises():
    with tempfile.TemporaryDirectory() as tmp:
        dataset_dir = Path(tmp) / "dataset"
        (dataset_dir / "ok_front").mkdir(parents=True)
        (dataset_dir / "def_front").mkdir(parents=True)
        with pytest.raises(RuntimeError):
            build_catalog(dataset_dir)


def test_stratified_split_covers_all_records():
    with tempfile.TemporaryDirectory() as tmp:
        dataset_dir = _make_fake_dataset(Path(tmp) / "dataset")
        records = build_catalog(dataset_dir)
    split = stratified_split(records, val_ratio=0.25, test_ratio=0.25, seed=0)
    assert len(split) == len(records)


def test_stratified_split_partitions_exclusive():
    with tempfile.TemporaryDirectory() as tmp:
        dataset_dir = _make_fake_dataset(Path(tmp) / "dataset")
        records = build_catalog(dataset_dir)
    split = stratified_split(records, val_ratio=0.25, test_ratio=0.25, seed=0)
    splits = [r.split for r in split]
    assert set(splits) <= {"train", "val", "test"}


def test_stratified_split_reproducible():
    with tempfile.TemporaryDirectory() as tmp:
        dataset_dir = _make_fake_dataset(Path(tmp) / "dataset")
        records1 = build_catalog(dataset_dir)
        records2 = build_catalog(dataset_dir)
    split1 = stratified_split(records1, val_ratio=0.25, test_ratio=0.25, seed=42)
    split2 = stratified_split(records2, val_ratio=0.25, test_ratio=0.25, seed=42)
    assert [r.sample_id for r in split1] == [r.sample_id for r in split2]
    assert [r.split for r in split1] == [r.split for r in split2]


def test_stratified_split_different_seeds_differ():
    with tempfile.TemporaryDirectory() as tmp:
        dataset_dir = _make_fake_dataset(Path(tmp) / "dataset")
        records = build_catalog(dataset_dir)
    split_a = stratified_split(records, val_ratio=0.25, test_ratio=0.25, seed=0)
    split_b = stratified_split(records, val_ratio=0.25, test_ratio=0.25, seed=99)
    assert [r.split for r in split_a] != [r.split for r in split_b]
