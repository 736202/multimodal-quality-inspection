from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random


LABEL_MAP = {
    "ok_front": 0,
    "def_front": 1,
}


@dataclass(slots=True)
class SampleRecord:
    """Lightweight descriptor for a single dataset sample.

    Attributes
    ----------
    sample_id:
        Unique identifier derived from the image filename stem.
    image_path:
        Absolute path to the JPEG image file.
    label:
        Binary class label — 0 for conforming parts, 1 for defective parts.
    split:
        Dataset partition: ``"train"``, ``"val"``, ``"test"``, or
        ``"unassigned"`` before splitting.
    """

    sample_id: str
    image_path: Path
    label: int
    split: str


def build_catalog(dataset_dir: Path) -> list[SampleRecord]:
    """Enumerate all labelled images and return an unsplit catalogue.

    The function expects the following directory layout::

        dataset_dir/
            ok_front/   # conforming parts
            def_front/  # defective parts

    Parameters
    ----------
    dataset_dir:
        Root of the casting dataset.

    Returns
    -------
    list[SampleRecord]
        Records sorted by sample_id, all with ``split="unassigned"``.

    Raises
    ------
    FileNotFoundError
        If a class subdirectory does not exist.
    RuntimeError
        If no JPEG images are found.
    """
    records: list[SampleRecord] = []
    for class_name, label in LABEL_MAP.items():
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for image_path in sorted(class_dir.glob("*.jpeg")):
            records.append(
                SampleRecord(
                    sample_id=image_path.stem,
                    image_path=image_path,
                    label=label,
                    split="unassigned",
                )
            )
    if not records:
        raise RuntimeError(f"No images found in {dataset_dir}")
    return records


def stratified_split(
    records: list[SampleRecord],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> list[SampleRecord]:
    """Assign each record to a train/val/test split, stratified by class.

    Stratification ensures that class proportions are preserved in every
    partition, which is important for imbalanced datasets.

    Parameters
    ----------
    records:
        Full catalogue returned by :func:`build_catalog`.
    val_ratio:
        Fraction of each class reserved for validation (e.g. ``0.15``).
    test_ratio:
        Fraction of each class reserved for testing (e.g. ``0.15``).
    seed:
        Random seed for reproducible shuffling.

    Returns
    -------
    list[SampleRecord]
        New record list with ``split`` field set, sorted by sample_id.
    """
    by_label: dict[int, list[SampleRecord]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record)

    rng = random.Random(seed)
    split_records: list[SampleRecord] = []

    for label_records in by_label.values():
        shuffled = label_records[:]
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_test = int(round(n_total * test_ratio))
        n_val = int(round(n_total * val_ratio))
        n_test = min(n_test, max(n_total - 2, 1))
        n_val = min(n_val, max(n_total - n_test - 1, 1))

        test_slice = shuffled[:n_test]
        val_slice = shuffled[n_test : n_test + n_val]
        train_slice = shuffled[n_test + n_val :]

        for record in train_slice:
            split_records.append(
                SampleRecord(record.sample_id, record.image_path, record.label, "train")
            )
        for record in val_slice:
            split_records.append(
                SampleRecord(record.sample_id, record.image_path, record.label, "val")
            )
        for record in test_slice:
            split_records.append(
                SampleRecord(record.sample_id, record.image_path, record.label, "test")
            )

    split_records.sort(key=lambda item: item.sample_id)
    return split_records
