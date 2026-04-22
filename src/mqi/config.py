from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(slots=True)
class TrainingConfig:
    """Centralised configuration for a single model training run.

    All hyperparameters and path settings are stored here so they can be
    exported alongside model artefacts for full reproducibility.
    """

    project_root: Path
    dataset_dir: Path
    output_dir: Path
    image_size: int = 224
    batch_size: int = 32
    max_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout_fusion_1: float = 0.4
    dropout_fusion_2: float = 0.3
    dropout_sensor: float = 0.3
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    num_workers: int = 0
    patience: int = 8
    pretrained: bool = True
    freeze_backbone_epochs: int = 3

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation of the configuration."""
        payload = asdict(self)
        payload["project_root"] = str(self.project_root)
        payload["dataset_dir"] = str(self.dataset_dir)
        payload["output_dir"] = str(self.output_dir)
        return payload


def default_config(project_root: Path, mode: str) -> TrainingConfig:
    """Build a default configuration for the given training mode.

    Parameters
    ----------
    project_root:
        Absolute path to the repository root.
    mode:
        One of ``"image"``, ``"sensor"``, or ``"multimodal"``.

    Returns
    -------
    TrainingConfig
        Configuration instance with output_dir set to ``outputs/<mode>``.
    """
    return TrainingConfig(
        project_root=project_root,
        dataset_dir=project_root / "casting_512x512",
        output_dir=project_root / "outputs" / mode,
    )
