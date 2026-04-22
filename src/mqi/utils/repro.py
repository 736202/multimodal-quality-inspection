from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set all relevant random seeds for reproducible experiments.

    Seeds Python's ``random``, NumPy, PyTorch (CPU and CUDA) so that
    data shuffling, weight initialisation, and dropout masks are deterministic
    across runs with the same seed.

    Parameters
    ----------
    seed:
        Integer seed value, typically defined in :class:`~mqi.config.TrainingConfig`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    """Create a directory (and any missing parents) if it does not exist.

    Parameters
    ----------
    path:
        Target directory path.

    Returns
    -------
    Path
        The same path, for convenient chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: dict, path: Path) -> None:
    """Write a dictionary to a JSON file with 2-space indentation.

    Parameters
    ----------
    data:
        JSON-serialisable dictionary.
    path:
        Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
