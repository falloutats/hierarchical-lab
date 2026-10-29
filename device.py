"""Device + reproducibility helpers.

Mirrors the ~/prototypical-toy idioms: MPS-first device pick, seed torch+numpy
together. Kept tiny and dependency-light on purpose.
"""

from __future__ import annotations

import numpy as np
import torch


def pick_device(choice: str = "auto") -> torch.device:
    """MPS -> CUDA -> CPU. `choice` other than 'auto' is honored verbatim."""
    if choice != "auto":
        return torch.device(choice)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 0) -> None:
    """Seed torch + numpy so runs are comparable across stages."""
    torch.manual_seed(seed)
    np.random.seed(seed)

