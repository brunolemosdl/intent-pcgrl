from __future__ import annotations

import os
import random

import numpy as np


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Args:
        seed: Integer seed applied to all supported backends.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
