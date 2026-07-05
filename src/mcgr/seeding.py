"""Deterministic seeding across every RNG the project touches.

numpy/torch are seeded only if importable, so this works before the heavy
deps land (they arrive with the model phases).
"""

import os
import random


def seed_everything(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
