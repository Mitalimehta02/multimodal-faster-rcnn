"""utils/seed_utils.py — Reproducibility Helpers"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, PyTorch (CPU + CUDA) to ensure
    fully reproducible experiments.

    Args:
        seed: integer seed value (default 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # for multi-GPU
    # Make CUDA operations deterministic (may reduce speed slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)
