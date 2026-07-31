"""Seed Manager for MER-Lab (MER-RULE-016, MER-RULE-212).

Ensures complete experiment reproducibility across Python, NumPy, PyTorch, and CUDA.
"""

import random
import numpy as np
import torch

from src.foundation.logging import get_logger

logger = get_logger("MERLab.Foundation.Seed")


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Sets random seeds for Python, NumPy, and PyTorch.
    
    Args:
        seed: Integer seed value.
        deterministic: If True, sets PyTorch CUDNN to deterministic mode.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to: {seed} (deterministic={deterministic})")
