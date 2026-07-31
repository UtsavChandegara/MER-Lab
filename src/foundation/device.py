"""Device Manager for MER-Lab (MER-RULE-212).

Detects and assigns computing devices (CUDA, MPS, CPU) for PyTorch modules.
"""

import torch
from src.foundation.logging import get_logger

logger = get_logger("MERLab.Foundation.Device")


def get_device(requested_device: str = "auto") -> torch.device:
    """Determines PyTorch compute device based on requested parameter and hardware availability.
    
    Args:
        requested_device: 'auto', 'cuda', 'mps', or 'cpu'.
        
    Returns:
        torch.device: Resolved PyTorch device object.
    """
    req = requested_device.lower()
    
    if req == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif req == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif req == "cpu":
        device = torch.device("cpu")
    elif req == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        logger.warning(
            f"Requested device '{requested_device}' not available. Falling back to CPU."
        )
        device = torch.device("cpu")

    logger.info(f"Using compute device: {device}")
    return device
