"""Loss Functions for Emotion Classification (MER-RULE-053).
"""

import torch
import torch.nn as nn
from src.foundation.registry import Registry

loss_registry = Registry("loss")


@loss_registry.register("cross_entropy")
@loss_registry.register("ce")
class EmotionCrossEntropyLoss(nn.Module):
    """Standard Cross-Entropy Loss for emotion classification."""

    def __init__(self, weight: torch.Tensor = None):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)
