"""Loss Functions for Emotion Classification (MER-RULE-053).
"""

from typing import Optional
import torch
import torch.nn as nn
from src.foundation.registry import Registry

loss_registry = Registry("loss")


@loss_registry.register("cross_entropy")
@loss_registry.register("ce")
class EmotionCrossEntropyLoss(nn.Module):
    """Standard Cross-Entropy Loss for emotion classification with optional class weights and label smoothing."""

    def __init__(self, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)


def compute_class_weights(labels: torch.Tensor, num_classes: int = 7) -> torch.Tensor:
    """Computes balanced class weights inverse to class frequency to mitigate class imbalance."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    total = labels.numel()
    # Avoid division by zero for classes with 0 count
    weights = total / (num_classes * torch.clamp(counts, min=1.0))
    # Normalize so mean weight is 1.0
    weights = weights / weights.mean()
    return weights

