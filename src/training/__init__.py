"""Training package exports for MER-Lab (MER-RULE-053)."""

from src.training.trainer import Trainer
from src.training.callbacks import BaseCallback, CheckpointCallback
from src.training.losses import EmotionCrossEntropyLoss, loss_registry

__all__ = [
    "Trainer",
    "BaseCallback",
    "CheckpointCallback",
    "EmotionCrossEntropyLoss",
    "loss_registry",
]
