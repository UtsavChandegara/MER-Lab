"""Emotion Classifier Implementations (MER-RULE-051, MER-RULE-224).

Maps joint multimodal representations to emotion classification logits.
"""

import torch
import torch.nn as nn

from src.ai_engine.multimodal.contracts import BaseClassifier
from src.ai_engine.multimodal.registry import classifier_registry


@classifier_registry.register("mlp")
@classifier_registry.register("mlp_classifier")
class MLPClassifier(BaseClassifier):
    """MLP Emotion Classifier with LayerNorm, Dropout, and non-linear activations."""

    def __init__(self, input_dim: int = 256, num_classes: int = 7, hidden_dim: int = 128, dropout: float = 0.1, **kwargs):
        super().__init__()
        self._input_dim = input_dim
        self._num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        return self.classifier(fused_features)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def num_classes(self) -> int:
        return self._num_classes


@classifier_registry.register("linear")
@classifier_registry.register("linear_classifier")
class LinearClassifier(BaseClassifier):
    """Linear Emotion Classifier mapping fused features directly to emotion logits."""

    def __init__(self, input_dim: int = 256, num_classes: int = 7, **kwargs):
        super().__init__()
        self._input_dim = input_dim
        self._num_classes = num_classes
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        return self.classifier(fused_features)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def num_classes(self) -> int:
        return self._num_classes
