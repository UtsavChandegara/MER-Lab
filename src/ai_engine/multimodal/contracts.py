"""Multimodal Fusion and Classifier Contracts (MER-RULE-050..051, MER-RULE-080..082).

Enforces strict separation between unimodal feature fusion and emotion classification.
"""

from abc import ABC, abstractmethod
from typing import Dict
import torch
import torch.nn as nn


class BaseFusion(nn.Module, ABC):
    """Abstract interface for Multimodal Fusion strategies (MER-RULE-050, MER-RULE-080).
    
    Fusion receives a dictionary mapping modality names to standard dimension feature tensors (e.g. 256),
    and outputs a combined joint feature representation tensor [Batch, 256].
    """

    @abstractmethod
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combines standardized multimodal features into a unified joint representation."""
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Returns the per-modality feature dimension."""
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Returns the joint multimodal feature dimension (MER-RULE-033, standard: 256)."""
        pass


class BaseClassifier(nn.Module, ABC):
    """Abstract interface for Emotion Classifiers (MER-RULE-051)."""

    @abstractmethod
    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """Maps joint multimodal representation [Batch, Fusion_Dim] to emotion logits [Batch, Num_Classes]."""
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Returns the expected joint feature dimension."""
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Returns the number of target emotion categories."""
        pass
