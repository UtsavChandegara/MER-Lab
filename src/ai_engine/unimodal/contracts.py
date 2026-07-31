"""Unimodal Encoder and Projection Contracts (MER-RULE-048..049, MER-RULE-074..079).

Enforces separation between native unimodal encoders and dimension standardization projections.
"""

from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Abstract interface for native unimodal feature encoders (MER-RULE-048, MER-RULE-074)."""

    @abstractmethod
    def forward(self, inputs: Any) -> torch.Tensor:
        """Encodes raw input modality into native feature tensor [Batch, Raw_Dim]."""
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Returns the native feature dimension produced by this encoder."""
        pass

    @property
    @abstractmethod
    def modality(self) -> str:
        """Returns the modality identifier (e.g. 'text', 'video', 'audio')."""
        pass


class BaseProjection(nn.Module, ABC):
    """Abstract interface for projection layers standardizing feature dimensions (MER-RULE-049, MER-RULE-077)."""

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Projects native feature tensor [Batch, Native_Dim] to standard dimension [Batch, Target_Dim]."""
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Returns expected input feature dimension."""
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Returns standard output feature dimension (e.g. 256)."""
        pass
