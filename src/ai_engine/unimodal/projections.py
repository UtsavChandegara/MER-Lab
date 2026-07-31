"""Projection implementations standardizing native modalities into common fusion dimensions (MER-RULE-033, MER-RULE-049, MER-RULE-075..079).
"""

import torch
import torch.nn as nn

from src.ai_engine.unimodal.contracts import BaseProjection
from src.ai_engine.unimodal.registry import projection_registry


@projection_registry.register("linear")
@projection_registry.register("linear_projection")
class LinearProjection(BaseProjection):
    """Linear layer mapping native feature dimension to standardized fusion dimension (e.g. 256)."""

    def __init__(self, input_dim: int, output_dim: int = 256):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim


@projection_registry.register("mlp")
@projection_registry.register("mlp_projection")
class MLPProjection(BaseProjection):
    """Two-layer MLP projection with non-linear activation and dropout (MER-RULE-078)."""

    def __init__(self, input_dim: int, output_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim


@projection_registry.register("identity")
class IdentityProjection(BaseProjection):
    """Identity projection when feature dimensions are already aligned."""

    def __init__(self, input_dim: int, output_dim: int = 256):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim
