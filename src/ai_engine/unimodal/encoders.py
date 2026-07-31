"""Unimodal Encoder implementations for Text and Video (MER-RULE-048, MER-RULE-074, MER-RULE-217..219).
"""

from typing import List, Union
import torch
import torch.nn as nn

from src.ai_engine.unimodal.contracts import BaseEncoder
from src.ai_engine.unimodal.registry import encoder_registry


@encoder_registry.register("mock_text_encoder")
class MockTextEncoder(BaseEncoder):
    """Mock text encoder returning native 768-dimensional sentence embeddings."""

    def __init__(self, native_dim: int = 768):
        super().__init__()
        self._native_dim = native_dim
        # Lightweight trainable projection to simulate real encoding parameters
        self._dummy_param = nn.Parameter(torch.randn(1, native_dim) * 0.01)

    def forward(self, inputs: Union[List[str], torch.Tensor]) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return inputs + self._dummy_param
        
        batch_size = len(inputs)
        # Deterministic generation based on sentence length for consistent mock outputs
        device = self._dummy_param.device
        features = torch.zeros(batch_size, self._native_dim, device=device)
        for i, text in enumerate(inputs):
            g = torch.Generator().manual_seed(len(text) + i)
            features[i] = torch.randn(self._native_dim, generator=g)
        return features + self._dummy_param

    @property
    def output_dim(self) -> int:
        return self._native_dim

    @property
    def modality(self) -> str:
        return "text"


@encoder_registry.register("mock_video_encoder")
class MockVideoEncoder(BaseEncoder):
    """Mock video encoder returning native 512-dimensional visual representations."""

    def __init__(self, native_dim: int = 512):
        super().__init__()
        self._native_dim = native_dim
        self._proj = nn.Linear(native_dim, native_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Expects inputs of shape [Batch, Native_Dim] or [Batch, Frames, Dim]
        if inputs.dim() == 3:
            inputs = inputs.mean(dim=1)  # Temporal pooling
        return self._proj(inputs)

    @property
    def output_dim(self) -> int:
        return self._native_dim

    @property
    def modality(self) -> str:
        return "video"
