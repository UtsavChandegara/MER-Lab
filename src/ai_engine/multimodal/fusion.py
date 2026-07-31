"""Multimodal Fusion Implementations (MER-RULE-033, MER-RULE-050, MER-RULE-080..082, MER-RULE-223).

Receives standardized modality representations and returns a unified joint feature vector of standard dimension (256).
"""

from typing import Dict
import torch
import torch.nn as nn

from src.ai_engine.multimodal.contracts import BaseFusion
from src.ai_engine.multimodal.registry import fusion_registry


@fusion_registry.register("concat")
@fusion_registry.register("concat_fusion")
class ConcatFusion(BaseFusion):
    """Concatenation Fusion: Concatenates modalities and projects to standard dimension (256)."""

    def __init__(self, projection_dim: int = 256, num_modalities: int = 2):
        super().__init__()
        self._projection_dim = projection_dim
        self._num_modalities = num_modalities
        
        # Maps [Batch, projection_dim * num_modalities] -> [Batch, projection_dim]
        self.fusion_layer = nn.Sequential(
            nn.Linear(projection_dim * num_modalities, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Sort modality keys for deterministic order
        tensors = [features[k] for k in sorted(features.keys())]
        concatenated = torch.cat(tensors, dim=-1)
        return self.fusion_layer(concatenated)

    @property
    def input_dim(self) -> int:
        return self._projection_dim

    @property
    def output_dim(self) -> int:
        return self._projection_dim


@fusion_registry.register("attention")
@fusion_registry.register("attention_fusion")
class AttentionFusion(BaseFusion):
    """Attention-based Fusion: Treats modalities as tokens in a Transformer self-attention block."""

    def __init__(self, projection_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self._projection_dim = projection_dim
        self.attn = nn.MultiheadAttention(embed_dim=projection_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(projection_dim)
        self.out_proj = nn.Linear(projection_dim, projection_dim)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Stack modalities: [Batch, Num_Modalities, Dim]
        tensors = [features[k] for k in sorted(features.keys())]
        seq = torch.stack(tensors, dim=1)
        
        attn_out, _ = self.attn(seq, seq, seq)
        fused = self.norm(seq + attn_out)
        
        # Pooled joint representation
        pooled = fused.mean(dim=1)
        return self.out_proj(pooled)

    @property
    def input_dim(self) -> int:
        return self._projection_dim

    @property
    def output_dim(self) -> int:
        return self._projection_dim


@fusion_registry.register("gated")
@fusion_registry.register("gated_fusion")
class GatedFusion(BaseFusion):
    """Gated Fusion: Learns elementwise modality importance weights."""

    def __init__(self, projection_dim: int = 256):
        super().__init__()
        self._projection_dim = projection_dim
        self.gate = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.Sigmoid()
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        sorted_keys = sorted(features.keys())
        feat1, feat2 = features[sorted_keys[0]], features[sorted_keys[1]]
        
        g = self.gate(torch.cat([feat1, feat2], dim=-1))
        fused = g * feat1 + (1 - g) * feat2
        return fused

    @property
    def input_dim(self) -> int:
        return self._projection_dim

    @property
    def output_dim(self) -> int:
        return self._projection_dim
