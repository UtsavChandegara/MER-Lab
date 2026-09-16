"""Multimodal Fusion Implementations (MER-RULE-033, MER-RULE-050, MER-RULE-080..082, MER-RULE-223).

Receives standardized modality representations and returns a unified joint feature vector of standard dimension (256).
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn

from src.ai_engine.multimodal.contracts import BaseFusion
from src.ai_engine.multimodal.registry import fusion_registry


@fusion_registry.register("concat")
@fusion_registry.register("concat_fusion")
class ConcatFusion(BaseFusion):
    """Concatenation Fusion: Concatenates modalities and projects to standard dimension (256)."""

    def __init__(self, projection_dim: int = 256, num_modalities: int = 3):
        super().__init__()
        self._projection_dim = projection_dim
        self._num_modalities = num_modalities
        
        # Maps [Batch, projection_dim * num_modalities] -> [Batch, projection_dim]
        self.fusion_layer = nn.Sequential(
            nn.Linear(projection_dim * num_modalities, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self._dynamic_layers = nn.ModuleDict()

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Sort modality keys for deterministic order
        tensors = [features[k] for k in sorted(features.keys())]
        concatenated = torch.cat(tensors, dim=-1)
        num_m = len(tensors)
        
        if num_m == self._num_modalities:
            return self.fusion_layer(concatenated)
        
        # Dynamically support different number of modalities (e.g. during ablations)
        k = str(num_m)
        if k not in self._dynamic_layers:
            device = concatenated.device
            layer = nn.Sequential(
                nn.Linear(self._projection_dim * num_m, self._projection_dim),
                nn.LayerNorm(self._projection_dim),
                nn.GELU(),
            ).to(device)
            self._dynamic_layers[k] = layer
        return self._dynamic_layers[k](concatenated)

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

@fusion_registry.register("average")
@fusion_registry.register("average_fusion")
class AverageFusion(BaseFusion):
    """Average Fusion: Baseline that computes element-wise mean across all modality representations."""

    def __init__(self, projection_dim: int = 256):
        super().__init__()
        self._projection_dim = projection_dim
        self.norm = nn.LayerNorm(projection_dim)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        tensors = [features[k] for k in sorted(features.keys())]
        stacked = torch.stack(tensors, dim=0)  # [Num_Modalities, Batch, Dim]
        averaged = stacked.mean(dim=0)          # [Batch, Dim]
        return self.norm(averaged)

    @property
    def input_dim(self) -> int:
        return self._projection_dim

    @property
    def output_dim(self) -> int:
        return self._projection_dim


@fusion_registry.register("dynamic_gated_cross_attention")
@fusion_registry.register("dgca_fusion")
class DynamicGatedCrossAttentionFusion(BaseFusion):
    """Proposed Research Architecture: Dynamic Gated Cross-Attention Fusion (H2, H3, H4).
    
    Features:
    1. Cross-modal Transformer attention capturing inter-modality dependencies.
    2. Input-conditioned dynamic gating network computing per-sample modality weights (H3).
    3. Modality dropout during training for missing-modality robustness (H4).
    4. Caches last_gating_weights for empirical interpretability and visualization.
    """

    def __init__(
        self,
        projection_dim: int = 256,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        modality_dropout: float = 0.15,
    ):
        super().__init__()
        self._projection_dim = projection_dim
        self.modality_dropout = modality_dropout

        # Cross-modal Transformer Self-Attention Layer
        self.attn = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(projection_dim)
        self.norm2 = nn.LayerNorm(projection_dim)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(projection_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, projection_dim),
        )

        # Dynamic Modality Gating Network (H3)
        self.gate_net = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.GELU(),
            nn.Linear(projection_dim // 2, 1),
        )

        # Final joint projection
        self.out_proj = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

        # Stored gating weights from the most recent forward pass (for analysis/plots)
        self.last_gating_weights: Optional[torch.Tensor] = None
        self.last_modality_keys: Optional[List[str]] = None

    def _apply_modality_dropout(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Randomly drops modalities during training to improve missing-modality robustness (H4)."""
        if not self.training or self.modality_dropout <= 0.0 or len(tensors) <= 1:
            return tensors

        device = tensors[0].device
        # Keep at least one modality active per batch item
        keep_mask = torch.rand(len(tensors), device=device) > self.modality_dropout
        if not keep_mask.any():
            keep_mask[torch.randint(0, len(tensors), (1,)).item()] = True

        out = []
        for i, t in enumerate(tensors):
            if keep_mask[i]:
                out.append(t)
            else:
                out.append(torch.zeros_like(t))
        return out

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        sorted_keys = sorted(features.keys())
        self.last_modality_keys = sorted_keys
        raw_tensors = [features[k] for k in sorted_keys]

        # Apply modality dropout during training
        processed_tensors = self._apply_modality_dropout(raw_tensors)

        # Sequence of modalities: [Batch, Num_Modalities, Dim]
        seq = torch.stack(processed_tensors, dim=1)

        # 1. Cross-modal Attention
        attn_out, _ = self.attn(seq, seq, seq)
        h = self.norm1(seq + attn_out)

        # 2. Feed-Forward Block
        h = self.norm2(h + self.ffn(h))

        # 3. Dynamic Gating: Compute attention weight per modality
        gate_logits = self.gate_net(h)  # [Batch, Num_Modalities, 1]
        gate_weights = torch.softmax(gate_logits, dim=1)  # [Batch, Num_Modalities, 1]
        self.last_gating_weights = gate_weights.detach().cpu()

        # 4. Gated Aggregation
        fused = torch.sum(h * gate_weights, dim=1)  # [Batch, Dim]

        # 5. Output Projection
        return self.out_proj(fused)

    @property
    def input_dim(self) -> int:
        return self._projection_dim

    @property
    def output_dim(self) -> int:
        return self._projection_dim

