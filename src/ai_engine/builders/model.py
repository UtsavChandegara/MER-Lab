"""MERModel Composition Container (MER-RULE-043, MER-RULE-083..085).

Explicitly wires: Raw Inputs -> Encoders -> Projections -> Fusion -> Classifier -> Emotion Logits.
"""

from typing import Dict, Any
import torch
import torch.nn as nn

from src.foundation.exceptions import ContractError, ModelError
from src.ai_engine.unimodal.contracts import BaseEncoder, BaseProjection
from src.ai_engine.multimodal.contracts import BaseFusion, BaseClassifier


class MERModel(nn.Module):
    """Unified Multimodal Emotion Recognition Model Container."""

    def __init__(
        self,
        encoders: Dict[str, BaseEncoder],
        projections: Dict[str, BaseProjection],
        fusion: BaseFusion,
        classifier: BaseClassifier,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.projections = nn.ModuleDict(projections)
        self.fusion = fusion
        self.classifier = classifier

        self._validate_contracts()

    def _validate_contracts(self) -> None:
        """Validates dimension promises across all component interfaces (MER-RULE-071..073)."""
        fusion_input_dim = self.fusion.input_dim

        for modality, encoder in self.encoders.items():
            if modality not in self.projections:
                raise ContractError(
                    f"Missing projection layer for encoder modality '{modality}'.",
                    hint=f"Register a projection layer for modality '{modality}'.",
                )
            proj = self.projections[modality]
            
            # Encoder -> Projection dimension match
            if encoder.output_dim != proj.input_dim:
                raise ContractError(
                    f"Dimension mismatch between '{modality}' encoder output ({encoder.output_dim}) "
                    f"and projection input ({proj.input_dim}).",
                    hint="Ensure projection input_dim matches encoder output_dim.",
                )

            # Projection -> Fusion dimension match (MER-RULE-033)
            if proj.output_dim != fusion_input_dim:
                raise ContractError(
                    f"Dimension mismatch between '{modality}' projection output ({proj.output_dim}) "
                    f"and fusion input ({fusion_input_dim}).",
                    hint="Ensure projection output_dim matches standard fusion input dimension.",
                )

        # Fusion -> Classifier dimension match
        if self.fusion.output_dim != self.classifier.input_dim:
            raise ContractError(
                f"Dimension mismatch between fusion output ({self.fusion.output_dim}) "
                f"and classifier input ({self.classifier.input_dim}).",
                hint="Ensure classifier input_dim matches fusion output_dim.",
            )

    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Full forward processing pass enforcing architectural layer order (MER-RULE-043)."""
        aligned_features: Dict[str, torch.Tensor] = {}

        # 1. Encoders & 2. Projections
        for modality, encoder in self.encoders.items():
            if modality not in inputs:
                raise ModelError(
                    f"Input dictionary missing required modality data for '{modality}'.",
                    hint=f"Provided input keys are: {list(inputs.keys())}",
                )
            raw_features = encoder(inputs[modality])
            aligned_features[modality] = self.projections[modality](raw_features)

        # 3. Fusion
        fused_representation = self.fusion(aligned_features)

        # 4. Classifier
        logits = self.classifier(fused_representation)
        return logits
