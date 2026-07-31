"""Unimodal package exports for MER-Lab (MER-RULE-048..049)."""

from src.ai_engine.unimodal.contracts import BaseEncoder, BaseProjection
from src.ai_engine.unimodal.registry import encoder_registry, projection_registry
from src.ai_engine.unimodal.encoders import MockTextEncoder, MockVideoEncoder
from src.ai_engine.unimodal.projections import LinearProjection, MLPProjection, IdentityProjection

__all__ = [
    "BaseEncoder",
    "BaseProjection",
    "encoder_registry",
    "projection_registry",
    "MockTextEncoder",
    "MockVideoEncoder",
    "LinearProjection",
    "MLPProjection",
    "IdentityProjection",
]
