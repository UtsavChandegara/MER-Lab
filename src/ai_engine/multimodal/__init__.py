"""Multimodal package exports for MER-Lab (MER-RULE-050..051)."""

from src.ai_engine.multimodal.contracts import BaseFusion, BaseClassifier
from src.ai_engine.multimodal.registry import fusion_registry, classifier_registry
from src.ai_engine.multimodal.fusion import ConcatFusion, AttentionFusion, GatedFusion
from src.ai_engine.multimodal.classifiers import MLPClassifier, LinearClassifier

__all__ = [
    "BaseFusion",
    "BaseClassifier",
    "fusion_registry",
    "classifier_registry",
    "ConcatFusion",
    "AttentionFusion",
    "GatedFusion",
    "MLPClassifier",
    "LinearClassifier",
]
