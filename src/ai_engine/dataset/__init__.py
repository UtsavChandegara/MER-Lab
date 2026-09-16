"""Dataset package exports for MER-Lab (MER-RULE-047)."""

from src.ai_engine.dataset.contracts import BaseDataset, MultimodalSample, MultimodalBatch
from src.ai_engine.dataset.registry import dataset_registry
from src.ai_engine.dataset.components import (
    SyntheticMELDDataset,
    MELDFeatureDataset,
    collate_multimodal_batch,
)

__all__ = [
    "BaseDataset",
    "MultimodalSample",
    "MultimodalBatch",
    "dataset_registry",
    "SyntheticMELDDataset",
    "MELDFeatureDataset",
    "collate_multimodal_batch",
]
