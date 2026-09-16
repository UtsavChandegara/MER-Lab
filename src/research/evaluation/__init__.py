"""Evaluation package exports for MER-Lab (MER-RULE-052)."""

from src.research.evaluation.metrics import (
    evaluate_predictions,
    evaluate_multitask_predictions,
    compute_modality_contributions,
    compute_fusion_deltas,
    compute_gating_metrics,
    profile_model_efficiency,
    compute_statistical_reliability,
)

__all__ = [
    "evaluate_predictions",
    "evaluate_multitask_predictions",
    "compute_modality_contributions",
    "compute_fusion_deltas",
    "compute_gating_metrics",
    "profile_model_efficiency",
    "compute_statistical_reliability",
]
