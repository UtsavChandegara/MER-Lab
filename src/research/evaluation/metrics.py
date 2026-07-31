"""Evaluation Metrics and Report Generators (MER-RULE-017, MER-RULE-052, MER-RULE-184, MER-RULE-227).

Computes accuracy, macro F1, weighted F1, per-class metrics, and confusion matrix data without modifying model state.
"""

from typing import List, Dict, Any, Tuple


def evaluate_predictions(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """Computes comprehensive emotion classification metrics.
    
    Args:
        y_true: List of ground-truth integer class labels.
        y_pred: List of predicted integer class labels.
        
    Returns:
        Dict[str, Any]: Metrics dictionary containing accuracy, macro_f1, weighted_f1, per_class F1, and confusion matrix.
    """
    if not y_true or not y_pred:
        return {"accuracy": 0.0, "weighted_f1": 0.0, "macro_f1": 0.0}

    total = len(y_true)
    correct = sum(1 for gt, p in zip(y_true, y_pred) if gt == p)
    accuracy = correct / total if total > 0 else 0.0

    # Determine unique classes
    classes = sorted(list(set(y_true) | set(y_pred)))
    
    per_class_precision: Dict[int, float] = {}
    per_class_recall: Dict[int, float] = {}
    per_class_f1: Dict[int, float] = {}
    class_counts: Dict[int, int] = {}

    # Compute confusion matrix dimensions
    max_c = max(classes) if classes else 0
    matrix = [[0] * (max_c + 1) for _ in range(max_c + 1)]

    for gt, p in zip(y_true, y_pred):
        matrix[gt][p] += 1
        class_counts[gt] = class_counts.get(gt, 0) + 1

    macro_f1_sum = 0.0
    weighted_f1_sum = 0.0

    for c in classes:
        tp = matrix[c][c]
        fp = sum(matrix[other][c] for other in range(max_c + 1) if other != c)
        fn = sum(matrix[c][other] for other in range(max_c + 1) if other != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class_precision[c] = precision
        per_class_recall[c] = recall
        per_class_f1[c] = f1

        macro_f1_sum += f1
        count = class_counts.get(c, 0)
        weighted_f1_sum += f1 * count

    macro_f1 = macro_f1_sum / len(classes) if classes else 0.0
    weighted_f1 = weighted_f1_sum / total if total > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "per_class_f1": {str(k): float(v) for k, v in per_class_f1.items()},
        "confusion_matrix": matrix,
    }
