"""Comprehensive Research Metric Suite for Multimodal Emotion & Intent Recognition (H1–H10).

Implements all 9 metric categories:
1. Classification Performance (Accuracy, Macro-F1, Weighted-F1, Micro-F1)
2. Class-Level Metrics (Precision, Recall, F1, Support)
3. Error Analysis (Confusion Matrix)
4. Multitask Metrics (Joint Accuracy, Combined F1, Delta F1)
5. Modality Contribution Metrics (D_T, D_A, D_V drops)
6. Fusion Metrics (Delta vs baseline fusion)
7. Efficiency Metrics (Params, Latency, Throughput, Memory, Model Size)
8. Dynamic-Gating Metrics (Mean weights, std, per-class distribution)
9. Statistical Reliability Metrics (Mean +/- std, 95% Confidence Intervals)
"""

import time
import math
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Classification, Class-Level & Error Analysis Metrics
# ---------------------------------------------------------------------------

def evaluate_predictions(y_true: List[int], y_pred: List[int], class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Computes full classification performance, class-level metrics, and confusion matrix.
    
    Args:
        y_true: Ground truth integer class labels.
        y_pred: Predicted integer class labels.
        class_names: Optional human-readable class names.
        
    Returns:
        Dict with Accuracy, Macro-F1, Weighted-F1, Micro-F1, per-class metrics, support, confusion matrix.
    """
    if not y_true or not y_pred:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "micro_f1": 0.0,
            "per_class": {},
            "confusion_matrix": [],
        }

    total = len(y_true)
    correct = sum(1 for gt, p in zip(y_true, y_pred) if gt == p)
    accuracy = correct / total if total > 0 else 0.0

    # Determine unique classes
    all_classes = sorted(list(set(y_true) | set(y_pred)))
    max_c = max(all_classes) if all_classes else 0
    matrix = [[0] * (max_c + 1) for _ in range(max_c + 1)]
    class_counts: Dict[int, int] = {}

    for gt, p in zip(y_true, y_pred):
        matrix[gt][p] += 1
        class_counts[gt] = class_counts.get(gt, 0) + 1

    per_class_metrics: Dict[str, Dict[str, Any]] = {}
    macro_f1_sum = 0.0
    weighted_f1_sum = 0.0
    total_tp = 0

    for c in all_classes:
        name = class_names[c] if class_names and c < len(class_names) else str(c)
        tp = matrix[c][c]
        fp = sum(matrix[other][c] for other in range(max_c + 1) if other != c)
        fn = sum(matrix[c][other] for other in range(max_c + 1) if other != c)
        support = class_counts.get(c, 0)
        total_tp += tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class_metrics[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }

        macro_f1_sum += f1
        weighted_f1_sum += f1 * support

    macro_f1 = macro_f1_sum / len(all_classes) if all_classes else 0.0
    weighted_f1 = weighted_f1_sum / total if total > 0 else 0.0
    micro_f1 = total_tp / total if total > 0 else 0.0  # Micro-F1 equals accuracy in single-label multi-class

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "micro_f1": float(micro_f1),
        "per_class": per_class_metrics,
        "per_class_f1": {k: v["f1"] for k, v in per_class_metrics.items()},
        "confusion_matrix": matrix,
    }


# ---------------------------------------------------------------------------
# 2. Multitask Metrics (Emotion + Communicative Intent)
# ---------------------------------------------------------------------------

def evaluate_multitask_predictions(
    y_true_emo: List[int],
    y_pred_emo: List[int],
    y_true_intent: List[int],
    y_pred_intent: List[int],
    emo_names: Optional[List[str]] = None,
    intent_names: Optional[List[str]] = None,
    single_task_emo_f1: Optional[float] = None,
    single_task_intent_f1: Optional[float] = None,
) -> Dict[str, Any]:
    """Computes Multitask Metrics including Joint Accuracy and Delta improvements (H2).
    
    Joint Accuracy: #(Emotion_correct AND Intent_correct) / N
    """
    assert len(y_true_emo) == len(y_pred_emo) == len(y_true_intent) == len(y_pred_intent)
    total = len(y_true_emo)
    if total == 0:
        return {"joint_accuracy": 0.0, "combined_f1": 0.0}

    # Evaluate individual tasks
    emo_res = evaluate_predictions(y_true_emo, y_pred_emo, class_names=emo_names)
    intent_res = evaluate_predictions(y_true_intent, y_pred_intent, class_names=intent_names)

    # 19. Joint Accuracy: both simultaneously correct
    joint_correct = sum(
        1 for e_gt, e_p, i_gt, i_p in zip(y_true_emo, y_pred_emo, y_true_intent, y_pred_intent)
        if (e_gt == e_p and i_gt == i_p)
    )
    joint_accuracy = joint_correct / total

    # 20. Combined F1 (mean of both task macro F1s)
    combined_f1 = (emo_res["macro_f1"] + intent_res["macro_f1"]) / 2.0

    # 21 & 22. Multitask Improvements over Single-task
    delta_f1_emo = (emo_res["macro_f1"] - single_task_emo_f1) if single_task_emo_f1 is not None else None
    delta_f1_intent = (intent_res["macro_f1"] - single_task_intent_f1) if single_task_intent_f1 is not None else None

    return {
        "joint_accuracy": float(joint_accuracy),
        "combined_f1": float(combined_f1),
        "delta_f1_emotion": delta_f1_emo,
        "delta_f1_intent": delta_f1_intent,
        "emotion": emo_res,
        "intent": intent_res,
    }


# ---------------------------------------------------------------------------
# 3. Modality Contribution & Fusion Comparative Metrics
# ---------------------------------------------------------------------------

def compute_modality_contributions(
    f1_tav: float,
    f1_av: float,
    f1_tv: float,
    f1_ta: float,
) -> Dict[str, float]:
    """Computes performance drops after removing individual modalities.
    
    D_T = F1_TAV - F1_AV
    D_A = F1_TAV - F1_TV
    D_V = F1_TAV - F1_TA
    """
    return {
        "D_Text (drop without text)": float(f1_tav - f1_av),
        "D_Audio (drop without audio)": float(f1_tav - f1_tv),
        "D_Video (drop without video)": float(f1_tav - f1_ta),
    }


def compute_fusion_deltas(proposed_f1: float, baseline_f1s: Dict[str, float]) -> Dict[str, float]:
    """Calculates Delta F1 of proposed fusion against baseline fusions (Concat, Average, etc.)."""
    return {f"delta_vs_{name}": float(proposed_f1 - base_f1) for name, base_f1 in baseline_f1s.items()}


# ---------------------------------------------------------------------------
# 4. Dynamic Gating Metrics (Interpretability Analysis)
# ---------------------------------------------------------------------------

def compute_gating_metrics(
    gate_weights: torch.Tensor,
    modality_keys: List[str],
    labels: Optional[List[int]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes gating statistics across modalities and per emotion class (H3, H5).
    
    Args:
        gate_weights: Tensor of shape [N, Num_Modalities, 1] or [N, Num_Modalities].
        modality_keys: List of modality names in corresponding order, e.g. ['audio', 'text', 'video'].
        labels: Optional ground-truth class labels for per-class breakdown.
        class_names: Optional class names.
    """
    if gate_weights.dim() == 3:
        gate_weights = gate_weights.squeeze(-1)  # [N, Num_Modalities]

    gate_weights = gate_weights.detach().cpu()
    mean_weights = gate_weights.mean(dim=0).tolist()
    std_weights = gate_weights.std(dim=0).tolist()

    overall_summary = {
        mod: {"mean": float(m), "std": float(s)}
        for mod, m, s in zip(modality_keys, mean_weights, std_weights)
    }

    per_class_summary: Dict[str, Dict[str, float]] = {}
    if labels is not None:
        labels_t = torch.tensor(labels)
        unique_classes = sorted(list(set(labels)))
        for c in unique_classes:
            mask = (labels_t == c)
            if mask.sum() > 0:
                cls_mean = gate_weights[mask].mean(dim=0).tolist()
                c_name = class_names[c] if class_names and c < len(class_names) else str(c)
                per_class_summary[c_name] = {
                    mod: float(val) for mod, val in zip(modality_keys, cls_mean)
                }

    return {
        "overall_gating": overall_summary,
        "per_class_gating": per_class_summary,
    }


# ---------------------------------------------------------------------------
# 5. Efficiency Metrics (Mandatory for Lightweight Hypothesis H5 / H7)
# ---------------------------------------------------------------------------

def profile_model_efficiency(
    model: nn.Module,
    sample_inputs: Dict[str, Any],
    device: torch.device,
    num_warmup: int = 5,
    num_runs: int = 50,
) -> Dict[str, Any]:
    """Profiles model parameters, memory, inference latency, and throughput.
    
    Measures:
    - Total & Trainable Parameters
    - Checkpoint parameter size (MB)
    - Inference latency (ms / sample)
    - Throughput (samples / sec)
    - Peak GPU memory allocated (MB)
    """
    model.eval()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = (total_params * 4) / (1024 * 1024)

    # Move sample inputs to target device
    device_inputs = {}
    batch_size = 1
    for k, v in sample_inputs.items():
        if isinstance(v, torch.Tensor):
            device_inputs[k] = v.to(device)
            batch_size = v.shape[0]
        else:
            device_inputs[k] = v

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(device_inputs)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(device_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
    total_elapsed = time.time() - start_time

    total_samples = batch_size * num_runs
    latency_ms_per_sample = (total_elapsed / total_samples) * 1000.0
    throughput_samples_per_sec = total_samples / total_elapsed if total_elapsed > 0 else 0.0

    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameter_size_mb": float(param_size_mb),
        "latency_ms_per_sample": float(latency_ms_per_sample),
        "throughput_samples_per_sec": float(throughput_samples_per_sec),
        "peak_gpu_memory_mb": float(peak_memory_mb),
    }


# ---------------------------------------------------------------------------
# 6. Statistical Reliability Metrics (Multi-Seed Aggregation)
# ---------------------------------------------------------------------------

def compute_statistical_reliability(values: List[float], confidence_level: float = 0.95) -> Dict[str, float]:
    """Computes mean, standard deviation, and confidence intervals across random seed trials.
    
    Example output format: 61.4 +/- 0.8
    """
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "formatted": "0.0 +/- 0.0"}
    if n == 1:
        return {"mean": values[0], "std": 0.0, "ci_lower": values[0], "ci_upper": values[0], "formatted": f"{values[0]:.4f} +/- 0.0000"}

    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    standard_error = std / math.sqrt(n)

    # Normal approximation for CI (z=1.96 for 95%)
    z = 1.96 if confidence_level == 0.95 else 2.576
    ci_half = z * standard_error

    return {
        "mean": float(mean),
        "std": float(std),
        "ci_lower": float(mean - ci_half),
        "ci_upper": float(mean + ci_half),
        "formatted": f"{mean:.4f} +/- {std:.4f}",
    }

