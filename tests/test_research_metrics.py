"""Unit tests for the Comprehensive Research Metric Suite (H1–H10)."""

import unittest
import torch
import torch.nn as nn

from src.research.evaluation.metrics import (
    evaluate_predictions,
    evaluate_multitask_predictions,
    compute_modality_contributions,
    compute_fusion_deltas,
    compute_gating_metrics,
    profile_model_efficiency,
    compute_statistical_reliability,
)


class TestResearchMetrics(unittest.TestCase):
    """Verifies all 9 categories of research metrics."""

    def test_classification_and_class_level_metrics(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 2, 2]
        names = ["neutral", "surprise", "fear"]

        res = evaluate_predictions(y_true, y_pred, class_names=names)
        self.assertIn("accuracy", res)
        self.assertIn("macro_f1", res)
        self.assertIn("weighted_f1", res)
        self.assertIn("micro_f1", res)
        self.assertIn("per_class", res)
        self.assertIn("confusion_matrix", res)

        # Check per-class details
        self.assertIn("neutral", res["per_class"])
        self.assertEqual(res["per_class"]["neutral"]["precision"], 1.0)
        self.assertEqual(res["per_class"]["neutral"]["recall"], 1.0)
        self.assertEqual(res["per_class"]["neutral"]["support"], 2)

    def test_multitask_metrics(self):
        y_true_emo = [0, 1, 2, 0]
        y_pred_emo = [0, 1, 1, 0]  # sample 0, 1, 3 correct (3/4)
        
        y_true_intent = [1, 1, 0, 0]
        y_pred_intent = [1, 0, 0, 0]  # sample 0, 2, 3 correct (3/4)

        # Simultaneously correct: sample 0 and sample 3 -> 2/4 = 0.50
        res = evaluate_multitask_predictions(
            y_true_emo, y_pred_emo,
            y_true_intent, y_pred_intent,
            single_task_emo_f1=0.60,
            single_task_intent_f1=0.60,
        )

        self.assertEqual(res["joint_accuracy"], 0.50)
        self.assertIn("combined_f1", res)
        self.assertIsNotNone(res["delta_f1_emotion"])
        self.assertIsNotNone(res["delta_f1_intent"])

    def test_modality_contributions(self):
        # D_T = F1_TAV - F1_AV, etc.
        drops = compute_modality_contributions(
            f1_tav=0.85,
            f1_av=0.75,  # Drop of 0.10 without text
            f1_tv=0.80,  # Drop of 0.05 without audio
            f1_ta=0.82,  # Drop of 0.03 without video
        )
        self.assertAlmostEqual(drops["D_Text (drop without text)"], 0.10, places=4)
        self.assertAlmostEqual(drops["D_Audio (drop without audio)"], 0.05, places=4)
        self.assertAlmostEqual(drops["D_Video (drop without video)"], 0.03, places=4)

    def test_fusion_deltas(self):
        deltas = compute_fusion_deltas(
            proposed_f1=0.85,
            baseline_f1s={"concat": 0.80, "average": 0.78}
        )
        self.assertAlmostEqual(deltas["delta_vs_concat"], 0.05, places=4)
        self.assertAlmostEqual(deltas["delta_vs_average"], 0.07, places=4)

    def test_dynamic_gating_metrics(self):
        # Batch of 4 samples, 3 modalities (Audio, Text, Video)
        gate_weights = torch.tensor([
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.2, 0.6, 0.2],
            [0.1, 0.7, 0.2],
        ])
        keys = ["audio", "text", "video"]
        labels = [0, 0, 1, 1]
        cnames = ["neutral", "joy"]

        g_res = compute_gating_metrics(gate_weights, keys, labels=labels, class_names=cnames)
        self.assertIn("overall_gating", g_res)
        self.assertIn("per_class_gating", g_res)
        self.assertIn("neutral", g_res["per_class_gating"])
        self.assertIn("joy", g_res["per_class_gating"])
        # Neutral class has audio mean (0.5 + 0.4)/2 = 0.45
        self.assertAlmostEqual(g_res["per_class_gating"]["neutral"]["audio"], 0.45, places=3)

    def test_profile_model_efficiency(self):
        dummy_model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
        dummy_input = {"features": torch.randn(4, 256)}
        # Wrap in small module to accept dict
        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m(x["features"])

        wrapped = Wrapper(dummy_model)
        eff = profile_model_efficiency(wrapped, dummy_input, torch.device("cpu"), num_warmup=2, num_runs=5)

        self.assertGreater(eff["total_parameters"], 0)
        self.assertEqual(eff["total_parameters"], eff["trainable_parameters"])
        self.assertGreater(eff["latency_ms_per_sample"], 0.0)
        self.assertGreater(eff["throughput_samples_per_sec"], 0.0)

    def test_statistical_reliability(self):
        seed_runs = [0.84, 0.85, 0.83, 0.86, 0.84]
        stat = compute_statistical_reliability(seed_runs)
        self.assertAlmostEqual(stat["mean"], 0.844, places=3)
        self.assertGreater(stat["std"], 0.0)
        self.assertIn("+/-", stat["formatted"])


if __name__ == "__main__":
    unittest.main()
