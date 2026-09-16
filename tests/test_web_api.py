"""Unit tests for MER-Lab Web Studio Backend API (MER-RULE-047, MER-RULE-071).

Tests API endpoint functions for dataset querying, model querying, training status,
and real-time multimodal inference predictions.
"""

import unittest
from pathlib import Path

from app import (
    app,
    get_info,
    get_datasets,
    get_models,
    get_training_status,
    predict_emotion,
    PredictionRequest,
    TrainingRequest,
)


class TestWebAPI(unittest.TestCase):
    """Tests Web Studio API endpoints directly."""

    def test_get_info_endpoint(self):
        """Verifies hardware and readiness status."""
        info = get_info()
        self.assertEqual(info["status"], "ready")
        self.assertIn("device", info)
        self.assertIn("cuda_available", info)
        self.assertIn("gpu_name", info)
        self.assertGreaterEqual(info["num_datasets"], 3)

    def test_get_datasets_endpoint(self):
        """Verifies dataset metadata returns MELD, IEMOCAP, and CMU-MOSEI."""
        datasets = get_datasets()
        self.assertIn("meld", datasets)
        self.assertIn("iemocap", datasets)
        self.assertIn("mosei", datasets)

        # Check classes
        self.assertEqual(len(datasets["meld"]["classes"]), 7)
        self.assertEqual(len(datasets["iemocap"]["classes"]), 4)
        self.assertEqual(len(datasets["mosei"]["classes"]), 6)

    def test_get_models_endpoint(self):
        """Verifies fusion strategies return proposed DGCA and baselines."""
        models = get_models()
        model_ids = [m["id"] for m in models]
        self.assertIn("dynamic_gated_cross_attention", model_ids)
        self.assertIn("concat_fusion", model_ids)
        self.assertIn("average_fusion", model_ids)
        self.assertIn("attention_fusion", model_ids)

    def test_get_training_status_endpoint(self):
        """Verifies training status structure."""
        status = get_training_status()
        self.assertIn("is_training", status)
        self.assertIn("status", status)
        self.assertIn("train_loss", status)
        self.assertIn("val_weighted_f1", status)

    def test_predict_emotion_meld(self):
        """Verifies real-time prediction and dynamic gating extraction for MELD."""
        req = PredictionRequest(
            dataset="meld",
            text_utterance="Oh, that is just brilliant!",
            audio_pitch_energy=0.85,
            visual_affect_intensity=0.70,
        )
        res = predict_emotion(req)
        self.assertEqual(res["dataset"], "meld")
        self.assertIn("predicted_emotion", res)
        self.assertIn("confidence", res)
        self.assertIn("gating_weights", res)

        # Check gating weights sum to approximately 1.0
        gw = res["gating_weights"]
        self.assertIn("text", gw)
        self.assertIn("audio", gw)
        self.assertIn("video", gw)
        total_gw = gw["text"] + gw["audio"] + gw["video"]
        self.assertAlmostEqual(total_gw, 1.0, places=1)

        # Check probabilities
        probs = res["class_probabilities"]
        self.assertEqual(len(probs), 7)
        self.assertTrue(probs[0]["is_top"])

    def test_predict_emotion_iemocap(self):
        """Verifies prediction for 4-class IEMOCAP."""
        req = PredictionRequest(
            dataset="iemocap",
            text_utterance="Why are you doing this?!",
            audio_pitch_energy=0.9,
            visual_affect_intensity=0.8,
        )
        res = predict_emotion(req)
        self.assertEqual(res["dataset"], "iemocap")
        self.assertEqual(len(res["class_probabilities"]), 4)

    def test_predict_emotion_mosei(self):
        """Verifies prediction for 6-class CMU-MOSEI."""
        req = PredictionRequest(
            dataset="mosei",
            text_utterance="This is the greatest experience!",
            audio_pitch_energy=0.7,
            visual_affect_intensity=0.9,
        )
        res = predict_emotion(req)
        self.assertEqual(res["dataset"], "mosei")
        self.assertEqual(len(res["class_probabilities"]), 6)

    def test_predict_with_dropped_modality(self):
        """Verifies prediction when video sensor drops out."""
        req = PredictionRequest(
            dataset="meld",
            text_utterance="Testing sensor failure.",
            audio_pitch_energy=0.5,
            visual_affect_intensity=0.5,
            modality_mask=["video"],
        )
        res = predict_emotion(req)
        self.assertIn("predicted_emotion", res)
        self.assertIn("gating_weights", res)


if __name__ == "__main__":
    unittest.main()
