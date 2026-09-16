"""Unit tests for Multi-Dataset Expansion: IEMOCAP & CMU-MOSEI (MER-RULE-047, MER-RULE-069).

Verifies that IEMOCAP and CMU-MOSEI datasets operate seamlessly side-by-side with MELD
without removing or affecting any MELD dataset code.
"""

import unittest
import torch
from torch.utils.data import DataLoader

from src.ai_engine.dataset.registry import dataset_registry
from src.ai_engine.dataset.components import (
    MELDFeatureDataset,
    IEMOCAPFeatureDataset,
    CMUMOSEIFeatureDataset,
    collate_multimodal_batch,
)
from src.ai_engine.builders.builder import ModelBuilder
from src.foundation.config import Config


class TestMultiDatasetExpansion(unittest.TestCase):
    """Tests multi-dataset plug-and-play functionality."""

    def test_meld_dataset_remains_intact(self):
        """Verifies that MELD dataset is completely unaffected and has 7 classes."""
        dataset = MELDFeatureDataset(num_samples=20, seed=42)
        self.assertEqual(len(dataset), 20)
        self.assertEqual(dataset.num_classes, 7)
        self.assertEqual(len(dataset.EMOTIONS), 7)
        self.assertIn("neutral", dataset.EMOTIONS)
        self.assertIn("anger", dataset.EMOTIONS)

        sample = dataset[0]
        self.assertEqual(sample.text.shape, (768,))
        self.assertEqual(sample.audio.shape, (768,))
        self.assertEqual(sample.video.shape, (512,))
        self.assertIn(sample.label, range(7))

    def test_iemocap_dataset_functionality(self):
        """Verifies that IEMOCAP dataset instantiates with 4 classes and correct shapes."""
        dataset = IEMOCAPFeatureDataset(num_samples=25, seed=42)
        self.assertEqual(len(dataset), 25)
        self.assertEqual(dataset.num_classes, 4)
        self.assertEqual(dataset.EMOTIONS, ["neutral", "happy", "sad", "angry"])

        sample = dataset[0]
        self.assertEqual(sample.text.shape, (768,))
        self.assertEqual(sample.audio.shape, (768,))
        self.assertEqual(sample.video.shape, (512,))
        self.assertIn(sample.label, range(4))
        self.assertEqual(sample.metadata.get("dataset"), "iemocap")

        # Test batch collation
        loader = DataLoader([dataset[i] for i in range(4)], batch_size=4, collate_fn=collate_multimodal_batch)
        batch = next(iter(loader))
        self.assertEqual(batch.inputs["text"].shape, (4, 768))
        self.assertEqual(batch.inputs["audio"].shape, (4, 768))
        self.assertEqual(batch.inputs["video"].shape, (4, 512))
        self.assertEqual(batch.labels.shape, (4,))

    def test_mosei_dataset_functionality(self):
        """Verifies that CMU-MOSEI dataset instantiates with 6 classes and correct shapes."""
        dataset = CMUMOSEIFeatureDataset(num_samples=30, seed=42)
        self.assertEqual(len(dataset), 30)
        self.assertEqual(dataset.num_classes, 6)
        self.assertEqual(dataset.EMOTIONS, ["happy", "sad", "anger", "fear", "disgust", "surprise"])

        sample = dataset[0]
        self.assertEqual(sample.text.shape, (768,))
        self.assertEqual(sample.audio.shape, (768,))
        self.assertEqual(sample.video.shape, (512,))
        self.assertIn(sample.label, range(6))
        self.assertEqual(sample.metadata.get("dataset"), "mosei")

    def test_dataset_registry_contains_all(self):
        """Verifies that all three datasets are registered simultaneously."""
        registered = dataset_registry.list_modules()
        self.assertIn("meld_features", registered)
        self.assertIn("iemocap_features", registered)
        self.assertIn("mosei_features", registered)

    def test_iemocap_model_forward_pass(self):
        """Verifies that ModelBuilder configures a 4-class model for IEMOCAP."""
        cfg = Config.from_yaml("configs/iemocap_trimodal.yaml")
        model = ModelBuilder.build_model(cfg)
        self.assertEqual(model.num_classes, 4)

        dummy_batch = {
            "text": torch.randn(2, 768),
            "audio": torch.randn(2, 768),
            "video": torch.randn(2, 512),
        }
        logits = model(dummy_batch)
        self.assertEqual(logits.shape, (2, 4))

    def test_mosei_model_forward_pass(self):
        """Verifies that ModelBuilder configures a 6-class model for CMU-MOSEI."""
        cfg = Config.from_yaml("configs/mosei_trimodal.yaml")
        model = ModelBuilder.build_model(cfg)
        self.assertEqual(model.num_classes, 6)

        dummy_batch = {
            "text": torch.randn(3, 768),
            "audio": torch.randn(3, 768),
            "video": torch.randn(3, 512),
        }
        logits = model(dummy_batch)
        self.assertEqual(logits.shape, (3, 6))


    def test_benchmark_suite_h6_ablation_iemocap(self):
        """Verifies that BenchmarkSuite runs H6 ablation on IEMOCAP (4 classes) without class dimension mismatch."""
        from src.research.experiments.benchmark_runner import BenchmarkSuite
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            suite = BenchmarkSuite(base_config_path="configs/iemocap_trimodal.yaml", output_dir=tmpdir)
            suite.base_config._data["training"]["epochs"] = 1
            suite.base_config._data["dataset"]["num_samples"] = 20

            # Run H6 Ablation Study (which includes Ablation 3: Linear Classifier)
            h6_res = suite.run_h6_ablation_study()
            self.assertIn("Full Proposed Model", h6_res)
            self.assertIn("Ablation 3: Linear Classifier (No MLP)", h6_res)
            self.assertGreater(h6_res["Ablation 3: Linear Classifier (No MLP)"]["accuracy"], 0.0)

            # Run H7 Per-Class Breakdown
            h7_res = suite.run_h7_per_class_breakdown()
            self.assertEqual(len(h7_res), 4)
            self.assertIn("neutral", h7_res)
            self.assertIn("happy", h7_res)


if __name__ == "__main__":
    unittest.main()
