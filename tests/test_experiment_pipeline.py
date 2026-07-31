"""System integration test executing end-to-end experiment pipeline (MER-RULE-164)."""

import unittest
import tempfile
import shutil
from pathlib import Path

from src.research.experiments.runner import ExperimentRunner


class TestExperimentPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_path = self.test_dir / "test_config.yaml"
        
        # Write minimal config for fast test execution
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write(f"""
project:
  name: "SystemTest"
  seed: 42
  output_dir: "{self.test_dir / 'outputs'}"

dataset:
  name: "synthetic_meld"
  data_dir: "data"
  batch_size: 4
  num_workers: 0
  num_samples: 16

model:
  fusion_dim: 256
  encoder:
    text:
      name: "mock_text_encoder"
      raw_dim: 768
      projection_type: "linear"
    video:
      name: "mock_video_encoder"
      raw_dim: 512
      projection_type: "linear"
  fusion:
    name: "concat_fusion"
    projection_dim: 256
  classifier:
    name: "mlp_classifier"
    num_classes: 7

training:
  epochs: 1
  learning_rate: 0.01
  device: "cpu"

evaluation:
  metrics: ["accuracy", "weighted_f1"]
""")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_full_experiment_pipeline(self):
        runner = ExperimentRunner(str(self.config_path))
        metrics = runner.run()

        self.assertIn("accuracy", metrics)
        self.assertIn("weighted_f1", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)

        # Check saved artifact files
        exp_dir = runner.exp_dir
        self.assertTrue((exp_dir / "config_snapshot.yaml").exists())
        self.assertTrue((exp_dir / "metrics.json").exists())
        self.assertTrue((exp_dir / "summary_report.txt").exists())
        self.assertTrue((exp_dir / "checkpoints" / "latest_checkpoint.pt").exists())


if __name__ == "__main__":
    unittest.main()
