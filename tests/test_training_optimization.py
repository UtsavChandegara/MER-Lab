"""Unit tests for Trimodal Training Optimization and Visual Diagnostics (H1-H8)."""

import unittest
from pathlib import Path
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.ai_engine.builders.model import MERModel
from src.training.trainer import Trainer
from src.training.losses import EmotionCrossEntropyLoss, compute_class_weights
from src.research.evaluation.visualizations import (
    plot_overfitting_underfitting_dynamics,
    plot_multimodal_learning_curves,
)


class DummyEncoder(nn.Module):
    def forward(self, x):
        return x

    @property
    def native_dim(self):
        return 16

    @property
    def output_dim(self):
        return 16


class DummyProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        return self.fc(x)

    @property
    def input_dim(self):
        return 16

    @property
    def output_dim(self):
        return 16


class DummyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_gating_weights = None

    def forward(self, features):
        return features["text"]

    @property
    def input_dim(self):
        return 16

    @property
    def output_dim(self):
        return 16


from src.ai_engine.multimodal.classifiers import LinearClassifier


class TestTrainingOptimization(unittest.TestCase):
    """Verifies best checkpoint restoration, gradient clipping, scheduler, and loss weighting."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.encoders = nn.ModuleDict({"text": DummyEncoder()})
        self.projections = nn.ModuleDict({"text": DummyProjection()})
        self.fusion = DummyFusion()
        self.classifier = LinearClassifier(input_dim=16, num_classes=7)
        self.model = MERModel(self.encoders, self.projections, self.fusion, self.classifier)

        # Synthetic dataset with 7 classes
        x = torch.randn(40, 16)
        y = torch.randint(0, 7, (40,))
        ds = TensorDataset(x, y)

        def collate_fn(batch):
            inputs = {"text": torch.stack([item[0] for item in batch])}
            targets = torch.tensor([item[1] for item in batch])
            return inputs, targets

        self.train_loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
        self.val_loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    def test_class_weights_computation(self):
        labels = torch.tensor([0, 0, 0, 0, 1, 2, 3, 4, 5, 6])
        weights = compute_class_weights(labels, num_classes=7)
        self.assertEqual(len(weights), 7)
        # Class 0 is most frequent, so its weight should be less than minority classes
        self.assertLess(weights[0].item(), weights[1].item())

    def test_trainer_best_model_restoration_and_history(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        criterion = EmotionCrossEntropyLoss(label_smoothing=0.05)

        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            scheduler=scheduler,
            max_grad_norm=1.0,
            early_stopping_patience=3,
            restore_best=True,
        )

        history = trainer.fit(self.train_loader, self.val_loader, epochs=4)
        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)
        self.assertIn("val_weighted_f1", history)
        self.assertIn("best_epoch", history)
        self.assertGreaterEqual(history["best_epoch"], 1)

    def test_overfitting_underfitting_plot_generation(self):
        history = {
            "train_loss": [1.5, 0.9, 0.4, 0.1, 0.05],
            "val_loss": [1.6, 1.1, 1.0, 1.4, 2.1],
            "train_weighted_f1": [0.3, 0.6, 0.8, 0.95, 0.99],
            "val_weighted_f1": [0.28, 0.55, 0.65, 0.52, 0.40],
            "best_epoch": 3,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            p = plot_overfitting_underfitting_dynamics(history, out_dir)
            self.assertTrue(p.exists())
            self.assertTrue((out_dir / "fig7_overfitting_underfitting_dynamics.pdf").exists())

    def test_multimodal_learning_curves_plot_generation(self):
        histories = {
            "Text-Only": {"val_loss": [1.8, 1.5, 1.3], "val_weighted_f1": [0.4, 0.5, 0.55]},
            "Trimodal DGCA": {"val_loss": [1.7, 1.2, 1.0], "val_weighted_f1": [0.45, 0.58, 0.68]},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            p = plot_multimodal_learning_curves(histories, out_dir)
            self.assertTrue(p.exists())
            self.assertTrue((out_dir / "fig8_multimodal_learning_curves.pdf").exists())


if __name__ == "__main__":
    unittest.main()
