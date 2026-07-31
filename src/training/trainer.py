"""Trainer Lifecycle Engine (MER-RULE-053, MER-RULE-225..226).

Manages model optimization, batch processing, validation runs, and metric collection.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.foundation.logging import get_logger
from src.ai_engine.builders.model import MERModel
from src.ai_engine.dataset.contracts import MultimodalBatch
from src.training.callbacks import BaseCallback
from src.research.evaluation.metrics import evaluate_predictions

logger = get_logger("MERLab.Training.Trainer")


class Trainer:
    """Orchestrates optimization and validation loops for MERModel instances."""

    def __init__(
        self,
        model: MERModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        callbacks: Optional[List[BaseCallback]] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.callbacks = callbacks or []

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Executes one training epoch across dataloader batches."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, MultimodalBatch):
                batch = batch.to(self.device)
                inputs = batch.inputs
                targets = batch.labels
            else:
                inputs, targets = batch
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Evaluates model performance over validation/test dataset split."""
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, MultimodalBatch):
                batch = batch.to(self.device)
                inputs = batch.inputs
                targets = batch.labels
            else:
                inputs, targets = batch
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                targets = targets.to(self.device)

            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            
            total_loss += loss.item()
            num_batches += 1

        val_loss = total_loss / max(num_batches, 1)
        metrics = evaluate_predictions(all_targets, all_preds)
        metrics["loss"] = val_loss
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> Dict[str, Any]:
        """Runs complete training process across specified number of epochs."""
        logger.info(f"Starting training for {epochs} epochs on device: {self.device}")

        for cb in self.callbacks:
            cb.on_train_start(self)

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_weighted_f1": []}

        for epoch in range(1, epochs + 1):
            for cb in self.callbacks:
                cb.on_epoch_start(epoch, self)

            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            metrics = {
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "val_macro_f1": val_metrics["macro_f1"],
            }

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_weighted_f1"].append(val_metrics["weighted_f1"])

            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | Val Weighted F1: {val_metrics['weighted_f1']:.4f}"
            )

            for cb in self.callbacks:
                cb.on_epoch_end(epoch, metrics, self)

        for cb in self.callbacks:
            cb.on_train_end(self)

        logger.info("Training complete.")
        return history
