"""Trainer Lifecycle Engine (MER-RULE-053, MER-RULE-225..226).

Manages model optimization, batch processing, validation runs, and metric collection.
"""

import copy
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.foundation.logging import get_logger
from src.ai_engine.builders.model import MERModel
from src.ai_engine.dataset.contracts import MultimodalBatch
from src.training.callbacks import BaseCallback

logger = get_logger("MERLab.Training.Trainer")


class Trainer:
    """Orchestrates optimization, validation loops, and learning dynamics for MERModel instances."""

    def __init__(
        self,
        model: MERModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        callbacks: Optional[List[BaseCallback]] = None,
        scheduler: Optional[Any] = None,
        max_grad_norm: float = 1.0,
        early_stopping_patience: Optional[int] = None,
        restore_best: bool = True,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.callbacks = callbacks or []
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience
        self.restore_best = restore_best

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Executes one training epoch across dataloader batches with gradient clipping."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_targets = []

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

            # Gradient clipping to eliminate cross-attention gradient explosions
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

        from src.research.evaluation.metrics import evaluate_predictions
        avg_loss = total_loss / max(num_batches, 1)
        train_eval = evaluate_predictions(all_targets, all_preds)
        return {
            "loss": avg_loss,
            "accuracy": train_eval["accuracy"],
            "weighted_f1": train_eval["weighted_f1"],
            "macro_f1": train_eval["macro_f1"],
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Evaluates model performance over validation/test dataset split."""
        from src.research.evaluation.metrics import evaluate_predictions
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
        """Runs complete training process with best-checkpoint restoration and early stopping."""
        logger.info(f"Starting training for {epochs} epochs on device: {self.device}")

        for cb in self.callbacks:
            cb.on_train_start(self)

        history: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_weighted_f1": [],
            "val_weighted_f1": [],
            "train_macro_f1": [],
            "val_macro_f1": [],
            "learning_rate": [],
            "best_epoch": 1,
            "best_val_weighted_f1": 0.0,
        }

        best_score = float("-inf")
        best_state_dict = None
        best_epoch = 1
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            for cb in self.callbacks:
                cb.on_epoch_start(epoch, self)

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            # Record history
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["train_weighted_f1"].append(train_metrics["weighted_f1"])
            history["val_weighted_f1"].append(val_metrics["weighted_f1"])
            history["train_macro_f1"].append(train_metrics["macro_f1"])
            history["val_macro_f1"].append(val_metrics["macro_f1"])
            history["learning_rate"].append(current_lr)

            # Monitor metric for best checkpoint (val_weighted_f1 or balanced accuracy)
            val_score = val_metrics["weighted_f1"]
            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
                star = " ⭐ [BEST CHECKPOINT]"
            else:
                epochs_no_improve += 1
                star = ""

            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | Val Weighted F1: {val_metrics['weighted_f1']:.4f}{star}"
            )

            metrics = {
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "val_macro_f1": val_metrics["macro_f1"],
            }

            for cb in self.callbacks:
                cb.on_epoch_end(epoch, metrics, self)

            # Early stopping check
            if self.early_stopping_patience and epochs_no_improve >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered at Epoch {epoch}: No improvement for {self.early_stopping_patience} consecutive epochs."
                )
                break

        history["best_epoch"] = best_epoch
        history["best_val_weighted_f1"] = best_score

        # Automatically restore the best model weights
        if self.restore_best and best_state_dict is not None:
            logger.info(f"Restoring best model checkpoint from Epoch {best_epoch} (Val Weighted F1: {best_score:.4f}).")
            self.model.load_state_dict(best_state_dict)

        for cb in self.callbacks:
            cb.on_train_end(self)

        logger.info("Training complete.")
        return history

