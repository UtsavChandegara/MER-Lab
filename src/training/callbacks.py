"""Training Callbacks and Checkpoint Management (MER-RULE-035, MER-RULE-184).
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch

from src.foundation.logging import get_logger

logger = get_logger("MERLab.Training.Callback")


class BaseCallback:
    """Abstract Base Class for Trainer callbacks."""

    def on_train_start(self, trainer: Any) -> None:
        pass

    def on_epoch_start(self, epoch: int, trainer: Any) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass


class CheckpointCallback(BaseCallback):
    """Saves model checkpoints, configuration, and optimizer state to disk automatically (MER-RULE-035)."""

    def __init__(self, checkpoint_dir: Path, save_best_only: bool = True, monitor_metric: str = "val_weighted_f1"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.monitor_metric = monitor_metric
        self.best_score = float("-inf")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer: Any) -> None:
        current_score = metrics.get(self.monitor_metric, 0.0)

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "metrics": metrics,
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint_data, latest_path)

        # Save best checkpoint if metric improved
        if current_score > self.best_score:
            self.best_score = current_score
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint_data, best_path)
            logger.info(f"Epoch {epoch}: Improved '{self.monitor_metric}' to {current_score:.4f}. Saved best checkpoint.")
