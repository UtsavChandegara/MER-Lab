"""Experiment Runner and Metadata Saver (MER-RULE-017, MER-RULE-035, MER-RULE-228..230).

Orchestrates full research experiments: setup -> data loading -> model building -> training -> evaluation -> artifact serialization.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader

from src.foundation.config import Config
from src.foundation.logging import setup_logger, get_logger
from src.foundation.seed import set_seed
from src.foundation.device import get_device
from src.ai_engine.dataset.registry import dataset_registry
from src.ai_engine.dataset.components import collate_multimodal_batch
from src.ai_engine.builders.builder import ModelBuilder
from src.training.trainer import Trainer
from src.training.callbacks import CheckpointCallback
from src.training.losses import loss_registry
from src.research.evaluation.metrics import evaluate_predictions

logger = get_logger("MERLab.Research.ExperimentRunner")


class ExperimentRunner:
    """Coordinates complete, reproducible research experiments (MER-RULE-228)."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = Config.from_yaml(self.config_path)
        
        # Experiment output directory setup
        exp_name = self.config.get("project.name", "experiment")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_root = Path(self.config.get("project.output_dir", "outputs"))
        self.exp_dir = output_root / f"{exp_name}_{timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Setup experiment logger
        setup_logger("MERLab", log_file=self.exp_dir / "experiment.log")
        logger.info(f"Initialized Experiment Directory at: {self.exp_dir}")

    def run(self) -> Dict[str, Any]:
        """Executes full experiment pipeline step-by-step (MER-RULE-200)."""
        logger.info("================ STARTING MER-LAB EXPERIMENT ================")
        
        # 1. Foundation Setup (Seed & Compute Device)
        seed = self.config.get("project.seed", 42)
        set_seed(seed)
        device = get_device(self.config.get("training.device", "auto"))

        # 2. Dataset System Setup
        dataset_name = self.config.get("dataset.name", "synthetic_meld")
        data_dir = self.config.get("dataset.data_dir", "data")
        num_samples = self.config.get("dataset.num_samples", 100)
        batch_size = self.config.get("dataset.batch_size", 16)

        logger.info(f"Loading Dataset: '{dataset_name}'")
        dataset_cls = dataset_registry.get(dataset_name)
        
        # Instantiate train and validation dataset splits
        train_dataset = dataset_cls(num_samples=num_samples, seed=seed)
        val_dataset = dataset_cls(num_samples=max(num_samples // 4, 10), seed=seed + 1)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_multimodal_batch,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_multimodal_batch,
        )

        # 3. AI Model Builder Setup
        logger.info("Building MERModel architecture from configuration...")
        model = ModelBuilder.build_model(self.config)

        # 4. Training Engine Setup
        lr = self.config.get("training.learning_rate", 0.001)
        weight_decay = self.config.get("training.weight_decay", 0.0001)
        epochs = self.config.get("training.epochs", 5)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = loss_registry.build("cross_entropy")

        checkpoint_callback = CheckpointCallback(
            checkpoint_dir=self.exp_dir / "checkpoints",
            save_best_only=True,
            monitor_metric="val_weighted_f1",
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            callbacks=[checkpoint_callback],
        )

        # 5. Fit Model & Execute Evaluation
        history = trainer.fit(train_loader, val_loader, epochs=epochs)
        final_metrics = trainer.evaluate(val_loader)

        # 6. Save Complete Metadata & Research Artifacts (MER-RULE-035, MER-RULE-184)
        self._save_artifacts(final_metrics, history)

        logger.info("================ EXPERIMENT COMPLETED SUCCESSFULLY ================")
        return final_metrics

    def _save_artifacts(self, final_metrics: Dict[str, Any], history: Dict[str, Any]) -> None:
        """Saves configuration snapshot, metrics JSON, and summary text report."""
        # Config snapshot
        with open(self.exp_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
            import yaml
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)

        # Metrics JSON
        artifact_data = {
            "config": self.config.to_dict(),
            "final_metrics": final_metrics,
            "training_history": history,
        }
        with open(self.exp_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(artifact_data, f, indent=2)

        # Summary Report TXT (MER-RULE-017)
        with open(self.exp_dir / "summary_report.txt", "w", encoding="utf-8") as f:
            f.write("================ MER-Lab Experiment Summary Report ================\n")
            f.write(f"Experiment Name : {self.config.get('project.name')}\n")
            f.write(f"Output Path     : {self.exp_dir}\n")
            f.write(f"Accuracy        : {final_metrics.get('accuracy', 0.0):.4f}\n")
            f.write(f"Weighted F1     : {final_metrics.get('weighted_f1', 0.0):.4f}\n")
            f.write(f"Macro F1        : {final_metrics.get('macro_f1', 0.0):.4f}\n")
            f.write("\nPer-Class F1 Scores:\n")
            for cls_name, f1 in final_metrics.get("per_class_f1", {}).items():
                f.write(f"  Class {cls_name}: {f1:.4f}\n")
            f.write("===================================================================\n")

        logger.info(f"Saved complete research artifacts to '{self.exp_dir}'.")
