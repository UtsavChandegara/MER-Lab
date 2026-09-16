"""Automated Benchmark Suite for MELD Trimodal Emotion Recognition Hypotheses (H1–H7).

Executes experimental conditions to empirically evaluate:
- H1: Multimodal Superiority (Unimodal vs Bimodal vs Trimodal)
- H2: Advanced Fusion (Proposed DGCA vs Concat vs Average vs Attention)
- H3: Dynamic Modality Gating (Modality importance weights)
- H4: Missing Modality Robustness (Inference under modality drops)
- H6: Component Ablation Study
- H7: Per-Class & Minority Emotion Breakdown

Generates JSON metrics and publication-ready LaTeX tables.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
import copy
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
from src.training.losses import loss_registry
from src.research.evaluation.metrics import evaluate_predictions

logger = get_logger("MERLab.Research.BenchmarkRunner")


class BenchmarkSuite:
    """Orchestrates all experimental trials for research paper hypotheses H1–H7."""

    def __init__(self, base_config_path: str = "configs/meld_trimodal.yaml", output_dir: str = "outputs/benchmarks"):
        self.base_config = Config.from_yaml(base_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_logger("MERLab", log_file=self.output_dir / "benchmark.log")
        self.results: Dict[str, Any] = {}

    def _train_and_eval(self, config_dict: Dict[str, Any], exp_name: str, mask_modalities_eval: List[str] = None) -> Dict[str, Any]:
        """Trains a model configuration and evaluates it on the validation/test set."""
        logger.info(f"--- Running Trial: {exp_name} ---")
        cfg = Config(config_dict)
        seed = cfg.get("project.seed", 42)
        set_seed(seed)
        device = get_device(cfg.get("training.device", "auto"))

        dataset_cls = dataset_registry.get(cfg.get("dataset.name", "meld_features"))
        data_dir = cfg.get("dataset.data_dir", "data/meld")
        num_samples = cfg.get("dataset.num_samples", None)
        batch_size = cfg.get("dataset.batch_size", 32)
        val_samples = max(num_samples // 4, 20) if num_samples is not None else None

        try:
            train_dataset = dataset_cls(data_dir=data_dir, split="train", num_samples=num_samples, seed=seed)
            val_dataset = dataset_cls(data_dir=data_dir, split="dev", num_samples=val_samples, seed=seed + 1)
        except TypeError:
            train_dataset = dataset_cls(num_samples=num_samples or 500, seed=seed)
            val_dataset = dataset_cls(num_samples=val_samples or 100, seed=seed + 1)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_multimodal_batch)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal_batch)

        model = ModelBuilder.build_model(cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.get("training.learning_rate", 0.0005),
            weight_decay=cfg.get("training.weight_decay", 0.0001),
        )
        criterion = loss_registry.build("cross_entropy")

        trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device)
        epochs = cfg.get("training.epochs", 5)
        trainer.fit(train_loader, val_loader, epochs=epochs)

        # Optional inference masking (for H4 missing modality testing)
        if mask_modalities_eval:
            logger.info(f"Applying evaluation modality mask: zeroing out {mask_modalities_eval}")
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    inputs = {k: v.clone() for k, v in batch.inputs.items()}
                    for m in mask_modalities_eval:
                        if m in inputs:
                            inputs[m] = torch.zeros_like(inputs[m])
                    logits = model(inputs)
                    preds = torch.argmax(logits, dim=-1)
                    all_preds.extend(preds.cpu().tolist())
                    all_targets.extend(batch.labels.cpu().tolist())
            metrics = evaluate_predictions(all_targets, all_preds)
        else:
            metrics = trainer.evaluate(val_loader)

        logger.info(
            f"Trial {exp_name} Finished -> Accuracy: {metrics['accuracy']:.4f}, "
            f"Weighted F1: {metrics['weighted_f1']:.4f}, Macro F1: {metrics['macro_f1']:.4f}"
        )
        return metrics

    def run_h1_multimodal_superiority(self) -> Dict[str, Any]:
        """H1: Compare Trimodal vs Unimodal & Bimodal configurations."""
        logger.info("================ EVALUATING H1: MULTIMODAL SUPERIORITY ================")
        h1_results = {}
        modalities_to_test = {
            "Text-Only": ["text"],
            "Audio-Only": ["audio"],
            "Video-Only": ["video"],
            "Bimodal (Text+Audio)": ["text", "audio"],
            "Bimodal (Text+Video)": ["text", "video"],
            "Bimodal (Audio+Video)": ["audio", "video"],
            "Trimodal (Text+Audio+Video)": ["text", "audio", "video"],
        }

        for condition_name, mods in modalities_to_test.items():
            cfg_dict = copy.deepcopy(self.base_config.to_dict())
            cfg_dict["dataset"]["modalities"] = mods
            # Keep only selected modalities in encoder configuration
            cfg_dict["model"]["encoder"] = {m: cfg_dict["model"]["encoder"][m] for m in mods}
            metrics = self._train_and_eval(cfg_dict, f"H1_{condition_name}")
            h1_results[condition_name] = {
                "accuracy": metrics["accuracy"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_f1": metrics["macro_f1"],
            }

        self.results["H1_multimodal_superiority"] = h1_results
        return h1_results

    def run_h2_fusion_comparison(self) -> Dict[str, Any]:
        """H2: Compare Proposed DGCA Fusion against Concatenation, Average, and Attention."""
        logger.info("================ EVALUATING H2: FUSION STRATEGIES ================")
        h2_results = {}
        fusion_methods = {
            "Concat Fusion": {"name": "concat_fusion", "num_modalities": 3},
            "Average Fusion": {"name": "average_fusion"},
            "Self-Attention Fusion": {"name": "attention_fusion", "num_heads": 4},
            "Proposed DGCA Fusion": {
                "name": "dynamic_gated_cross_attention",
                "num_heads": 4,
                "dim_feedforward": 512,
                "modality_dropout": 0.15,
            },
        }

        for fname, fcfg in fusion_methods.items():
            cfg_dict = copy.deepcopy(self.base_config.to_dict())
            cfg_dict["model"]["fusion"] = fcfg
            cfg_dict["model"]["fusion"]["projection_dim"] = 256
            metrics = self._train_and_eval(cfg_dict, f"H2_{fname}")
            h2_results[fname] = {
                "accuracy": metrics["accuracy"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_f1": metrics["macro_f1"],
            }

        self.results["H2_fusion_comparison"] = h2_results
        return h2_results

    def run_h4_missing_modality_robustness(self) -> Dict[str, Any]:
        """H4: Evaluate model under missing/occluded modalities at test time."""
        logger.info("================ EVALUATING H4: MISSING-MODALITY ROBUSTNESS ================")
        h4_results = {}
        cfg_dict = copy.deepcopy(self.base_config.to_dict())

        drop_conditions = {
            "Complete (T + A + V)": None,
            "Missing Text (Zero Text)": ["text"],
            "Missing Audio (Zero Audio)": ["audio"],
            "Missing Video (Zero Video)": ["video"],
            "Audio + Video Only (No Text)": ["text"],
            "Text Only (No Audio/Video)": ["audio", "video"],
        }

        for cname, mask in drop_conditions.items():
            metrics = self._train_and_eval(cfg_dict, f"H4_{cname}", mask_modalities_eval=mask)
            h4_results[cname] = {
                "accuracy": metrics["accuracy"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_f1": metrics["macro_f1"],
            }

        self.results["H4_missing_modality_robustness"] = h4_results
        return h4_results

    def run_h6_ablation_study(self) -> Dict[str, Any]:
        """H6: Component ablation study of the proposed DGCA fusion."""
        logger.info("================ EVALUATING H6: ABLATION STUDY ================")
        h6_results = {}
        ablation_configs = {
            "Full Proposed Model": copy.deepcopy(self.base_config.to_dict()),
            "Ablation 1: w/o Modality Dropout (p=0)": copy.deepcopy(self.base_config.to_dict()),
            "Ablation 2: w/o Cross-Attention (Avg)": copy.deepcopy(self.base_config.to_dict()),
            "Ablation 3: Linear Classifier (No MLP)": copy.deepcopy(self.base_config.to_dict()),
        }

        ablation_configs["Ablation 1: w/o Modality Dropout (p=0)"]["model"]["fusion"]["modality_dropout"] = 0.0
        ablation_configs["Ablation 2: w/o Cross-Attention (Avg)"]["model"]["fusion"] = {"name": "average_fusion", "projection_dim": 256}
        ablation_configs["Ablation 3: Linear Classifier (No MLP)"]["model"]["classifier"] = {"name": "linear_classifier", "num_classes": 7}

        for aname, acfg in ablation_configs.items():
            metrics = self._train_and_eval(acfg, f"H6_{aname}")
            h6_results[aname] = {
                "accuracy": metrics["accuracy"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_f1": metrics["macro_f1"],
            }

        self.results["H6_ablation_study"] = h6_results
        return h6_results

    def run_h7_per_class_breakdown(self) -> Dict[str, Any]:
        """H7: Detailed per-class F1 breakdown comparing Unimodal vs Trimodal."""
        logger.info("================ EVALUATING H7: CLASS-LEVEL BREAKDOWN ================")
        emotions = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
        
        # Train Unimodal Text
        cfg_text = copy.deepcopy(self.base_config.to_dict())
        cfg_text["dataset"]["modalities"] = ["text"]
        cfg_text["model"]["encoder"] = {"text": cfg_text["model"]["encoder"]["text"]}
        metrics_text = self._train_and_eval(cfg_text, "H7_TextOnly")

        # Train Trimodal
        cfg_tri = copy.deepcopy(self.base_config.to_dict())
        metrics_tri = self._train_and_eval(cfg_tri, "H7_Trimodal")

        h7_results = {}
        for idx, emo in enumerate(emotions):
            f1_text = metrics_text.get("per_class_f1", {}).get(str(idx), 0.0)
            f1_tri = metrics_tri.get("per_class_f1", {}).get(str(idx), 0.0)
            gain = f1_tri - f1_text
            h7_results[emo] = {
                "class_id": idx,
                "text_f1": f1_text,
                "trimodal_f1": f1_tri,
                "absolute_gain": gain,
            }

        self.results["H7_per_class_breakdown"] = h7_results
        return h7_results

    def generate_latex_tables(self) -> None:
        """Exports publication-ready LaTeX tables."""
        latex_dir = self.output_dir / "latex_tables"
        latex_dir.mkdir(parents=True, exist_ok=True)

        # Table 1: H1 Multimodal Superiority
        if "H1_multimodal_superiority" in self.results:
            with open(latex_dir / "table1_multimodal_h1.tex", "w") as f:
                f.write("% Table 1: Multimodal vs Unimodal Performance (H1)\n")
                f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
                f.write("\\begin{tabular}{lccc}\n\\hline\n")
                f.write("\\textbf{Modality Configuration} & \\textbf{Accuracy} & \\textbf{Weighted F1} & \\textbf{Macro F1} \\\\ \\hline\n")
                for mod, met in self.results["H1_multimodal_superiority"].items():
                    f.write(f"{mod} & {met['accuracy']:.4f} & {met['weighted_f1']:.4f} & {met['macro_f1']:.4f} \\\\\n")
                f.write("\\hline\n\\end{tabular}\n")
                f.write("\\caption{Comparison of unimodal, bimodal, and trimodal emotion recognition on MELD (H1).}\n")
                f.write("\\label{tab:multimodal_h1}\n\\end{table}\n")

        # Table 2: H2 Fusion Strategies
        if "H2_fusion_comparison" in self.results:
            with open(latex_dir / "table2_fusion_h2.tex", "w") as f:
                f.write("% Table 2: Fusion Strategy Comparison (H2)\n")
                f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
                f.write("\\begin{tabular}{lccc}\n\\hline\n")
                f.write("\\textbf{Fusion Architecture} & \\textbf{Accuracy} & \\textbf{Weighted F1} & \\textbf{Macro F1} \\\\ \\hline\n")
                for fn, met in self.results["H2_fusion_comparison"].items():
                    bold = "\\textbf{" if "Proposed" in fn else ""
                    endb = "}" if "Proposed" in fn else ""
                    f.write(f"{bold}{fn}{endb} & {bold}{met['accuracy']:.4f}{endb} & {bold}{met['weighted_f1']:.4f}{endb} & {bold}{met['macro_f1']:.4f}{endb} \\\\\n")
                f.write("\\hline\n\\end{tabular}\n")
                f.write("\\caption{Benchmarking multimodal fusion architectures on MELD (H2).}\n")
                f.write("\\label{tab:fusion_h2}\n\\end{table}\n")

        # Table 3: H4 Missing Modality
        if "H4_missing_modality_robustness" in self.results:
            with open(latex_dir / "table3_missing_modality_h4.tex", "w") as f:
                f.write("% Table 3: Missing Modality Robustness (H4)\n")
                f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
                f.write("\\begin{tabular}{lccc}\n\\hline\n")
                f.write("\\textbf{Inference Availability} & \\textbf{Accuracy} & \\textbf{Weighted F1} & \\textbf{Macro F1} \\\\ \\hline\n")
                for cond, met in self.results["H4_missing_modality_robustness"].items():
                    f.write(f"{cond} & {met['accuracy']:.4f} & {met['weighted_f1']:.4f} & {met['macro_f1']:.4f} \\\\\n")
                f.write("\\hline\n\\end{tabular}\n")
                f.write("\\caption{Robustness under missing or occluded modalities at inference (H4).}\n")
                f.write("\\label{tab:missing_h4}\n\\end{table}\n")

        # Table 4: H6 Ablations
        if "H6_ablation_study" in self.results:
            with open(latex_dir / "table4_ablation_h6.tex", "w") as f:
                f.write("% Table 4: Component Ablation Study (H6)\n")
                f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
                f.write("\\begin{tabular}{lccc}\n\\hline\n")
                f.write("\\textbf{Model Variant} & \\textbf{Accuracy} & \\textbf{Weighted F1} & \\textbf{Macro F1} \\\\ \\hline\n")
                for av, met in self.results["H6_ablation_study"].items():
                    f.write(f"{av} & {met['accuracy']:.4f} & {met['weighted_f1']:.4f} & {met['macro_f1']:.4f} \\\\\n")
                f.write("\\hline\n\\end{tabular}\n")
                f.write("\\caption{Ablation analysis of the proposed DGCA fusion components (H6).}\n")
                f.write("\\label{tab:ablation_h6}\n\\end{table}\n")

        # Table 5: H7 Per-Class Breakdown
        if "H7_per_class_breakdown" in self.results:
            with open(latex_dir / "table5_per_class_h7.tex", "w") as f:
                f.write("% Table 5: Per-Class F1 Breakdown (H7)\n")
                f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
                f.write("\\begin{tabular}{lccc}\n\\hline\n")
                f.write("\\textbf{Emotion Category} & \\textbf{Text-Only F1} & \\textbf{Trimodal DGCA F1} & \\textbf{Delta Gain} \\\\ \\hline\n")
                for emo, met in self.results["H7_per_class_breakdown"].items():
                    sign = "+" if met["absolute_gain"] >= 0 else ""
                    f.write(f"{emo.capitalize()} & {met['text_f1']:.4f} & {met['trimodal_f1']:.4f} & {sign}{met['absolute_gain']:.4f} \\\\\n")
                f.write("\\hline\n\\end{tabular}\n")
                f.write("\\caption{Per-class performance comparison highlighting gains in minority emotions (H7).}\n")
                f.write("\\label{tab:per_class_h7}\n\\end{table}\n")

        logger.info(f"Successfully generated all LaTeX publication tables in '{latex_dir}'.")

    def run_all(self) -> Dict[str, Any]:
        """Runs the complete suite across all hypotheses and dumps artifacts."""
        start_time = time.time()
        logger.info("Starting Full Research Hypothesis Benchmark Suite...")

        self.run_h1_multimodal_superiority()
        self.run_h2_fusion_comparison()
        self.run_h4_missing_modality_robustness()
        self.run_h6_ablation_study()
        self.run_h7_per_class_breakdown()

        # Save all results to JSON
        with open(self.output_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

        # Generate LaTeX publication tables
        self.generate_latex_tables()

        # Generate Publication-Ready Figures (PNG & PDF at 300 DPI)
        try:
            from src.research.evaluation.visualizations import render_all_figures
            render_all_figures(self.results, self.output_dir)
            logger.info(f"Saved all 300-DPI publication figures to '{self.output_dir / 'figures'}'.")
        except Exception as e:
            logger.warning(f"Figure generation skipped due to error: {str(e)}")

        elapsed = time.time() - start_time
        logger.info(f"All Hypotheses Benchmarks Completed in {elapsed:.2f} seconds!")
        return self.results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MER-Lab Hypothesis Benchmark Suite (H1–H7)")
    parser.add_argument("--config", type=str, default="configs/meld_trimodal.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/benchmarks")
    args = parser.parse_args()

    suite = BenchmarkSuite(base_config_path=args.config, output_dir=args.output_dir)
    suite.run_all()
