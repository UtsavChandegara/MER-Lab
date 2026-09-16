"""Publication-Ready Figure & Plot Generator for MER-Lab (H1–H10).

Generates 300-DPI academic figures in both PNG and PDF formats using Matplotlib:
1. Figure 1: Multimodal Superiority (H1: Unimodal vs Bimodal vs Trimodal)
2. Figure 2: Fusion Architecture Benchmark (H2: Baselines vs Proposed DGCA)
3. Figure 3: Dynamic Modality Gating Distribution per Emotion (H3, H5)
4. Figure 4: Missing-Modality Robustness Degradation (H4)
5. Figure 5: Normalized Confusion Matrix Heatmap (Tier 1 & 2)
6. Figure 6: Minority Class F1 Delta Gains (H7)
7. Figure 7: Efficiency vs Performance Frontier (H5)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Academic publication style setup
def _setup_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def plot_multimodal_superiority(h1_results: Dict[str, Dict[str, float]], save_dir: Path) -> Path:
    """Figure 1: Multimodal Superiority (H1). Grouped bar chart for Unimodal vs Bimodal vs Trimodal."""
    _setup_plot_style()
    save_dir.mkdir(parents=True, exist_ok=True)

    conditions = list(h1_results.keys())
    accs = [h1_results[c].get("accuracy", 0.0) for c in conditions]
    macro_f1s = [h1_results[c].get("macro_f1", 0.0) for c in conditions]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    rects1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#3470A3")
    rects2 = ax.bar(x + width / 2, macro_f1s, width, label="Macro-F1", color="#59A14F")

    # Highlight the trimodal bar with a subtle outline
    if len(conditions) > 0:
        rects1[-1].set_edgecolor("#1B3C59")
        rects1[-1].set_linewidth(1.5)
        rects2[-1].set_edgecolor("#2D5726")
        rects2[-1].set_linewidth(1.5)

    ax.set_ylabel("Score")
    ax.set_title("H1: Emotion Recognition Across Modality Configurations on MELD", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True)

    # Value labels on top of bars
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = save_dir / "fig1_multimodal_superiority.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / "fig1_multimodal_superiority.pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fusion_comparison(h2_results: Dict[str, Dict[str, float]], save_dir: Path) -> Path:
    """Figure 2: Multimodal Fusion Comparison (H2)."""
    _setup_plot_style()
    save_dir.mkdir(parents=True, exist_ok=True)

    methods = list(h2_results.keys())
    macro_f1s = [h2_results[m].get("macro_f1", 0.0) for m in methods]
    weighted_f1s = [h2_results[m].get("weighted_f1", 0.0) for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    colors_f1 = ["#A0CBE8", "#A0CBE8", "#A0CBE8", "#E15759"]  # Highlight proposed in coral red
    colors_wf1 = ["#BAB0AC", "#BAB0AC", "#BAB0AC", "#F28E2B"]

    ax.bar(x - width / 2, macro_f1s, width, label="Macro-F1", color=colors_f1)
    ax.bar(x + width / 2, weighted_f1s, width, label="Weighted-F1", color=colors_wf1)

    ax.set_ylabel("F1 Score")
    ax.set_title("H2: Benchmarking Multimodal Fusion Strategies on MELD", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True)

    plt.tight_layout()
    out_path = save_dir / "fig2_fusion_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / "fig2_fusion_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_dynamic_gating_heatmap(per_class_gating: Dict[str, Dict[str, float]], save_dir: Path) -> Path:
    """Figure 3: Dynamic Gating Modality Weights per Emotion Class (H3, H5)."""
    _setup_plot_style()
    save_dir.mkdir(parents=True, exist_ok=True)

    classes = list(per_class_gating.keys())
    if not classes:
        return save_dir

    modalities = list(per_class_gating[classes[0]].keys())
    # Matrix: [Num_Classes, Num_Modalities]
    matrix = np.array([[per_class_gating[c].get(m, 0.0) for m in modalities] for c in classes])

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Learned Modality Weight (α)", rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(modalities)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels([m.capitalize() for m in modalities], fontweight="bold")
    ax.set_yticklabels([c.capitalize() for c in classes])
    ax.set_title("H3: Adaptive Modality Gating Weights (α) by Emotion Class", fontweight="bold", pad=12)

    # Print numerical values inside heatmap cells
    for i in range(len(classes)):
        for j in range(len(modalities)):
            val = matrix[i, j]
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold")

    plt.tight_layout()
    out_path = save_dir / "fig3_dynamic_gating_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / "fig3_dynamic_gating_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_missing_modality_robustness(h4_results: Dict[str, Dict[str, float]], save_dir: Path) -> Path:
    """Figure 4: Missing Modality Robustness Degradation (H4)."""
    _setup_plot_style()
    save_dir.mkdir(parents=True, exist_ok=True)

    conditions = list(h4_results.keys())
    macro_f1s = [h4_results[c].get("macro_f1", 0.0) for c in conditions]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    y_pos = np.arange(len(conditions))
    
    # Invert to have full condition on top
    y_pos = y_pos[::-1]
    conditions = conditions[::-1]
    macro_f1s = macro_f1s[::-1]

    bars = ax.barh(y_pos, macro_f1s, color="#4E79A7", edgecolor="#2E4A6F", height=0.55)
    bars[-1].set_color("#2CA02C")  # Full condition in distinct green

    ax.set_yticks(y_pos)
    ax.set_yticklabels(conditions)
    ax.set_xlabel("Macro-F1 Score")
    ax.set_title("H4: Model Robustness Under Modality Occlusion / Missingness", fontweight="bold", pad=12)
    ax.set_xlim(0.0, 1.05)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.3f}", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9)

    plt.tight_layout()
    out_path = save_dir / "fig4_missing_modality_robustness.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / "fig4_missing_modality_robustness.pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrix_heatmap(matrix: List[List[int]], class_names: List[str], save_dir: Path) -> Path:
    """Figure 5: Normalized Confusion Matrix Heatmap (Tier 1 & 2)."""
    _setup_plot_style()
    save_dir.mkdir(parents=True, exist_ok=True)

    mat = np.array(matrix, dtype=float)
    # Row normalize to get percentages
    row_sums = mat.sum(axis=1, keepdims=True)
    norm_mat = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
    im = ax.imshow(norm_mat, cmap="Purples", aspect="equal", vmin=0.0, vmax=1.0)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized Proportion", rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels([c.capitalize() for c in class_names], rotation=35, ha="right")
    ax.set_yticklabels([c.capitalize() for c in class_names])

    ax.set_xlabel("Predicted Emotion Label", fontweight="bold", labelpad=8)
    ax.set_ylabel("Ground Truth Emotion Label", fontweight="bold", labelpad=8)
    ax.set_title("Trimodal DGCA Confusion Matrix on MELD", fontweight="bold", pad=12)

    # Print annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = norm_mat[i, j]
            count = int(mat[i, j])
            text_color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.1%}\n({count})", ha="center", va="center", color=text_color, fontsize=7.5)

    plt.tight_layout()
    out_path = save_dir / "fig5_confusion_matrix.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / "fig5_confusion_matrix.pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_minority_class_gains(h7_results: Dict[str, Dict[str, Any]], save_dir: Path) -> Path:
    """Figure 6: Per-Class F1 Delta Gain (H7: Highlighting gains on minority emotions)."""
    _setup_plot_style()
    save_dir.mkdir(parents=True, exist_ok=True)

    classes = list(h7_results.keys())
    gains = [h7_results[c].get("absolute_gain", 0.0) for c in classes]

    colors = ["#2CA02C" if g >= 0 else "#D62728" for g in gains]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    x = np.arange(len(classes))
    bars = ax.bar(x, gains, color=colors, edgecolor="#333333", width=0.55)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes], rotation=25, ha="right")
    ax.set_ylabel("Absolute F1 Gain (Trimodal − Text-Only)")
    ax.set_title("H7: Multimodal Value-Add Across Emotion Categories", fontweight="bold", pad=12)

    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.annotate(f"{h:+.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3 if h >= 0 else -10), textcoords="offset points", ha="center", va=va, fontsize=8.5)

    plt.tight_layout()
    out_path = save_dir / "fig6_minority_class_f1_gains.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / "fig6_minority_class_f1_gains.pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_overfitting_underfitting_dynamics(history: Dict[str, Any], save_dir: Path) -> Path:
    """Figure 7: Training Dynamics, Overfitting & Underfitting Diagnostic Curves."""
    _setup_plot_style()
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_f1 = history.get("train_weighted_f1", history.get("train_accuracy", []))
    val_f1 = history.get("val_weighted_f1", history.get("val_accuracy", []))

    if not train_loss or not val_loss:
        return save_dir

    epochs = np.arange(1, len(train_loss) + 1)
    best_epoch = history.get("best_epoch", int(np.argmax(val_f1) + 1) if val_f1 else 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Subplot 1: Loss Convergence & Generalization Gap
    ax1.plot(epochs, train_loss, marker="o", markersize=4, label="Training Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, val_loss, marker="s", markersize=4, label="Validation Loss", color="#d62728", linewidth=2)
    ax1.axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.8, label=f"Best Checkpoint (Epoch {best_epoch})")
    ax1.plot(best_epoch, val_loss[best_epoch - 1], marker="*", markersize=14, color="#ffd700", markeredgecolor="#2ca02c", markeredgewidth=1.5)

    # Shaded Zones: Underfitting and Overfitting
    if best_epoch > 1:
        ax1.axvspan(1, best_epoch, alpha=0.08, color="#1f77b4", label="Underfitting / Convergence Zone")
    if best_epoch < len(epochs):
        ax1.axvspan(best_epoch, len(epochs), alpha=0.12, color="#d62728", label="Overfitting Regime (Generalization Gap)")

    ax1.set_xlabel("Epoch Number", fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss", fontweight="bold")
    ax1.set_title("Training vs. Validation Loss (Generalization Gap)", fontweight="bold", pad=10)
    ax1.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax1.set_xticks(epochs)

    # Subplot 2: F1 Score Performance Trajectory
    ax2.plot(epochs, train_f1, marker="o", markersize=4, label="Train Weighted-F1", color="#3b528b", linewidth=2)
    ax2.plot(epochs, val_f1, marker="^", markersize=4, label="Val Weighted-F1", color="#2ca02c", linewidth=2)
    ax2.axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.8)
    ax2.plot(best_epoch, val_f1[best_epoch - 1], marker="*", markersize=14, color="#ffd700", markeredgecolor="#2ca02c", markeredgewidth=1.5, label=f"Peak Val F1 ({val_f1[best_epoch - 1]:.3f})")

    if best_epoch < len(epochs):
        ax2.axvspan(best_epoch, len(epochs), alpha=0.10, color="#d62728")

    ax2.set_xlabel("Epoch Number", fontweight="bold")
    ax2.set_ylabel("Weighted F1 Score", fontweight="bold")
    ax2.set_title("Model F1 Progression & Peak Generalization Point", fontweight="bold", pad=10)
    ax2.legend(loc="lower right", frameon=True, fontsize=8.5)
    ax2.set_xticks(epochs)

    fig.suptitle("MER-Lab Diagnostic: Learning Dynamics, Overfitting & Optimal Checkpoint Selection", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_path = save_dir / "fig7_overfitting_underfitting_dynamics.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / "fig7_overfitting_underfitting_dynamics.pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_multimodal_learning_curves(histories: Dict[str, Dict[str, Any]], save_dir: Path) -> Path:
    """Figure 8: Comparative Validation Trajectories across Modality Configurations."""
    _setup_plot_style()
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    colors = {"Text-Only": "#1f77b4", "Audio-Only": "#ff7f0e", "Video-Only": "#9467bd", "Trimodal DGCA": "#2ca02c"}

    for name, hist in histories.items():
        val_loss = hist.get("val_loss", [])
        val_f1 = hist.get("val_weighted_f1", hist.get("val_accuracy", []))
        if not val_loss:
            continue
        epochs = np.arange(1, len(val_loss) + 1)
        c = colors.get(name, None)
        lw = 2.5 if "Trimodal" in name else 1.8
        ls = "-" if "Trimodal" in name else "--"

        ax1.plot(epochs, val_loss, label=name, color=c, linewidth=lw, linestyle=ls, marker="o", markersize=3.5)
        ax2.plot(epochs, val_f1, label=name, color=c, linewidth=lw, linestyle=ls, marker="s", markersize=3.5)

    ax1.set_xlabel("Epoch Number", fontweight="bold")
    ax1.set_ylabel("Validation Loss", fontweight="bold")
    ax1.set_title("Validation Loss Trajectories across Modalities", fontweight="bold", pad=10)
    ax1.legend(frameon=True, fontsize=9)

    ax2.set_xlabel("Epoch Number", fontweight="bold")
    ax2.set_ylabel("Validation Weighted-F1", fontweight="bold")
    ax2.set_title("Validation F1 Trajectories across Modalities", fontweight="bold", pad=10)
    ax2.legend(frameon=True, fontsize=9)

    fig.suptitle("Comparative Multimodal Learning Dynamics: Convergence & Stability", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_path = save_dir / "fig8_multimodal_learning_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_dir / "fig8_multimodal_learning_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_all_figures(results: Dict[str, Any], save_dir: Path) -> List[Path]:
    """Generates all publication figures from the experiment results dictionary."""
    fig_paths = []
    figures_dir = save_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. H1 Multimodal Superiority
    if "H1_multimodal_superiority" in results:
        p1 = plot_multimodal_superiority(results["H1_multimodal_superiority"], figures_dir)
        fig_paths.append(p1)

    # 2. H2 Fusion Strategies
    if "H2_fusion_comparison" in results:
        p2 = plot_fusion_comparison(results["H2_fusion_comparison"], figures_dir)
        fig_paths.append(p2)

    # 3. H3 Dynamic Gating (if gating data present)
    if "H3_dynamic_gating" in results and "per_class_gating" in results["H3_dynamic_gating"]:
        p3 = plot_dynamic_gating_heatmap(results["H3_dynamic_gating"]["per_class_gating"], figures_dir)
        fig_paths.append(p3)

    # 4. H4 Missing Modality
    if "H4_missing_modality_robustness" in results:
        p4 = plot_missing_modality_robustness(results["H4_missing_modality_robustness"], figures_dir)
        fig_paths.append(p4)

    # 5. Confusion Matrix (Figure 5)
    if "confusion_matrix" in results:
        class_names = results.get("class_names", ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"])
        p5 = plot_confusion_matrix_heatmap(results["confusion_matrix"], class_names, figures_dir)
        fig_paths.append(p5)

    # 6. H7 Minority Class Analysis
    if "H7_per_class_breakdown" in results:
        p6 = plot_minority_class_gains(results["H7_per_class_breakdown"], figures_dir)
        fig_paths.append(p6)

    # 7. Figure 7: Overfitting & Underfitting Dynamics
    if "training_history" in results:
        p7 = plot_overfitting_underfitting_dynamics(results["training_history"], figures_dir)
        fig_paths.append(p7)

    # 8. Figure 8: Multimodal Learning Curves Comparison
    if "multimodal_histories" in results:
        p8 = plot_multimodal_learning_curves(results["multimodal_histories"], figures_dir)
        fig_paths.append(p8)

    return fig_paths

