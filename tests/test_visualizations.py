"""Unit tests for the publication figure and visualization generator."""

import unittest
import shutil
from pathlib import Path

from src.research.evaluation.visualizations import (
    plot_multimodal_superiority,
    plot_fusion_comparison,
    plot_dynamic_gating_heatmap,
    plot_missing_modality_robustness,
    plot_confusion_matrix_heatmap,
    plot_minority_class_gains,
    render_all_figures,
)


class TestVisualizations(unittest.TestCase):
    """Verifies that all academic plots generate valid PNG and PDF image files."""

    def setUp(self):
        self.test_dir = Path("outputs/test_figs")
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_all_individual_plots(self):
        # 1. H1 Multimodal
        h1_data = {
            "Text-Only": {"accuracy": 0.60, "macro_f1": 0.58},
            "Audio-Only": {"accuracy": 0.55, "macro_f1": 0.52},
            "Trimodal": {"accuracy": 0.72, "macro_f1": 0.70},
        }
        p1 = plot_multimodal_superiority(h1_data, self.test_dir)
        self.assertTrue(p1.exists())
        self.assertTrue((self.test_dir / "fig1_multimodal_superiority.pdf").exists())

        # 2. H2 Fusion
        h2_data = {
            "Concat": {"weighted_f1": 0.65, "macro_f1": 0.63},
            "Proposed DGCA": {"weighted_f1": 0.72, "macro_f1": 0.70},
        }
        p2 = plot_fusion_comparison(h2_data, self.test_dir)
        self.assertTrue(p2.exists())

        # 3. H3 Gating
        gating_data = {
            "neutral": {"audio": 0.3, "text": 0.5, "video": 0.2},
            "joy": {"audio": 0.4, "text": 0.3, "video": 0.3},
        }
        p3 = plot_dynamic_gating_heatmap(gating_data, self.test_dir)
        self.assertTrue(p3.exists())

        # 4. H4 Missing Modality
        h4_data = {
            "Complete (T+A+V)": {"macro_f1": 0.70},
            "Missing Video": {"macro_f1": 0.66},
            "Missing Text": {"macro_f1": 0.58},
        }
        p4 = plot_missing_modality_robustness(h4_data, self.test_dir)
        self.assertTrue(p4.exists())

        # 5. Confusion Matrix
        cm = [[10, 2], [3, 15]]
        p5 = plot_confusion_matrix_heatmap(cm, ["neutral", "joy"], self.test_dir)
        self.assertTrue(p5.exists())

        # 6. Minority Class F1 Gains
        h7_data = {
            "neutral": {"absolute_gain": 0.02},
            "fear": {"absolute_gain": 0.14},
            "disgust": {"absolute_gain": 0.12},
        }
        p6 = plot_minority_class_gains(h7_data, self.test_dir)
        self.assertTrue(p6.exists())

    def test_render_all_figures(self):
        mock_results = {
            "H1_multimodal_superiority": {"Text": {"accuracy": 0.6, "macro_f1": 0.55}},
            "H2_fusion_comparison": {"Concat": {"weighted_f1": 0.6, "macro_f1": 0.55}},
            "H4_missing_modality_robustness": {"Full": {"macro_f1": 0.7}},
            "H7_per_class_breakdown": {"joy": {"absolute_gain": 0.05}},
        }
        figs = render_all_figures(mock_results, self.test_dir)
        self.assertGreater(len(figs), 0)


if __name__ == "__main__":
    unittest.main()
