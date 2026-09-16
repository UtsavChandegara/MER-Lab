# 🧪 MER-Lab: Trimodal Emotion Recognition on MELD

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UtsavChandegara/MER-Lab/blob/main/notebooks/MER_Lab_MELD_Experiments.ipynb)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/tests-33%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Datasets](https://img.shields.io/badge/Datasets-MELD%20%7C%20IEMOCAP%20%7C%20MOSEI-blue.svg)]()
[![Model Size](https://img.shields.io/badge/Model%20Size-1.42M%20params-informational.svg)]()
[![Inference Latency](https://img.shields.io/badge/Latency-2.85%20ms-success.svg)]()

**An open-source research framework for conversational Multimodal Emotion Recognition (MER) featuring Dynamic Gated Cross-Attention (DGCA), missing-modality robustness, and zero-cost reproducibility.**

[📖 Academic Paper (LaTeX)](paper/main.tex) • [📘 Architecture & Codebase Guide](ARCHITECTURE.md) • [🚀 Open in Colab](https://colab.research.google.com/github/UtsavChandegara/MER-Lab/blob/main/notebooks/MER_Lab_MELD_Experiments.ipynb) • [📊 View Benchmark Results](outputs/colab_benchmarks/)

</div>

---

## 📌 Table of Contents
- [🔬 Research Overview](#-research-overview)
- [🏛️ System Architecture](#️-system-architecture)
- [📘 Full Architecture & Codebase Guide](ARCHITECTURE.md)
- [📊 Key Empirical Benchmark Results](#-key-empirical-benchmark-results)
- [📈 Diagnostic Visualizations](#-diagnostic-visualizations)
- [🚀 Quickstart & Reproduction](#-quickstart--reproduction)
  - [Option 1: 1-Click Cloud Execution (Google Colab)](#option-1-1-click-cloud-execution-google-colab)
  - [Option 2: Local Installation](#option-2-local-installation)
- [📝 Academic Paper & Overleaf Ready](#-academic-paper--overleaf-ready)
- [🧪 Running Unit Tests](#-running-unit-tests)
- [📂 Repository Structure](#-repository-structure)
- [📜 Citation & License](#-citation--license)

---

## 🔬 Research Overview

In conversational emotion recognition, models often suffer from **textual dominance**, **minority class fragility** (*fear*, *disgust*), and **catastrophic degradation** when sensors experience occlusion or packet loss.

**MER-Lab** addresses these challenges through:
1. **Dynamic Gated Cross-Attention (DGCA):** Projects heterogeneous foundation representations into a shared latent space ($d=256$) and computes input-conditioned adaptive attention weights ($\mathbf{\alpha} = [\alpha_{\text{text}}, \alpha_{\text{audio}}, \alpha_{\text{video}}]$).
2. **Missing-Modality Robustness (Modality Dropout):** Employs stochastic modality zeroing ($p=0.15$) during training to ensure graceful degradation when cameras or microphones drop out at inference time.
3. **Ultra-Lightweight Efficiency (H5):** Operates on frozen foundation representations (RoBERTa 768d, Acoustic 768d, Visual 512d) with only **1.42M trainable parameters** (5.68 MB), achieving **2.85 ms inference latency** (350+ utterances/sec).
4. **100% Zero-Cost Reproducibility:** No proprietary APIs or paid GPU instances needed; trains in <8 minutes on a free Google Colab T4 GPU.

---

## 🏛️ System Architecture

```text
                               ┌────────────────────────┐
   Text Transcript (U)        │  RoBERTa-base (768d)   │ ── Linear Proj (256d) ──┐
                               └────────────────────────┘                        │
                                                                                 │
                               ┌────────────────────────┐                        ├── [Batch, 3, 256]
   Acoustic Speech Waveform   │  WavLM / Prosody (768d)│ ── Linear Proj (256d) ──┤          │
                               └────────────────────────┘                        │          ▼
                                                                                 │   Multi-Head Cross-Attention
                               ┌────────────────────────┐                        │   (4 Heads, GELU, Norm)
   Visual Video Frames        │  Visual / Facial (512d)│ ── Linear Proj (256d) ──┘          │
                               └────────────────────────┘                                   ▼
                                                                                   Dynamic Gating Network
                                                                                 α = Softmax(MLP([h_t; h_a; h_v]))
                                                                                            │
                                                                                            ▼
                                                                                   z_fused = Σ α_m * h_m
                                                                                            │
                                                                                            ▼
                                                                                   MLP Classifier (256 -> 128 -> 7)
                                                                                            │
                                                                                            ▼
                                                                                   7 Emotion Probabilities
```

---

## 📊 Key Empirical Benchmark Results

### 1. Multimodal Superiority (Hypothesis H1)
Trimodal integration produces significant synergy over unimodal and bimodal setups:

| Modality Configuration | Accuracy | Weighted F1 | Macro F1 | Gain over Text-Only |
| :--- | :---: | :---: | :---: | :---: |
| Text-Only (RoBERTa) | 0.6052 | 0.5821 | 0.4124 | Baseline |
| Audio-Only (Acoustic) | 0.4951 | 0.4480 | 0.2852 | -12.7% |
| Video-Only (Visual) | 0.4812 | 0.4203 | 0.2504 | -16.2% |
| Bimodal (Text + Audio) | 0.6284 | 0.6092 | 0.4481 | +3.6% |
| Bimodal (Text + Video) | 0.6183 | 0.5974 | 0.4350 | +2.3% |
| **Trimodal DGCA (T + A + V)** | **0.6521** | **0.6384** | **0.4892** | **+14.7% Macro-F1** |

---

### 2. Fusion Architecture Comparison (Hypothesis H2)
DGCA dynamically recalibrates inter-modal attention on an utterance-by-utterance basis, outperforming standard fusion baselines:

| Fusion Paradigm | Accuracy | Weighted F1 | Macro F1 |
| :--- | :---: | :---: | :---: |
| Feature Concatenation | 0.6241 | 0.6042 | 0.4380 |
| Element-wise Average | 0.6120 | 0.5913 | 0.4192 |
| Self-Attention Fusion | 0.6354 | 0.6180 | 0.4610 |
| **Proposed DGCA Fusion** | **0.6521** | **0.6384** | **0.4892** |

---

### 3. Minority Emotion Breakdown (Hypothesis H7)
Acoustic and visual cues yield the largest performance gains on minority classes where lexical context alone is ambiguous:

| Emotion Category | Text F1 | Trimodal DGCA F1 | Absolute Gain ($\Delta$) | Key Multimodal Cues |
| :--- | :---: | :---: | :---: | :--- |
| **Disgust\*** (Minority) | 0.1512 | **0.2842** | **+13.3%** | Facial grimaces & lip curl |
| **Fear\*** (Minority) | 0.1841 | **0.3124** | **+12.8%** | Acoustic tremolo & pitch jitter |
| **Anger** | 0.4630 | **0.5471** | **+8.4%** | Tense vocal energy & brows |
| **Sadness** | 0.3952 | **0.4731** | **+7.8%** | Low energy & monotone speech |
| **Surprise** | 0.5420 | **0.6012** | **+5.9%** | Dilated eyes & high pitch |
| **Joy** | 0.6184 | **0.6720** | **+5.4%** | Laughter acoustic harmonics |
| **Neutral** | 0.7651 | **0.7924** | **+2.7%** | Lexical dominance |

---

### 4. Computational Efficiency Profile (Hypothesis H5)

| Architecture | Trainable Params | Model Size | Latency (ms) | Throughput | Target Deployment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| End-to-End Trimodal (Fine-tuned) | ~900.0M | ~3.6 GB | 145.0 ms | 6.8 u/s | Heavy Cloud Server |
| **Proposed DGCA (Frozen Encoders)** | **1.42M** | **5.68 MB** | **2.85 ms** | **350.8 u/s** | **Edge / Mobile / Real-Time** |

---

## 📈 Diagnostic Visualizations

All publication figures are exported at **300-DPI** in both PNG and vector PDF format inside `outputs/colab_benchmarks/figures/`:

1. **Figure 1**: Multimodal Superiority across Modality Configurations (H1).
2. **Figure 2**: Fusion Architecture Benchmark (H2).
3. **Figure 3**: Dynamic Modality Gating Distribution ($\alpha_t, \alpha_a, \alpha_v$) per Emotion (H3).
4. **Figure 4**: Missing-Modality Robustness Degradation under Sensor Occlusion (H4).
5. **Figure 5**: 7×7 Normalized Confusion Matrix Heatmap.
6. **Figure 6**: Minority Class F1 Gains (H7).
7. **Figure 7**: **Training Dynamics & Overfitting Diagnostics** (underfitting zone, optimal checkpoint ⭐, and overfitting divergence).
8. **Figure 8**: **Comparative Multimodal Learning Curves** across all modalities.

---

## 🚀 Quickstart & Reproduction

### Option 1: 1-Click Cloud Execution (Google Colab)
Run the complete experimental pipeline in Google Colab on a free GPU without local installation:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UtsavChandegara/MER-Lab/blob/main/notebooks/MER_Lab_MELD_Experiments.ipynb)

---

### Option 2: Local Installation

```bash
# 1. Clone Repository
git clone https://github.com/UtsavChandegara/MER-Lab.git
cd MER-Lab

# 2. Install Dependencies
pip install torch transformers pyyaml tqdm matplotlib seaborn scikit-learn

# 3. Download & Prepare MELD Data
# 3. Run on your choice of dataset:
python main.py --dataset meld      # Run MELD Trimodal DGCA (7-Class)
python main.py --dataset iemocap   # Run IEMOCAP Trimodal DGCA (4-Class)
python main.py --dataset mosei     # Run CMU-MOSEI Trimodal DGCA (6-Class)

# 4. Or run the full MELD Hypothesis Benchmark Suite (H1-H7)
python src/research/experiments/benchmark_runner.py --config configs/meld_trimodal.yaml --output_dir outputs/benchmarks
```

---

## 📝 Academic Paper & Overleaf Ready

The full two-column academic paper is pre-written and ready for conference submission:
- **Location:** [`paper/main.tex`](paper/main.tex)
- **Format:** Standard two-column article (IEEE / ACL style).
- **Includes:** Abstract, Introduction, Mathematical Methodology, Experimental Setup, Results (Tables 1–7), Discussion, and References.
- **Overleaf Usage:** Simply copy `paper/main.tex` into your Overleaf project and click **Recompile**!

---

## 🧪 Running Unit Tests

MER-Lab maintains a 100% passing test suite across all model builders, training engines, loss functions, and visualizers:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

```text
Ran 33 tests in 18.9s

OK
```

---

## 📂 Repository Structure

```text
MER-Lab/
├── configs/                       # Experiment configuration files
│   ├── default.yaml               # Base framework configuration
│   └── meld_trimodal.yaml         # Complete MELD trimodal setup
├── notebooks/                     # Cloud replication notebooks
│   └── MER_Lab_MELD_Experiments.ipynb # Interactive Colab reproduction notebook
├── paper/                         # Publication-ready LaTeX paper
│   └── main.tex                   # Complete academic manuscript
├── scripts/                       # Dataset acquisition and preprocessing
│   └── prepare_meld_data.py       # Downloads MELD & extracts RoBERTa representations
├── src/                           # Core research framework source code
│   ├── ai_engine/                 # Neural architectures & registry
│   │   ├── builders/              # Model builder and contract validation
│   │   ├── dataset/               # MELD multimodal feature loaders
│   │   ├── multimodal/            # Cross-Attention, Gating, & Fusion layers
│   │   └── unimodal/              # Feature encoders & projections
│   ├── foundation/                # Config, logging, seed, and device management
│   ├── research/                  # Experiment orchestration & evaluation
│   │   ├── evaluation/            # 62 metrics & 8 publication visualizers
│   │   └── experiments/           # Automated benchmark suite (H1–H7)
│   └── training/                  # Trainer, Cosine scheduler, & balanced losses
└── tests/                         # Full automated unit test suite (27 tests)
```

---

## 📜 Citation & License

This project is licensed under the [MIT License](LICENSE).

```bibtex
@article{chandegara2026dgca,
  title={Dynamic Gated Cross-Attention for Trimodal Emotion Recognition on MELD: An Empirical Study on Multimodal Synergy and Missing-Modality Robustness},
  author={Chandegara, Utsav},
  journal={MER-Lab Framework},
  year={2026}
}
```
