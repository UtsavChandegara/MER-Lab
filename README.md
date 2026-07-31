# MER-Lab: Multimodal Emotion Recognition Research Framework

> **Build a Research Framework, Not Just a Model.** (MER-RULE-001)

MER-Lab is a modular, research-oriented Python framework designed for systematic comparison and development of multimodal fusion strategies in emotion recognition.

---

## 🏛️ Design Philosophy & Governance

MER-Lab is governed by the **MER-Lab Master Rulebook** (243 core rules):
- **Framework First**: Prioritize fair baseline comparisons, modularity, and total reproducibility over single-model benchmark hunting.
- **Strict Layer Separation**: `Raw Data → Dataset → Encoder → Projection → Fusion → Classifier → Prediction → Evaluation`.
- **Standard Fusion Dimensions**: All unimodal feature representations are aligned by the projection layer to a unified dimension ($D_{fusion} = 256$) before reaching fusion modules (MER-RULE-033).
- **Configuration Controls Everything**: Zero hardcoded hyperparameters in Python source files (MER-RULE-034).

---

## 📂 Architecture

```text
src/
├── foundation/          # Domain-agnostic infrastructure (Logger, Config, Seed, Device, Registry)
├── ai_engine/           # Core AI modules (Datasets, Encoders, Projections, Fusion, Classifiers, Builders)
├── research/            # Research execution (Config schema, Experiment runner, Evaluation metrics)
├── training/            # Training engine (Trainer lifecycle, Callbacks, Losses, Optimizers)
└── development/         # Developer utilities & test helpers
```

---

## 🚀 Quick Start

### 1. Verify Bootstrap & Environment
```bash
python main.py --config configs/default.yaml
```

---

## 📜 License
MIT License
