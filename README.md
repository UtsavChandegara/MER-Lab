# 🧪 MER-Lab: Multimodal Emotion Recognition Research Framework

> **"Build a Research Framework, Not Just a Model."** — *MER-RULE-001*

**MER-Lab** is a modular, research-oriented Python framework designed for systematic benchmarking, development, and evaluation of multimodal fusion strategies in Emotion Recognition (MER).

---

## 📌 Table of Contents

- [✨ Key Features](#-key-features)
- [🏛️ Architecture & Layer Separation](#️-architecture--layer-separation)
- [🚀 Quick Start & Installation](#-quick-start--installation)
  - [Prerequisites](#prerequisites)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Install Dependencies](#2-install-dependencies)
  - [3. Verify Installation](#3-verify-installation)
- [💡 Running Experiments](#-running-experiments)
  - [Run with Default Configuration](#run-with-default-configuration)
  - [Run with Custom Configuration](#run-with-custom-configuration)
- [⚙️ Configuration Guide](#️-configuration-guide)
- [🧩 Extending MER-Lab (Adding Custom Components)](#-extending-mer-lab-adding-custom-components)
  - [1. Add a Custom Encoder](#1-add-a-custom-encoder)
  - [2. Add a Custom Fusion Strategy](#2-add-a-custom-fusion-strategy)
  - [3. Add a Custom Dataset](#3-add-a-custom-dataset)
- [🧪 Running Unit Tests](#-running-unit-tests)
- [📂 Project Directory Structure](#-project-directory-structure)
- [📜 License](#-license)

---

## ✨ Key Features

- **Strict Modularity & Registry System**: Dynamic component lookup for Encoders, Datasets, Fusion strategies, and Classifiers using simple decorators (`@registry.register`).
- **Standardized Fusion Alignments**: All unimodal feature vectors are projected to a unified dimension ($D_{fusion} = 256$) before fusion for fair baseline comparison.
- **100% Config-Driven**: Zero hardcoded hyperparameters in Python logic. All parameters are managed via YAML configurations.
- **Multiple Fusion Strategies**: Built-in implementations for **Concatenation**, **Self-Attention**, and **Gated Fusion**.
- **Comprehensive Logging & Reproducibility**: Automated seed locking (CPU/GPU/Deterministic), structured logging, and automated experiment artifact saving.

---

## 🏛️ Architecture & Layer Separation

MER-Lab enforces a strict pipeline separation:
```text
Raw Data ➔ Dataset ➔ Encoder ➔ Projection Layer ➔ Fusion Module ➔ Classifier ➔ Prediction ➔ Evaluation
```

- **`src/foundation/`**: Core infrastructure including Logger, Config parser, Seed controller, Compute device selector, and Base Registries.
- **`src/ai_engine/`**: Implementations of Encoders, Projection layers, Multimodal Fusion algorithms, Classifiers, and Model Builders.
- **`src/training/`**: Training engine, custom loss functions, learning rate schedules, and lifecycle callbacks (e.g., Checkpointing).
- **`src/research/`**: Experiment Runner, metric evaluators (Accuracy, Weighted F1, Macro F1, Confusion Matrix), and artifact generators.

---

## 🚀 Quick Start & Installation

### Prerequisites
- **Python**: `>= 3.9`
- **PyTorch**: `>= 2.0.0`

### 1. Clone Repository

```bash
git clone https://github.com/UtsavChandegara/MER-Lab.git
cd MER-Lab
```

### 2. Install Dependencies

You can install `MER-Lab` in editable mode along with required dependencies:

```bash
pip install -e .
```

*Or install required packages manually:*
```bash
pip install torch pyyaml numpy tqdm
```

### 3. Verify Installation

Run the infrastructure readiness check:

```bash
python main.py --verify
```

*Expected Output:*
```text
[INFO] Executing MER-Lab Infrastructure Verification (Phase 1 & 2)...
[INFO] Loaded Configuration: project.name='MER-Lab-Default-Experiment'
[INFO] Infrastructure verification PASSED! Ready for experiment execution.
```

---

## 💡 Running Experiments

### Run with Default Configuration
To execute an experiment pipeline using the default parameters (`configs/default.yaml`):

```bash
python main.py
```

### Run with Custom Configuration
You can pass any custom YAML configuration file using the `--config` flag:

```bash
python main.py --config configs/my_experiment.yaml
```

*Console Output upon Completion:*
```text
================ FINAL EXPERIMENT RESULTS ================
Accuracy   : 0.8500
Weighted F1: 0.8421
Macro F1   : 0.8350
==========================================================
```

All experiment logs, metrics, configuration backups, and model checkpoints will be saved inside the `outputs/` directory.

---

## ⚙️ Configuration Guide

Experiments in MER-Lab are defined completely via YAML configuration files.

Here is an example configuration (`configs/default.yaml`):

```yaml
project:
  name: "MER-Lab-Default-Experiment"
  seed: 42
  output_dir: "outputs"

dataset:
  name: "synthetic_meld"
  data_dir: "data"
  batch_size: 16
  num_workers: 0
  modalities: ["text", "video"]
  num_samples: 100

model:
  fusion_dim: 256
  encoder:
    text:
      name: "mock_text_encoder"
      raw_dim: 768
      projection_type: "linear"
    video:
      name: "mock_video_encoder"
      raw_dim: 512
      projection_type: "linear"
  fusion:
    name: "concat_fusion"        # Options: concat_fusion, attention_fusion, gated_fusion
    projection_dim: 256
  classifier:
    name: "mlp_classifier"
    num_classes: 7
    hidden_dim: 128

training:
  epochs: 5
  learning_rate: 0.001
  weight_decay: 0.0001
  device: "auto"                 # Options: auto, cpu, cuda
  checkpoint_interval: 1

evaluation:
  metrics: ["accuracy", "weighted_f1", "macro_f1"]
  generate_confusion_matrix: true
```

---

## 🧩 Extending MER-Lab (Adding Custom Components)

MER-Lab's registry system allows adding new encoders, fusion mechanisms, or datasets without modifying core framework code.

### 1. Add a Custom Encoder
Decorate your class with `@encoder_registry.register("<encoder_name>")`:

```python
import torch
from src.ai_engine.unimodal.contracts import BaseEncoder
from src.ai_engine.unimodal.registry import encoder_registry

@encoder_registry.register("custom_audio_encoder")
class CustomAudioEncoder(BaseEncoder):
    def __init__(self, native_dim: int = 128):
        super().__init__()
        self._native_dim = native_dim
        self.fc = torch.nn.Linear(native_dim, native_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(inputs)

    @property
    def output_dim(self) -> int:
        return self._native_dim

    @property
    def modality(self) -> str:
        return "audio"
```

### 2. Add a Custom Fusion Strategy
Decorate your fusion module with `@fusion_registry.register("<fusion_name>")`:

```python
import torch
from typing import Dict
from src.ai_engine.multimodal.contracts import BaseFusion
from src.ai_engine.multimodal.registry import fusion_registry

@fusion_registry.register("my_custom_fusion")
class MyCustomFusion(BaseFusion):
    def __init__(self, projection_dim: int = 256):
        super().__init__()
        self._projection_dim = projection_dim

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Sum all projected modality feature tensors
        tensors = list(features.values())
        return torch.stack(tensors, dim=0).sum(dim=0)

    @property
    def input_dim(self) -> int:
        return self._projection_dim

    @property
    def output_dim(self) -> int:
        return self._projection_dim
```

### 3. Add a Custom Dataset
Decorate your dataset class with `@dataset_registry.register("<dataset_name>")`:

```python
from torch.utils.data import Dataset
from src.ai_engine.dataset.registry import dataset_registry

@dataset_registry.register("my_dataset")
class MyCustomDataset(Dataset):
    def __init__(self, data_dir: str, **kwargs):
        super().__init__()
        # Load dataset files from data_dir

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return {
            "inputs": {"text": "sample text", "video": torch.randn(512)},
            "label": 0
        }
```

---

## 🧪 Running Unit Tests

MER-Lab includes a suite of unit tests verifying foundational utilities, model building, and end-to-end training pipelines.

Run tests using standard Python `unittest`:

```bash
python3 -m unittest discover tests
```

---

## 📂 Project Directory Structure

```text
MER-Lab/
├── configs/
│   └── default.yaml             # Default experiment configuration
├── src/
│   ├── ai_engine/
│   │   ├── builders/            # MERModel assembly logic
│   │   ├── dataset/             # Dataset contracts and registries
│   │   ├── multimodal/          # Fusion algorithms & Classifiers
│   │   └── unimodal/            # Modality Encoders & Projections
│   ├── foundation/              # Logging, Config, Seed, Device, Exceptions
│   ├── research/                # Experiment Runner & Evaluation Metrics
│   └── training/                # Trainer, Callbacks, Loss functions
├── tests/                       # Automated test suite
├── main.py                      # CLI entry point
├── pyproject.toml               # Project metadata & dependencies
├── README.md                    # Framework documentation
└── LICENSE                      # MIT License
```

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
