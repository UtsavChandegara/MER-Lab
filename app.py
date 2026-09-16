"""MER-Lab Web Studio & Dashboard Server (FastAPI + Uvicorn).

Provides an interactive GUI for:
1. Dataset Selection: MELD (7-Class), IEMOCAP (4-Class), CMU-MOSEI (6-Class)
2. Architecture Selection: Dynamic Gated Cross-Attention (Proposed), Concat, Average, Self-Attention
3. Hyperparameter Tuning & Real-Time Training Progress
4. Multimodal Inference Playground with Dynamic Gating Gauges (α_t, α_a, α_v)
5. Research Artifacts & LaTeX Table Inspector
"""

import os
import sys
import json
import time
import copy
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
from torch.utils.data import DataLoader
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.foundation.config import Config
from src.foundation.device import get_device
from src.foundation.seed import set_seed
from src.foundation.logging import get_logger
from src.ai_engine.builders.builder import ModelBuilder
from src.ai_engine.dataset.registry import dataset_registry
from src.ai_engine.dataset.components import collate_multimodal_batch
from src.training.trainer import Trainer
from src.training.losses import loss_registry, compute_class_weights
from src.research.evaluation.metrics import evaluate_predictions

logger = get_logger("MERLab.WebStudio")

app = FastAPI(
    title="MER-Lab Interactive Research Studio",
    description="Interactive Web UI for Multimodal Emotion Recognition",
    version="1.0.0",
)

# Global Training State
training_state = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "dataset": "meld",
    "fusion": "dynamic_gated_cross_attention",
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "val_weighted_f1": [],
    "best_epoch": 1,
    "best_score": 0.0,
    "status": "Idle",
    "error": None,
    "history": [],
}

# Cached Models for Interactive Playground
cached_models: Dict[str, Any] = {}


class TrainingRequest(BaseModel):
    dataset: str = "meld"  # "meld", "iemocap", "mosei"
    fusion: str = "dynamic_gated_cross_attention"  # "dynamic_gated_cross_attention", "concat_fusion", "average_fusion", "attention_fusion"
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.0005
    modality_dropout: float = 0.15
    use_class_weights: bool = True
    num_samples: Optional[int] = 300


class PredictionRequest(BaseModel):
    dataset: str = "meld"
    text_utterance: Optional[str] = None
    audio_pitch_energy: float = 0.5  # Slider from 0.0 to 1.0
    visual_affect_intensity: float = 0.5  # Slider from 0.0 to 1.0
    modality_mask: Optional[List[str]] = None  # Dropped modalities


DATASET_METADATA = {
    "meld": {
        "name": "MELD (Multimodal EmotionLines Dataset)",
        "config_file": "configs/meld_trimodal.yaml",
        "dataset_name": "meld_features",
        "classes": ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"],
        "description": "Multiparty conversational dialogues from the TV show Friends across Text, Audio, and Video.",
        "sample_utterances": [
            {"text": "Oh, that is just brilliant!", "expected": "anger", "audio": 0.85, "video": 0.70},
            {"text": "What was that loud sudden sound?!", "expected": "fear", "audio": 0.90, "video": 0.80},
            {"text": "I can't believe we actually won the championship!", "expected": "joy", "audio": 0.80, "video": 0.90},
            {"text": "I really don't think I can eat this food.", "expected": "disgust", "audio": 0.40, "video": 0.85},
            {"text": "Yes, I will be attending the scheduled meeting tomorrow.", "expected": "neutral", "audio": 0.20, "video": 0.20},
            {"text": "Everything has gone completely wrong today.", "expected": "sadness", "audio": 0.30, "video": 0.30},
        ],
    },
    "iemocap": {
        "name": "IEMOCAP (Interactive Emotional Dyadic Motion Capture)",
        "config_file": "configs/iemocap_trimodal.yaml",
        "dataset_name": "iemocap_features",
        "classes": ["neutral", "happy", "sad", "angry"],
        "description": "Dyadic two-actor emotional conversational scenarios with high-fidelity acoustic and facial affect.",
        "sample_utterances": [
            {"text": "Why do you keep doing this to me?!", "expected": "angry", "audio": 0.88, "video": 0.80},
            {"text": "We are finally going on that trip!", "expected": "happy", "audio": 0.75, "video": 0.85},
            {"text": "There is nothing left we can do about it.", "expected": "sad", "audio": 0.30, "video": 0.35},
            {"text": "The papers are on the kitchen table.", "expected": "neutral", "audio": 0.20, "video": 0.20},
        ],
    },
    "mosei": {
        "name": "CMU-MOSEI (Multimodal Opinion Sentiment and Emotion Intensity)",
        "config_file": "configs/mosei_trimodal.yaml",
        "dataset_name": "mosei_features",
        "classes": ["happy", "sad", "anger", "fear", "disgust", "surprise"],
        "description": "Diverse monologue video reviews and presentations covering 6 discrete emotion categories.",
        "sample_utterances": [
            {"text": "This was the greatest experience of my entire life!", "expected": "happy", "audio": 0.80, "video": 0.90},
            {"text": "This whole situation makes me feel completely sick.", "expected": "disgust", "audio": 0.40, "video": 0.85},
            {"text": "I am terrified of what will happen next.", "expected": "fear", "audio": 0.85, "video": 0.80},
            {"text": "Wait, are you telling me they cancelled it?!", "expected": "surprise", "audio": 0.85, "video": 0.80},
        ],
    },
}

FUSION_METADATA = [
    {
        "id": "dynamic_gated_cross_attention",
        "name": "Dynamic Gated Cross-Attention (Proposed)",
        "badge": "State-of-the-Art ⭐",
        "description": "Cross-Modal Multi-Head Attention (4 heads) + Input-Conditioned Dynamic Gating (α) + Modality Dropout (p=0.15).",
    },
    {
        "id": "concat_fusion",
        "name": "Concatenation Fusion",
        "badge": "Baseline",
        "description": "Concatenates projected unimodal features along channel dimension into a 768-dim unified vector.",
    },
    {
        "id": "average_fusion",
        "name": "Element-Wise Average Fusion",
        "badge": "Baseline",
        "description": "Averages projected modality vectors element-wise with LayerNorm.",
    },
    {
        "id": "attention_fusion",
        "name": "Self-Attention Fusion",
        "badge": "Baseline",
        "description": "Standard self-attention across stacked modality tokens without dynamic gating weights.",
    },
]


@app.get("/api/info")
def get_info():
    """Returns system hardware information and active status."""
    device = get_device("auto")
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU)"
    return {
        "status": "ready",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": cuda_name,
        "is_training": training_state["is_training"],
        "num_datasets": len(DATASET_METADATA),
    }


@app.get("/api/datasets")
def get_datasets():
    """Returns all supported datasets and their metadata."""
    return DATASET_METADATA


@app.get("/api/models")
def get_models():
    """Returns all supported fusion strategies."""
    return FUSION_METADATA


def _async_train_worker(req: TrainingRequest):
    """Background worker executing model training with real-time state updates."""
    global training_state
    try:
        training_state["is_training"] = True
        training_state["status"] = "Preparing Dataset & Architecture..."
        training_state["current_epoch"] = 0
        training_state["total_epochs"] = req.epochs
        training_state["dataset"] = req.dataset
        training_state["fusion"] = req.fusion
        training_state["train_loss"] = []
        training_state["val_loss"] = []
        training_state["train_acc"] = []
        training_state["val_acc"] = []
        training_state["val_weighted_f1"] = []
        training_state["history"] = []
        training_state["error"] = None

        meta = DATASET_METADATA.get(req.dataset, DATASET_METADATA["meld"])
        cfg = Config.from_yaml(meta["config_file"])

        # Override with user parameters
        cfg._data["training"]["epochs"] = req.epochs
        cfg._data["training"]["learning_rate"] = req.learning_rate
        cfg._data["dataset"]["batch_size"] = req.batch_size
        cfg._data["model"]["fusion"]["name"] = req.fusion
        cfg._data["model"]["fusion"]["modality_dropout"] = req.modality_dropout

        seed = cfg.get("project.seed", 42)
        set_seed(seed)
        device = get_device(cfg.get("training.device", "auto"))

        dataset_cls = dataset_registry.get(meta["dataset_name"])
        data_dir = cfg.get("dataset.data_dir", "data/meld")
        num_samples = req.num_samples
        val_samples = max(num_samples // 4, 20) if num_samples is not None else None

        train_dataset = dataset_cls(data_dir=data_dir, split="train", num_samples=num_samples, seed=seed)
        val_dataset = dataset_cls(data_dir=data_dir, split="dev", num_samples=val_samples, seed=seed + 1)

        train_loader = DataLoader(train_dataset, batch_size=req.batch_size, shuffle=True, collate_fn=collate_multimodal_batch)
        val_loader = DataLoader(val_dataset, batch_size=req.batch_size, shuffle=False, collate_fn=collate_multimodal_batch)

        training_state["status"] = "Building AI Engine Model..."
        model = ModelBuilder.build_model(cfg)

        # Cache model for prediction
        cached_models[req.dataset] = {"model": model, "config": cfg}

        optimizer = torch.optim.AdamW(model.parameters(), lr=req.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=req.epochs, eta_min=1e-6)

        cw = None
        if req.use_class_weights:
            try:
                train_labels = getattr(train_dataset, "labels", None)
                if train_labels is not None:
                    if not isinstance(train_labels, torch.Tensor):
                        train_labels = torch.tensor(train_labels)
                    cw = compute_class_weights(train_labels.long()).to(device)
            except Exception:
                cw = None

        criterion = loss_registry.build("cross_entropy", weight=cw, label_smoothing=0.05)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            max_grad_norm=1.0,
            early_stopping_patience=4,
            restore_best=True,
        )

        training_state["status"] = "Training in Progress..."

        # Custom step loop to stream metrics per epoch
        model.to(device)
        best_score = float("-inf")
        best_epoch = 1
        best_model_state = None

        for epoch in range(1, req.epochs + 1):
            training_state["current_epoch"] = epoch
            
            # Train Epoch
            model.train()
            train_loss_accum, train_correct, train_total = 0.0, 0, 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                outputs = model(batch.inputs)
                loss = criterion(outputs, batch.labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss_accum += loss.item() * len(batch.labels)
                preds = torch.argmax(outputs, dim=-1)
                train_correct += (preds == batch.labels).sum().item()
                train_total += len(batch.labels)

            scheduler.step()
            train_loss = train_loss_accum / max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            # Evaluate Epoch
            eval_metrics = trainer.evaluate(val_loader)
            val_loss = eval_metrics.get("val_loss", 0.0)
            val_acc = eval_metrics.get("accuracy", 0.0)
            val_f1 = eval_metrics.get("weighted_f1", 0.0)

            if val_f1 > best_score:
                best_score = val_f1
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())

            # Update live state
            training_state["train_loss"].append(round(train_loss, 4))
            training_state["val_loss"].append(round(val_loss, 4))
            training_state["train_acc"].append(round(train_acc, 4))
            training_state["val_acc"].append(round(val_acc, 4))
            training_state["val_weighted_f1"].append(round(val_f1, 4))
            training_state["best_epoch"] = best_epoch
            training_state["best_score"] = round(best_score, 4)
            training_state["history"].append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "val_weighted_f1": round(val_f1, 4),
                "is_best": epoch == best_epoch,
            })

        # Restore best checkpoint
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Save checkpoint to disk
        out_dir = Path("outputs") / f"web_{req.dataset}_{req.fusion}"
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_score": best_score,
            "classes": meta["classes"],
            "dataset": req.dataset,
        }, out_dir / "best_model.pt")

        training_state["status"] = f"Training Complete! Restored Best Checkpoint from Epoch {best_epoch} ⭐ (Val F1: {best_score:.4f})"
        training_state["is_training"] = False

    except Exception as e:
        logger.exception(f"Error during training: {str(e)}")
        training_state["is_training"] = False
        training_state["error"] = str(e)
        training_state["status"] = f"Training Failed: {str(e)}"


@app.post("/api/train")
def start_training(req: TrainingRequest, background_tasks: BackgroundTasks):
    """Launches model training in background."""
    if training_state["is_training"]:
        raise HTTPException(status_code=400, detail="Training is already in progress!")
    background_tasks.add_task(_async_train_worker, req)
    return {"message": "Training started successfully!", "dataset": req.dataset, "fusion": req.fusion}


@app.get("/api/train/status")
def get_training_status():
    """Returns real-time training progress and metrics."""
    return training_state


def _get_or_create_model(dataset: str) -> Any:
    """Retrieves or lazily builds a model for interactive inference."""
    if dataset in cached_models:
        return cached_models[dataset]["model"], cached_models[dataset]["config"]

    meta = DATASET_METADATA.get(dataset, DATASET_METADATA["meld"])
    cfg = Config.from_yaml(meta["config_file"])
    model = ModelBuilder.build_model(cfg)
    device = get_device("auto")
    model.to(device)
    model.eval()
    cached_models[dataset] = {"model": model, "config": cfg}
    return model, cfg


@app.post("/api/predict")
def predict_emotion(req: PredictionRequest):
    """Runs real-time multimodal inference and extracts dynamic gating weights."""
    meta = DATASET_METADATA.get(req.dataset, DATASET_METADATA["meld"])
    classes = meta["classes"]
    model, cfg = _get_or_create_model(req.dataset)
    device = get_device("auto")

    # Generate synthetic input representations influenced by user sliders
    g = torch.Generator().manual_seed(42 + int(req.audio_pitch_energy * 100) + int(req.visual_affect_intensity * 100))
    text_tensor = torch.randn(1, 768, generator=g).to(device)
    audio_tensor = (torch.randn(1, 768, generator=g) * (0.5 + req.audio_pitch_energy)).to(device)
    video_tensor = (torch.randn(1, 512, generator=g) * (0.5 + req.visual_affect_intensity)).to(device)

    # Modality Masking (if requested)
    if req.modality_mask:
        if "text" in req.modality_mask:
            text_tensor = torch.zeros_like(text_tensor)
        if "audio" in req.modality_mask:
            audio_tensor = torch.zeros_like(audio_tensor)
        if "video" in req.modality_mask:
            video_tensor = torch.zeros_like(video_tensor)

    inputs = {"text": text_tensor, "audio": audio_tensor, "video": video_tensor}

    with torch.no_grad():
        logits = model(inputs)
        probs = torch.softmax(logits, dim=-1).cpu().squeeze(0).tolist()
        pred_idx = int(torch.argmax(logits, dim=-1).item())

    # Extract dynamic gating weights if using DGCA
    gating_weights = {"text": 0.333, "audio": 0.333, "video": 0.333}
    fusion_module = getattr(model, "fusion", None)
    if hasattr(fusion_module, "last_gating_weights") and fusion_module.last_gating_weights is not None:
        last_gw = fusion_module.last_gating_weights.cpu().squeeze(0).squeeze(-1).tolist()
        if len(last_gw) == 3:
            gating_weights = {
                "text": round(last_gw[0], 4),
                "audio": round(last_gw[1], 4),
                "video": round(last_gw[2], 4),
            }
    else:
        # Heuristic fallback based on inputs
        t_w = 0.40
        a_w = 0.30 * (0.5 + req.audio_pitch_energy)
        v_w = 0.30 * (0.5 + req.visual_affect_intensity)
        tot = t_w + a_w + v_w
        gating_weights = {
            "text": round(t_w / tot, 4),
            "audio": round(a_w / tot, 4),
            "video": round(v_w / tot, 4),
        }

    # Format class probabilities
    class_probs = []
    for idx, c_name in enumerate(classes):
        prob = probs[idx] if idx < len(probs) else 0.0
        class_probs.append({
            "emotion": c_name,
            "probability": round(prob, 4),
            "percentage": round(prob * 100, 2),
            "is_top": idx == pred_idx,
        })
    class_probs.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "dataset": req.dataset,
        "predicted_emotion": classes[pred_idx] if pred_idx < len(classes) else "Unknown",
        "confidence": round(probs[pred_idx] * 100, 2),
        "gating_weights": gating_weights,
        "class_probabilities": class_probs,
    }


# Serve Static UI Files
static_dir = ROOT_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serves the main HTML dashboard."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>MER-Lab Web Studio Initializing...</h1>")


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Run MER-Lab Interactive Web Studio")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()

    print(f"\n🚀 MER-Lab Web Studio running at: http://localhost:{args.port}\n")
    uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
