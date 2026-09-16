"""Dataset implementations and batch collation utilities (MER-RULE-047, MER-RULE-214..216).
"""

from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import DataLoader

from src.ai_engine.dataset.contracts import BaseDataset, MultimodalSample, MultimodalBatch
from src.ai_engine.dataset.registry import dataset_registry


def collate_multimodal_batch(batch: List[MultimodalSample]) -> MultimodalBatch:
    """Collates a list of MultimodalSample objects into a single MultimodalBatch."""
    sample_ids = [s.sample_id for s in batch]
    metadata = [s.metadata or {} for s in batch]
    
    inputs: Dict[str, torch.Tensor] = {}
    
    # Text modality
    if batch[0].text is not None:
        if isinstance(batch[0].text, torch.Tensor):
            inputs['text'] = torch.stack([s.text for s in batch])
        elif isinstance(batch[0].text, str):
            # Pass list of strings for downstream tokenizer/encoder handling
            inputs['text'] = [s.text for s in batch]
            
    # Video modality
    if batch[0].video is not None:
        if isinstance(batch[0].video, torch.Tensor):
            inputs['video'] = torch.stack([s.video for s in batch])

    # Audio modality
    if batch[0].audio is not None:
        if isinstance(batch[0].audio, torch.Tensor):
            inputs['audio'] = torch.stack([s.audio for s in batch])

    labels = None
    if batch[0].label is not None:
        labels = torch.tensor([s.label for s in batch], dtype=torch.long)

    return MultimodalBatch(
        sample_ids=sample_ids,
        inputs=inputs,
        labels=labels,
        metadata=metadata,
    )


@dataset_registry.register("synthetic_meld")
class SyntheticMELDDataset(BaseDataset):
    """Synthetic dataset generating synthetic MELD-like samples for rapid zero-dependency experiments."""

    EMOTIONS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]

    def __init__(
        self,
        num_samples: int = 100,
        text_dim: int = 768,
        video_dim: int = 512,
        audio_dim: int = 768,
        seed: int = 42,
    ):
        super().__init__()
        self._num_samples = num_samples
        self._text_dim = text_dim
        self._video_dim = video_dim
        self._audio_dim = audio_dim
        
        g = torch.Generator().manual_seed(seed)
        self._labels = torch.randint(0, len(self.EMOTIONS), (num_samples,), generator=g).tolist()

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> MultimodalSample:
        # Generate reproducible dummy features for testing pipeline
        g = torch.Generator().manual_seed(index)
        text_data = f"Sample text utterance {index} expressing emotion."
        video_data = torch.randn(self._video_dim, generator=g)
        audio_data = torch.randn(self._audio_dim, generator=g)

        return MultimodalSample(
            sample_id=f"meld_synth_{index:04d}",
            text=text_data,
            video=video_data,
            audio=audio_data,
            label=self._labels[index],
            metadata={"dialogue_id": index // 10, "utterance_id": index % 10},
        )

    @property
    def num_classes(self) -> int:
        return len(self.EMOTIONS)


@dataset_registry.register("meld_features")
class MELDFeatureDataset(BaseDataset):
    """MELD Trimodal Feature Dataset.
    
    Loads pre-extracted feature tensors (RoBERTa for text, WavLM for audio, CLIP/OpenFace for video).
    Falls back gracefully to synthetic tensor simulation if the real feature file is not yet downloaded.
    """

    EMOTIONS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]

    def __init__(
        self,
        data_dir: str = "data/meld",
        split: str = "train",
        num_samples: Optional[int] = None,
        text_dim: int = 768,
        audio_dim: int = 768,
        video_dim: int = 512,
        seed: int = 42,
    ):
        super().__init__()
        import os
        from pathlib import Path
        from src.foundation.logging import get_logger
        logger = get_logger("MERLab.AIEngine.Dataset")

        self.data_dir = Path(data_dir)
        self.split = split
        self._text_dim = text_dim
        self._audio_dim = audio_dim
        self._video_dim = video_dim

        feature_file = self.data_dir / f"{split}_features.pt"
        if not feature_file.exists():
            feature_file = self.data_dir / f"meld_{split}.pt"

        if feature_file.exists():
            logger.info(f"Loading pre-extracted MELD features from '{feature_file}'...")
            data = torch.load(feature_file, map_location="cpu")
            self.text_features = data["text"]
            self.audio_features = data["audio"]
            self.video_features = data["video"]
            self.labels = data["labels"]
            self.sample_ids = data.get("sample_ids", [f"meld_{split}_{i}" for i in range(len(self.labels))])
            if num_samples is not None and num_samples < len(self.labels):
                self.text_features = self.text_features[:num_samples]
                self.audio_features = self.audio_features[:num_samples]
                self.video_features = self.video_features[:num_samples]
                self.labels = self.labels[:num_samples]
                self.sample_ids = self.sample_ids[:num_samples]
            self._len = len(self.labels)
            logger.info(f"Loaded {self._len} real MELD samples for split '{split}'.")
        else:
            # Fallback to simulated feature tensors
            self._len = num_samples or (500 if split == "train" else 100)
            logger.warning(
                f"MELD feature file not found at '{feature_file}'. "
                f"Simulating {self._len} reproducible trimodal feature tensors for split '{split}'."
            )
            g = torch.Generator().manual_seed(seed + (0 if split == "train" else 100))
            self.text_features = torch.randn(self._len, self._text_dim, generator=g)
            self.audio_features = torch.randn(self._len, self._audio_dim, generator=g)
            self.video_features = torch.randn(self._len, self._video_dim, generator=g)
            self.labels = torch.randint(0, len(self.EMOTIONS), (self._len,), generator=g)
            self.sample_ids = [f"meld_sim_{split}_{i:04d}" for i in range(self._len)]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> MultimodalSample:
        return MultimodalSample(
            sample_id=self.sample_ids[index],
            text=self.text_features[index],
            audio=self.audio_features[index],
            video=self.video_features[index],
            label=int(self.labels[index].item() if isinstance(self.labels[index], torch.Tensor) else self.labels[index]),
            metadata={"split": self.split, "index": index},
        )

    @property
    def num_classes(self) -> int:
        return len(self.EMOTIONS)


@dataset_registry.register("iemocap_features")
class IEMOCAPFeatureDataset(BaseDataset):
    """IEMOCAP Trimodal Feature Dataset (4-Class Emotion Classification).
    
    Standard 4-class conversational emotion categories: Neutral, Happy/Excited, Sad, Angry.
    Loads pre-extracted feature tensors or simulates reproducible tensors for zero-setup execution.
    """

    EMOTIONS = ["neutral", "happy", "sad", "angry"]

    def __init__(
        self,
        data_dir: str = "data/iemocap",
        split: str = "train",
        num_samples: Optional[int] = None,
        text_dim: int = 768,
        audio_dim: int = 768,
        video_dim: int = 512,
        seed: int = 42,
    ):
        super().__init__()
        from pathlib import Path
        from src.foundation.logging import get_logger
        logger = get_logger("MERLab.AIEngine.Dataset")

        self.data_dir = Path(data_dir)
        self.split = split
        self._text_dim = text_dim
        self._audio_dim = audio_dim
        self._video_dim = video_dim

        feature_file = self.data_dir / f"{split}_features.pt"
        if not feature_file.exists():
            feature_file = self.data_dir / f"iemocap_{split}.pt"

        if feature_file.exists():
            logger.info(f"Loading pre-extracted IEMOCAP features from '{feature_file}'...")
            data = torch.load(feature_file, map_location="cpu")
            self.text_features = data["text"]
            self.audio_features = data["audio"]
            self.video_features = data["video"]
            self.labels = data["labels"]
            self.sample_ids = data.get("sample_ids", [f"iemocap_{split}_{i}" for i in range(len(self.labels))])
            if num_samples is not None and num_samples < len(self.labels):
                self.text_features = self.text_features[:num_samples]
                self.audio_features = self.audio_features[:num_samples]
                self.video_features = self.video_features[:num_samples]
                self.labels = self.labels[:num_samples]
                self.sample_ids = self.sample_ids[:num_samples]
            self._len = len(self.labels)
            logger.info(f"Loaded {self._len} real IEMOCAP samples for split '{split}'.")
        else:
            self._len = num_samples or (400 if split == "train" else 100)
            logger.warning(
                f"IEMOCAP feature file not found at '{feature_file}'. "
                f"Simulating {self._len} reproducible trimodal feature tensors for split '{split}'."
            )
            g = torch.Generator().manual_seed(seed + (200 if split == "train" else 300))
            self.text_features = torch.randn(self._len, self._text_dim, generator=g)
            self.audio_features = torch.randn(self._len, self._audio_dim, generator=g)
            self.video_features = torch.randn(self._len, self._video_dim, generator=g)
            self.labels = torch.randint(0, len(self.EMOTIONS), (self._len,), generator=g)
            self.sample_ids = [f"iemocap_sim_{split}_{i:04d}" for i in range(self._len)]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> MultimodalSample:
        return MultimodalSample(
            sample_id=self.sample_ids[index],
            text=self.text_features[index],
            audio=self.audio_features[index],
            video=self.video_features[index],
            label=int(self.labels[index].item() if isinstance(self.labels[index], torch.Tensor) else self.labels[index]),
            metadata={"split": self.split, "index": index, "dataset": "iemocap"},
        )

    @property
    def num_classes(self) -> int:
        return len(self.EMOTIONS)


@dataset_registry.register("mosei_features")
class CMUMOSEIFeatureDataset(BaseDataset):
    """CMU-MOSEI Trimodal Feature Dataset (6-Class Emotion Classification).
    
    Standard 6-class discrete emotion categories: Happy, Sad, Anger, Fear, Disgust, Surprise.
    Loads pre-extracted feature tensors or simulates reproducible tensors for zero-setup execution.
    """

    EMOTIONS = ["happy", "sad", "anger", "fear", "disgust", "surprise"]

    def __init__(
        self,
        data_dir: str = "data/mosei",
        split: str = "train",
        num_samples: Optional[int] = None,
        text_dim: int = 768,
        audio_dim: int = 768,
        video_dim: int = 512,
        seed: int = 42,
    ):
        super().__init__()
        from pathlib import Path
        from src.foundation.logging import get_logger
        logger = get_logger("MERLab.AIEngine.Dataset")

        self.data_dir = Path(data_dir)
        self.split = split
        self._text_dim = text_dim
        self._audio_dim = audio_dim
        self._video_dim = video_dim

        feature_file = self.data_dir / f"{split}_features.pt"
        if not feature_file.exists():
            feature_file = self.data_dir / f"mosei_{split}.pt"

        if feature_file.exists():
            logger.info(f"Loading pre-extracted CMU-MOSEI features from '{feature_file}'...")
            data = torch.load(feature_file, map_location="cpu")
            self.text_features = data["text"]
            self.audio_features = data["audio"]
            self.video_features = data["video"]
            self.labels = data["labels"]
            self.sample_ids = data.get("sample_ids", [f"mosei_{split}_{i}" for i in range(len(self.labels))])
            if num_samples is not None and num_samples < len(self.labels):
                self.text_features = self.text_features[:num_samples]
                self.audio_features = self.audio_features[:num_samples]
                self.video_features = self.video_features[:num_samples]
                self.labels = self.labels[:num_samples]
                self.sample_ids = self.sample_ids[:num_samples]
            self._len = len(self.labels)
            logger.info(f"Loaded {self._len} real CMU-MOSEI samples for split '{split}'.")
        else:
            self._len = num_samples or (500 if split == "train" else 100)
            logger.warning(
                f"CMU-MOSEI feature file not found at '{feature_file}'. "
                f"Simulating {self._len} reproducible trimodal feature tensors for split '{split}'."
            )
            g = torch.Generator().manual_seed(seed + (400 if split == "train" else 500))
            self.text_features = torch.randn(self._len, self._text_dim, generator=g)
            self.audio_features = torch.randn(self._len, self._audio_dim, generator=g)
            self.video_features = torch.randn(self._len, self._video_dim, generator=g)
            self.labels = torch.randint(0, len(self.EMOTIONS), (self._len,), generator=g)
            self.sample_ids = [f"mosei_sim_{split}_{i:04d}" for i in range(self._len)]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> MultimodalSample:
        return MultimodalSample(
            sample_id=self.sample_ids[index],
            text=self.text_features[index],
            audio=self.audio_features[index],
            video=self.video_features[index],
            label=int(self.labels[index].item() if isinstance(self.labels[index], torch.Tensor) else self.labels[index]),
            metadata={"split": self.split, "index": index, "dataset": "mosei"},
        )

    @property
    def num_classes(self) -> int:
        return len(self.EMOTIONS)


