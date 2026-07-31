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
        seed: int = 42,
    ):
        super().__init__()
        self._num_samples = num_samples
        self._text_dim = text_dim
        self._video_dim = video_dim
        
        g = torch.Generator().manual_seed(seed)
        self._labels = torch.randint(0, len(self.EMOTIONS), (num_samples,), generator=g).tolist()

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> MultimodalSample:
        # Generate reproducible dummy features for testing pipeline
        g = torch.Generator().manual_seed(index)
        text_data = f"Sample text utterance {index} expressing emotion."
        video_data = torch.randn(self._video_dim, generator=g)

        return MultimodalSample(
            sample_id=f"meld_synth_{index:04d}",
            text=text_data,
            video=video_data,
            label=self._labels[index],
            metadata={"dialogue_id": index // 10, "utterance_id": index % 10},
        )

    @property
    def num_classes(self) -> int:
        return len(self.EMOTIONS)
