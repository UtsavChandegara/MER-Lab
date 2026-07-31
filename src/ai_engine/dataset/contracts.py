"""Dataset Contracts and Sample Standardizers (MER-RULE-047, MER-RULE-069..071, MER-RULE-215).

Defines standardized data contracts across all datasets in MER-Lab.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod


@dataclass
class MultimodalSample:
    """Standardized single data sample returned by all MER-Lab datasets (MER-RULE-215)."""
    sample_id: str
    text: Optional[Any] = None
    video: Optional[Any] = None
    audio: Optional[Any] = None
    label: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MultimodalBatch:
    """Standardized collated batch used throughout training and inference."""
    sample_ids: List[str]
    inputs: Dict[str, torch.Tensor]  # e.g., {'text': tensor, 'video': tensor}
    labels: Optional[torch.Tensor] = None
    metadata: Optional[List[Dict[str, Any]]] = None

    def to(self, device: torch.device) -> "MultimodalBatch":
        """Move all batch tensor modalities and labels to specified compute device."""
        moved_inputs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in self.inputs.items()
        }
        moved_labels = self.labels.to(device) if self.labels is not None else None
        return MultimodalBatch(
            sample_ids=self.sample_ids,
            inputs=moved_inputs,
            labels=moved_labels,
            metadata=self.metadata,
        )


class BaseDataset(Dataset, ABC):
    """Abstract base class for all MER-Lab datasets (MER-RULE-059)."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> MultimodalSample:
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Returns the number of emotion classification targets."""
        pass
