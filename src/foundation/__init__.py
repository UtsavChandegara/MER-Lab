"""Foundation Layer package exports (MER-RULE-209..213).

Provides shared, domain-agnostic infrastructure for all higher layers.
"""

from src.foundation.exceptions import (
    MERLabException,
    ConfigurationError,
    DatasetError,
    ModelError,
    ContractError,
    TrainingError,
    RegistryError,
)
from src.foundation.logging import setup_logger, get_logger
from src.foundation.config import Config
from src.foundation.seed import set_seed
from src.foundation.device import get_device
from src.foundation.registry import Registry

__all__ = [
    "MERLabException",
    "ConfigurationError",
    "DatasetError",
    "ModelError",
    "ContractError",
    "TrainingError",
    "RegistryError",
    "setup_logger",
    "get_logger",
    "Config",
    "set_seed",
    "get_device",
    "Registry",
]
