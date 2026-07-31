"""Configuration Manager for MER-Lab (MER-RULE-034, MER-RULE-124..127, MER-RULE-206).

Ensures all framework behaviors and hyper-parameters are controlled via configuration files,
preventing hard-coded values in implementation files.
"""

from pathlib import Path
from typing import Any, Dict, Union
import yaml

from src.foundation.exceptions import ConfigurationError


class Config:
    """Immutable or dictionary-like wrapper around YAML configuration settings."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Config":
        """Loads configuration from a YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: '{config_path}'",
                hint=f"Ensure the path '{config_path}' exists and contains a valid YAML configuration.",
            )
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls(data)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to parse configuration YAML at '{config_path}': {str(e)}",
                hint="Check YAML syntax and formatting.",
            )

    def get(self, key_path: str, default: Any = None) -> Any:
        """Retrieves a value using dot-separated key notation (e.g., 'model.encoder.text.name')."""
        keys = key_path.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def to_dict(self) -> Dict[str, Any]:
        """Returns raw configuration dictionary for logging and serialization."""
        return self._data

    def __getitem__(self, key: str) -> Any:
        if key not in self._data:
            raise ConfigurationError(
                f"Missing required configuration top-level key: '{key}'",
                hint=f"Available keys are: {list(self._data.keys())}",
            )
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Config({self._data})"
