"""Registry Pattern Implementation for MER-Lab (MER-RULE-015, MER-RULE-065, MER-RULE-212).

Enables plug-and-play architecture by allowing dynamic registration and retrieval of modules 
(Datasets, Encoders, Projections, Fusion methods, Classifiers, Losses) without editing framework source files.
"""

from typing import Any, Callable, Dict, Optional, Type
from src.foundation.exceptions import RegistryError
from src.foundation.logging import get_logger

logger = get_logger("MERLab.Foundation.Registry")


class Registry:
    """A generic registry to register and instantiate named classes dynamically."""

    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type[Any]] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, name: Optional[str] = None) -> Callable:
        """Decorator to register a module class.
        
        Usage:
            @dataset_registry.register("meld")
            class MELDDataset(BaseDataset):
                ...
        """
        def _register(cls: Type[Any]) -> Type[Any]:
            key = name if name is not None else cls.__name__
            if key in self._module_dict:
                logger.warning(
                    f"Module '{key}' already registered in registry '{self._name}'. Overwriting."
                )
            self._module_dict[key.lower()] = cls
            logger.debug(f"Registered '{key.lower()}' into registry '{self._name}'.")
            return cls

        return _register

    def get(self, name: str) -> Type[Any]:
        """Retrieves a registered module class by name."""
        key = name.lower()
        if key not in self._module_dict:
            available = list(self._module_dict.keys())
            raise RegistryError(
                f"Module '{name}' not found in registry '{self._name}'.",
                hint=f"Available components in '{self._name}': {available}",
            )
        return self._module_dict[key]

    def build(self, name: str, *args, **kwargs) -> Any:
        """Retrieves registered class by name and instantiates it with provided arguments."""
        cls = self.get(name)
        try:
            return cls(*args, **kwargs)
        except Exception as e:
            raise RegistryError(
                f"Failed to instantiate module '{name}' from registry '{self._name}': {str(e)}",
                hint=f"Ensure instantiation arguments match constructor signature for '{cls.__name__}'.",
            )

    def list_modules(self) -> Dict[str, Type[Any]]:
        return dict(self._module_dict)

    def __repr__(self) -> str:
        return f"Registry(name='{self._name}', modules={list(self._module_dict.keys())})"
