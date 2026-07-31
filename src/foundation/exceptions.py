"""Domain-specific exception classes for MER-Lab (MER-RULE-139).

Every exception provides informative context, root cause explanation, and actionable recovery hints (MER-RULE-132).
"""

class MERLabException(Exception):
    """Base exception for all MER-Lab framework errors."""

    def __init__(self, message: str, hint: str = None):
        formatted_message = message
        if hint:
            formatted_message += f"\n💡 Recovery Hint: {hint}"
        super().__init__(formatted_message)
        self.message = message
        self.hint = hint


class ConfigurationError(MERLabException):
    """Raised when configuration validation or loading fails."""
    pass


class DatasetError(MERLabException):
    """Raised when dataset loading, sampling, or formatting fails."""
    pass


class ModelError(MERLabException):
    """Raised when model building, interface contract validation, or forward pass fails."""
    pass


class ContractError(MERLabException):
    """Raised when an interface contract (e.g. output shape or type) is violated (MER-RULE-072)."""
    pass


class TrainingError(MERLabException):
    """Raised during training pipeline execution errors."""
    pass


class RegistryError(MERLabException):
    """Raised when component registration or lookup fails."""
    pass
