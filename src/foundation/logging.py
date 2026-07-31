"""Logging Infrastructure Manager for MER-Lab (MER-RULE-128..131, MER-RULE-207).

Provides standard, structured logging across all framework components.
"""

import sys
import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "MERLab",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configures and returns a logger instance with console and optional file handlers.
    
    Args:
        name: Name of the logger module.
        log_file: Optional path to log file.
        level: Logging level (default: logging.INFO).
        
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "MERLab") -> logging.Logger:
    """Gets an existing logger instance or returns a default logger."""
    return logging.getLogger(name)
