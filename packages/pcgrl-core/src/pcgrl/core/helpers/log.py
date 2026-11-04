from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: str = "info", log_file: Path | None = None) -> logging.Logger:
    """Configure the root ``pcgrl`` logger with stream and optional file output.

    Args:
        level: Log level name (e.g. ``info``, ``debug``).
        log_file: Optional file path for persistent logs.

    Returns:
        Configured ``pcgrl`` logger instance.
    """
    logger = logging.getLogger("pcgrl")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    """Return the shared ``pcgrl`` logger."""
    return logging.getLogger("pcgrl")
