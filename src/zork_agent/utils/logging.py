"""Logging helpers for the scaffold.

TODO: add structured log sinks if experiment monitoring becomes more demanding.
"""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_path: Path, level: str = "INFO") -> logging.Logger:
    """Configure and return the project logger."""

    # TODO: separate CLI/user logs from verbose debug traces when real agents arrive.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("zork_agent")
    logger.setLevel(level.upper())
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

