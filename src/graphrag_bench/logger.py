"""Logging subsystem for Graph RAG Benchmark.

Provides unified console and file logging with standardized formatting,
configurable verbosity, and automated directory initialization.
"""

import logging
import sys
from pathlib import Path

from graphrag_bench.config import Settings, get_settings

_INITIALIZED = False
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    settings: Settings | None = None,
    force_reconfigure: bool = False,
) -> logging.Logger:
    """Configure root and package logger with console and rotating file handlers.

    Args:
        settings: Optional Settings instance. Defaults to get_settings().
        force_reconfigure: If True, clear existing handlers and reconfigure.

    Returns:
        Configured root logger for the 'graphrag_bench' package.
    """
    global _INITIALIZED
    if _INITIALIZED and not force_reconfigure:
        return logging.getLogger("graphrag_bench")

    if settings is None:
        settings = get_settings()

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    pkg_logger = logging.getLogger("graphrag_bench")

    if force_reconfigure:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
        for handler in list(pkg_logger.handlers):
            pkg_logger.removeHandler(handler)

    formatter = logging.Formatter(fmt=DEFAULT_FORMAT, datefmt=DATE_FORMAT)

    # 1. Console Stream Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # 2. File Handler
    log_file_path = Path(settings.log_file).resolve()
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(
        filename=str(log_file_path),
        encoding="utf-8",
        mode="a",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Attach handlers to pkg_logger
    pkg_logger.setLevel(log_level)
    pkg_logger.propagate = False  # Avoid duplicate messages to root

    # Only add handlers if not already present
    existing_handler_types = [type(h) for h in pkg_logger.handlers]
    if logging.StreamHandler not in existing_handler_types:
        pkg_logger.addHandler(console_handler)
    if logging.FileHandler not in existing_handler_types:
        pkg_logger.addHandler(file_handler)

    _INITIALIZED = True
    pkg_logger.debug(
        "Logger initialized. Level: %s, File: %s",
        settings.log_level,
        log_file_path,
    )
    return pkg_logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Retrieve a logger namespaced under 'graphrag_bench'.

    Args:
        name: Sub-module name (e.g., 'ollama_client', 'health').

    Returns:
        logging.Logger instance.
    """
    if not _INITIALIZED:
        setup_logger()

    if not name or name == "graphrag_bench":
        return logging.getLogger("graphrag_bench")

    if not name.startswith("graphrag_bench."):
        name = f"graphrag_bench.{name}"

    return logging.getLogger(name)
