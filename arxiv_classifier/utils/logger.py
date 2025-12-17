"""Logging configuration utilities."""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "arxiv_classifier",
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """Setup and configure logger.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional path to log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get logger instance.

    Args:
        name: Optional logger name, defaults to 'arxiv_classifier'

    Returns:
        Logger instance
    """
    if name is None:
        name = "arxiv_classifier"
    return logging.getLogger(name)
