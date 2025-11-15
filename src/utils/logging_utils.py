"""
Universal logging utilities for the project.
Provides quiet logging to file with optional console output for errors.
"""

import logging
import os
from pathlib import Path


def setup_logger(name: str = "pie_bench", log_file: str = "logs/pie_bench.log", level: int = logging.INFO):
    """
    Set up a logger that writes to file and optionally prints errors to console.

    Args:
        name: Logger name
        log_file: Path to log file (relative to project root)
        level: Logging level

    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler - logs everything
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    # Console handler - only for errors and above
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.ERROR)  # Only print errors
    logger.addHandler(console_handler)

    return logger


def log_error(logger: logging.Logger, message: str, exc_info=None):
    """
    Log an error with optional exception info.

    Args:
        logger: The logger instance
        message: Error message
        exc_info: Exception info (from sys.exc_info())
    """
    logger.error(message, exc_info=exc_info)


def log_info(logger: logging.Logger, message: str):
    """
    Log an info message.

    Args:
        logger: The logger instance
        message: Info message
    """
    logger.info(message)
