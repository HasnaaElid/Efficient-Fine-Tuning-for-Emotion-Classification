"""
============================================
Title: Logging Utility for GoEmotions Project
Author: Hasnaa elidrissi
Date: 2025-11-15
Description:
    Provides a small wrapper around Python's logging module to ensure
    consistent, timestamped logging across the project. Supports both
    console output and optional log-to-file behavior. Used throughout
    data processing, training, calibration, and explainability scripts.

Attribution:
    - Built on Python's standard logging module.
    - Pattern inspired by common ML pipeline logging utilities
============================================
"""
import logging
from typing import Optional

def get_logger(name: str = "goemo", level: int = logging.INFO, to_file: Optional[str] = None) -> logging.Logger:
    """
    Create a configured logger for consistent, clean console/file logs.

    Args:
        name: Logger name.
        level: Logging level (e.g., logging.INFO).
        to_file: Optional path to write logs.

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if to_file:
            fh = logging.FileHandler(to_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger
