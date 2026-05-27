"""utils/logger.py — Logging Setup

Creates a logger that simultaneously writes to:
  - Console (stdout)
  - A rotating file at `log_path`
"""

import logging
import os
import sys


def setup_logger(name: str, log_path: str) -> logging.Logger:
    """
    Configure and return a logger.

    Args:
        name     : logger name (e.g. "exp1_v1")
        log_path : absolute path to the log file

    Returns:
        logging.Logger instance with console + file handlers attached.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if called more than once
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- File handler ---
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # --- Console handler ---
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger
