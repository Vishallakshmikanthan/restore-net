"""Logging utilities for image restoration."""

import os
import sys
from pathlib import Path
import yaml

# Remove local directory from sys.path to ensure standard library logging is loaded
current_dir = str(Path(__file__).resolve().parent)
if current_dir in sys.path:
    sys.path.remove(current_dir)

import logging


def get_config(path: str) -> dict:
    """Loads a YAML configuration file and returns it as a dictionary.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Parsed dictionary from the YAML file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config is not None else {}


def setup_logger(name: str = "RestoreNet", log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Sets up a standard logger with stream and optional file output handlers.

    Args:
        name (str): Logger name.
        log_file (str, optional): Path to output log file.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger
