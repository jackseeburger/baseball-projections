"""Common utility functions."""
import logging
from pathlib import Path

def setup_logging(name: str, level=logging.INFO) -> logging.Logger:
    """Set up a logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def parquet_path(name: str) -> Path:
    """Get path for a named parquet file in the data directory."""
    from src.config import PARQUET_DIR
    return PARQUET_DIR / f"{name}.parquet"
