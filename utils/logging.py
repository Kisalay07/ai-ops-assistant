from __future__ import annotations

import logging
import os
from typing import Optional

from rich.logging import RichHandler

def setup_logging(level: Optional[str] = None) -> None:
    """Configure app-wide logging with nice console output."""
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
