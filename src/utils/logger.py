"""
Logging utilities
"""
import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "automl",
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Setup logger with standard configuration"""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger



