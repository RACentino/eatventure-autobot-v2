"""Structured logging module."""

import logging
import sys
from pathlib import Path
from core import config

def setup_logger(name: str) -> logging.Logger:
    """Configures and returns a strictly formatted logger."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        log_format = '%(asctime)s - [%(levelname)s] - %(name)s:%(lineno)d - %(message)s'
        formatter = logging.Formatter(log_format)

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # File Handler
        log_file = Path(config.LOGS_DIR) / 'eatventure_bot.log'
        file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
    return logger

log = setup_logger("core.logger")
