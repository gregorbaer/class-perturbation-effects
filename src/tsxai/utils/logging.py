# logger_config.py
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    module_name: str,
    log_level: int = logging.INFO,
    enable_file_logging: bool = False,
    log_dir: Optional[Path] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.

    Args:
        module_name: Name of the module requesting the logger
        log_level: The logging level to use (default: logging.INFO)
        enable_file_logging: Whether to save logs to file (default: False)
        log_dir: Directory for log files if file logging is enabled (default: None)
            If None but file logging is enabled, uses ./logs/
        log_file: Name of the log file to use (default: None).
            If None, uses module_name.log

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)8s | %(module)15s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Get or create logger
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)

    # Remove any existing handlers
    logger.handlers.clear()

    # Create console handler with formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if enabled
    if enable_file_logging:
        log_dir = Path(log_dir) if log_dir else Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = (
            log_file if log_file else f"{module_name}_{datetime.now():%Y%m%d}.log"
        )
        log_path = str(log_dir / log_file)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
