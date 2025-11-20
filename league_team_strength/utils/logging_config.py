"""
Logging configuration for the entire system.
"""

import logging
import sys
from datetime import datetime

def setup_logging(log_file: str = None, level: int = logging.INFO):
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional log file path. If None, uses default filename.
        level: Logging level (default: INFO)
    """
    
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"strength_calculation_{timestamp}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Suppress noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('pyodbc').setLevel(logging.WARNING)
    
    logging.info(f"Logging initialized. Log file: {log_file}")