"""
Simple logger for train pipeline
"""

import logging
from pathlib import Path

def setup_train_logger(log_file="logs/Sentiment_Analysis_Train.log", 
                       log_level="INFO", 
                       console_output=True):
    """
    Setup logger for train pipeline
    
    Parameters:
    -----------
    log_file : str
        Path to log file
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR)
    console_output : bool
        Whether to output logs to console
    
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    # Create logs directory
    log_dir = Path(log_file).parent
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Create logger
    logger = logging.getLogger("train_logger")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Format
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


# Global logger instance
train_logger = None


def get_logger():
    """Get the global logger instance"""
    global train_logger
    if train_logger is None:
        train_logger = setup_train_logger()
    return train_logger