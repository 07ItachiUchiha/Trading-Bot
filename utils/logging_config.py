import logging
import os
from datetime import datetime
import sys
from pathlib import Path

class LoggingConfig:
    """Centralized logging configuration for the trading bot"""
    
    @staticmethod
    def setup():
        """Set up logging with file and console handlers"""
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Create log filename with current date
        current_date = datetime.now().strftime('%Y%m%d')
        log_filename = logs_dir / f"trading_{current_date}.log"
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Clear any existing handlers to avoid duplicate logs
        if root_logger.handlers:
            for handler in root_logger.handlers:
                root_logger.removeHandler(handler)
                
        # Create file handler
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Return the configured logger and filename
        return root_logger, log_filename
    
    @staticmethod
    def get_logger(name):
        """Get a logger with the specified name"""
        return logging.getLogger(name)

# Initialize logging when this module is imported
logger, log_file = LoggingConfig.setup()
