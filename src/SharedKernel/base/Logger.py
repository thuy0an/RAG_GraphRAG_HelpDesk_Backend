import logging
import sys
import traceback
from pathlib import Path

def setup_logging() -> None:
    """Configure basic logging for the application."""
    class ColoredFormatter(logging.Formatter):
        """Custom formatter with red color for ERROR level and blue for INFO level."""
        
        RED = '\033[91m'
        BLUE = '\033[94m'
        RESET = '\033[0m'
        
        def format(self, record):
            message = record.getMessage()
            
            if record.levelno == logging.ERROR:
                # Use traceback to get caller information
                tb = traceback.extract_stack()
                # Find the caller frame (skip logging frames)
                for frame in reversed(tb):
                    if 'logging' not in frame.filename and 'Logger.py' not in frame.filename:
                        file_path = frame.filename
                        line_no = frame.lineno
                        
                        # Try to extract feature folder and filename
                        if 'src' in file_path:
                            try:
                                parts = file_path.split('src\\')
                                if len(parts) > 1:
                                    relative_path = parts[1].replace('\\', '/')
                                    return f"{self.RED}[ERROR]: {relative_path}:{line_no} - {message}{self.RESET}"
                            except:
                                pass
                        
                        return f"{self.RED}[ERROR]: {file_path}:{line_no} - {message}{self.RESET}"
                
                return f"{self.RED}[ERROR]: {message}{self.RESET}"
            elif record.levelno == logging.INFO:
                return f"{self.BLUE}[INFO]: {message}{self.RESET}"
            else:
                return f"{message}"
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())
    
    # Remove existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance.
    """
    # Setup logging if not already configured
    if not logging.getLogger().handlers:
        setup_logging()
    
    return logging.getLogger(name)


# Setup logging when module is imported
setup_logging()


# Default logger for the module
logger = get_logger(__name__)
