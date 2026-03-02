import logging
import sys
from typing import Optional

class CustomFormatter(logging.Formatter):
    GREEN = "\033[0;32m"  # Green
    BLUE = "\033[0;34m"   # Blue
    YELLOW = "\033[1;33m" # Yellow
    RED = "\033[1;31m"    # Red
    BOLD_RED = "\033[1;31m"  # Bold Red
    RESET = "\033[0m"     # Reset color

    FORMAT = "%(levelname)-1s: [%(name)-1s] - %(message)s - [%(asctime)s]"

    FORMATS = {
        logging.DEBUG: f"{BLUE}{FORMAT}{RESET}",
        logging.INFO: f"{GREEN}{FORMAT}{RESET}",
        logging.WARNING: f"{YELLOW}{FORMAT}{RESET}",
        logging.ERROR: f"{RED}{FORMAT}{RESET}",
        logging.CRITICAL: f"{BOLD_RED}{FORMAT}{RESET}"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

class Logger:
    _instances = {}
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        if name not in cls._instances:
            logger = logging.getLogger(name)
            logger.propagate = False
            logger.setLevel(logging.INFO)
            
            if logger.handlers:
                logger.handlers = []
                
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(CustomFormatter())
            logger.addHandler(handler)
            
            cls._instances[name] = logger
            
        return cls._instances[name]

logger = Logger().get_logger("")
def get_logger(name: str = None):
    return Logger.get_logger(name)