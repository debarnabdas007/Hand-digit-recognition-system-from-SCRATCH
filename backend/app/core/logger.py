import logging
import sys

def get_logger(name: str):
    logger = logging.getLogger(name)
    
    # Prevent duplicate logs if the logger is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Output logs to the terminal
        handler = logging.StreamHandler(sys.stdout)
        
        # Format: [Time] - [File] - [Level] - [Message]
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger