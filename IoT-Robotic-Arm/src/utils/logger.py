import logging
import os
import sys
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Create a logger with the specified name.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
