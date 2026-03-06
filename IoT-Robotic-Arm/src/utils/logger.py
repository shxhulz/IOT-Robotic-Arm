import logging
import os
from datetime import datetime
from rich.logging import RichHandler

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(file_formatter)

rich_formatter = logging.Formatter("%(name)s - %(message)s")

stream_handler = RichHandler(rich_tracebacks=True, markup=True)
stream_handler.setFormatter(rich_formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Create a logger with the specified name.
    """
    child_logger = logging.getLogger(name)
    child_logger.setLevel(logging.INFO)
    return child_logger