#!/usr/bin/env python3
"""Prepare the workspace: clean staging dirs, create train/val split directories."""

import os
import sys
import shutil
import logging

# Add parent so we can import config as a sibling module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    IMAGES_DIR,
    LABELS_DIR,
    TRAINING_DATA_DIR,
    POSTMORTEM_ANNOTATED_DIR,
    POSTMORTEM_FEATURES_DIR,
    POSTMORTEM_REPORTS_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        logger.info("PyTorch version: %s", torch.__version__)
        logger.info("CUDA available: %s", cuda_available)
        if cuda_available:
            device_count = torch.cuda.device_count()
            logger.info("CUDA device count: %d", device_count)
            for i in range(device_count):
                logger.info("CUDA device %d: %s", i, torch.cuda.get_device_name(i))
            logger.info("CUDA version: %s", torch.version.cuda)
        else:
            logger.warning("No CUDA GPU detected — training will use CPU.")
    except Exception as e:
        logger.warning("Could not check CUDA availability: %s", e)

    for directory in [
        IMAGES_DIR,
        LABELS_DIR,
        POSTMORTEM_ANNOTATED_DIR,
        POSTMORTEM_FEATURES_DIR,
        POSTMORTEM_REPORTS_DIR,
    ]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            logger.info("Cleaned directory: %s", directory)
        os.makedirs(directory, exist_ok=True)
        logger.info("Created directory: %s", directory)

    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            split_dir = os.path.join(TRAINING_DATA_DIR, split, sub)
            if os.path.exists(split_dir):
                shutil.rmtree(split_dir)
            os.makedirs(split_dir, exist_ok=True)
            logger.info("Created split directory: %s", split_dir)

    logger.info("Workspace preparation complete.")


if __name__ == "__main__":
    main()
