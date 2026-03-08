#!/usr/bin/env python3
"""Process images with SAM2 to generate YOLO-format labels."""

import os
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import IMAGES_DIR
from sam_processor import SAMProcessor, SAM2_AVAILABLE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def main():
    if not SAM2_AVAILABLE:
        logger.error("SAM2 is not installed in this environment.")
        sys.exit(1)

    image_paths = [
        os.path.join(IMAGES_DIR, f)
        for f in os.listdir(IMAGES_DIR)
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    ]

    if not image_paths:
        logger.error("No images found in %s after download step.", IMAGES_DIR)
        sys.exit(1)

    logger.info("Processing %d images with SAM2...", len(image_paths))

    processor = SAMProcessor()
    processed = processor.process_images(image_paths)

    logger.info("SAM2 processing complete: %d / %d images.", processed, len(image_paths))


if __name__ == "__main__":
    main()
