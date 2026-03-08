#!/usr/bin/env python3
"""Split staged images + labels into train/val (80/20)."""

import os
import sys
import shutil
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import IMAGES_DIR, LABELS_DIR, TRAINING_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SPLIT_RATIO = 0.8


def main():
    image_files = sorted(
        f
        for f in os.listdir(IMAGES_DIR)
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    )

    if not image_files:
        logger.error("No image files to split in %s.", IMAGES_DIR)
        sys.exit(1)

    split_idx = int(len(image_files) * SPLIT_RATIO)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    for split_name, files in [("train", train_files), ("val", val_files)]:
        img_dst = os.path.join(TRAINING_DATA_DIR, split_name, "images")
        lbl_dst = os.path.join(TRAINING_DATA_DIR, split_name, "labels")

        for img_name in files:
            src_img = os.path.join(IMAGES_DIR, img_name)
            shutil.copy2(src_img, os.path.join(img_dst, img_name))

            label_name = os.path.splitext(img_name)[0] + ".txt"
            src_lbl = os.path.join(LABELS_DIR, label_name)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(lbl_dst, label_name))
            else:
                logger.warning("No label file for %s — skipping label copy.", img_name)

    logger.info("Dataset split complete: %d train, %d val.", len(train_files), len(val_files))


if __name__ == "__main__":
    main()
