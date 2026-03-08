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


def _collect_split_class_counts(labels_dir: str) -> dict:
    counts = {"paper": 0, "plastic": 0, "metal": 0, "unknown": 0}
    class_map = {"0": "paper", "1": "plastic", "2": "metal"}

    label_files = [
        f for f in os.listdir(labels_dir)
        if f.lower().endswith(".txt")
    ]
    for label_file in label_files:
        path = os.path.join(labels_dir, label_file)
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                parts = raw_line.strip().split()
                if not parts:
                    continue
                class_id = parts[0]
                class_name = class_map.get(class_id, "unknown")
                counts[class_name] += 1
    return counts


def main():
    logger.info("Split stage started.")
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
    logger.info(
        "Split plan: total=%d train=%d val=%d ratio=%.2f",
        len(image_files),
        len(train_files),
        len(val_files),
        SPLIT_RATIO,
    )

    for split_name, files in [("train", train_files), ("val", val_files)]:
        img_dst = os.path.join(TRAINING_DATA_DIR, split_name, "images")
        lbl_dst = os.path.join(TRAINING_DATA_DIR, split_name, "labels")
        copied_labels = 0
        missing_labels = 0

        for img_name in files:
            src_img = os.path.join(IMAGES_DIR, img_name)
            shutil.copy2(src_img, os.path.join(img_dst, img_name))

            label_name = os.path.splitext(img_name)[0] + ".txt"
            src_lbl = os.path.join(LABELS_DIR, label_name)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(lbl_dst, label_name))
                copied_labels += 1
            else:
                logger.warning("No label file for %s - skipping label copy.", img_name)
                missing_labels += 1

        class_counts = _collect_split_class_counts(lbl_dst)
        logger.info(
            "Split %s summary: images=%d labels_copied=%d missing_labels=%d class_counts=%s",
            split_name,
            len(files),
            copied_labels,
            missing_labels,
            class_counts,
        )

    logger.info("Dataset split complete: %d train, %d val.", len(train_files), len(val_files))


if __name__ == "__main__":
    main()
