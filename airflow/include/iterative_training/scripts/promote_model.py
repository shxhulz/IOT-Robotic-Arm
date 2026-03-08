#!/usr/bin/env python3
"""Promote newly trained best.pt to the production model path."""

import os
import sys
import shutil
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import YOLO_MODEL_PATH, TRAINING_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    marker_path = os.path.join(TRAINING_DATA_DIR, ".best_weights_path")

    if not os.path.exists(marker_path):
        logger.error("Marker file not found at %s. Training step may have failed.", marker_path)
        sys.exit(1)

    with open(marker_path, "r") as f:
        best_weights = f.read().strip()

    if not best_weights or not os.path.exists(best_weights):
        logger.error("Best weights not found at '%s'.", best_weights)
        sys.exit(1)

    os.makedirs(os.path.dirname(YOLO_MODEL_PATH), exist_ok=True)
    shutil.copy2(best_weights, YOLO_MODEL_PATH)
    logger.info("Promoted model: %s → %s", best_weights, YOLO_MODEL_PATH)

    os.remove(marker_path)


if __name__ == "__main__":
    main()
