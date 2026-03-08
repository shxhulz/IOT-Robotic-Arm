"""
Standalone entry-point for running the iterative training pipeline
outside of Airflow (e.g. for local testing).
"""

import logging
import os
import shutil

from config import (
    IMAGES_DIR,
    LABELS_DIR,
    TRAINING_DATA_DIR,
    YOLO_MODEL_PATH,
)
from minio_handler import MinioHandler
from sam_processor import SAMProcessor, SAM2_AVAILABLE
from yolo_trainer import YOLOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _prepare_split_dirs():
    """Create train/val directories."""
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(TRAINING_DATA_DIR, split, sub), exist_ok=True)


def _split_dataset(ratio: float = 0.8):
    """Split staged images + labels into train/val."""
    valid_ext = {".png", ".jpg", ".jpeg"}
    files = sorted(
        f for f in os.listdir(IMAGES_DIR)
        if os.path.splitext(f)[1].lower() in valid_ext
    )

    split_idx = int(len(files) * ratio)
    splits = {"train": files[:split_idx], "val": files[split_idx:]}

    for name, file_list in splits.items():
        for img in file_list:
            shutil.copy2(
                os.path.join(IMAGES_DIR, img),
                os.path.join(TRAINING_DATA_DIR, name, "images", img),
            )
            lbl = os.path.splitext(img)[0] + ".txt"
            lbl_src = os.path.join(LABELS_DIR, lbl)
            if os.path.exists(lbl_src):
                shutil.copy2(
                    lbl_src,
                    os.path.join(TRAINING_DATA_DIR, name, "labels", lbl),
                )

    logger.info(
        "Split: %d train, %d val.", len(splits["train"]), len(splits["val"])
    )


def run_iterative_training():
    """Run the full iterative training pipeline."""
    logger.info("Starting iterative training loop...")

    # 1. Download images from MinIO
    minio_handler = MinioHandler()
    downloaded_images = minio_handler.download_images(limit=100_000)

    if not downloaded_images:
        logger.warning("No images downloaded. Skipping training.")
        return

    # 2. Process images with SAM2
    if not SAM2_AVAILABLE:
        logger.error("SAM2 is not available. Cannot generate labels.")
        return

    processor = SAMProcessor()
    processor.process_images(downloaded_images)
    logger.info("Processed %d images with SAM2.", len(downloaded_images))

    # 3. Split into train/val
    _prepare_split_dirs()
    _split_dataset()

    # 4. Train YOLO model
    trainer = YOLOTrainer()
    trainer.train()
    logger.info("Training complete.")

    # 5. Promote new weights
    new_weights = trainer.get_latest_weights()
    if new_weights:
        os.makedirs(os.path.dirname(YOLO_MODEL_PATH), exist_ok=True)
        shutil.copy2(new_weights, YOLO_MODEL_PATH)
        logger.info("Updated model at %s with %s", YOLO_MODEL_PATH, new_weights)
    else:
        logger.warning("Could not find newly trained weights.")

    logger.info("Iterative training loop finished successfully.")


if __name__ == "__main__":
    run_iterative_training()
