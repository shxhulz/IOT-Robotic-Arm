import os
import logging

import cv2
import numpy as np
import torch

from config import (
    LABELS_DIR,
    SAM2_CHECKPOINT,
    SAM2_MODEL_CFG,
    SAM2_BATCH_SIZE,
)

logger = logging.getLogger(__name__)

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    logger.warning(
        "SAM2 library not found. Install it to use segmentation features."
    )


class SAMProcessor:
    """Generate YOLO-format labels from images using SAM2 auto-masking."""

    def __init__(self, batch_size=SAM2_BATCH_SIZE):
        if not SAM2_AVAILABLE:
            raise RuntimeError("SAM2 is not installed.")

        # Ensure the checkpoint exists; download if it doesn't.
        if not os.path.exists(SAM2_CHECKPOINT):
            self._download_checkpoint(SAM2_CHECKPOINT)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        logger.info("SAM2 using device: %s, batch_size: %d", self.device, self.batch_size)

        self.sam2_model = build_sam2(
            SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=self.device
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model)

    def _download_checkpoint(self, checkpoint_path: str):
        """Downloads the SAM2 checkpoint from the official Meta repository."""
        import urllib.request
        # Default to hiera_large download link since it's the default in config.py
        # You could map SAM2_CHECKPOINT to specific URLs if multiple models are supported.
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
        logger.info("SAM2 checkpoint not found. Downloading from %s to %s", url, checkpoint_path)
        try:
            urllib.request.urlretrieve(url, checkpoint_path)
            logger.info("SAM2 checkpoint downloaded successfully.")
        except Exception as e:
            logger.error("Failed to download SAM2 checkpoint: %s", e)
            raise RuntimeError(f"Could not download SAM2 checkpoint from {url}") from e

    def process_images(self, image_paths: list[str]) -> int:
        """Process a list of images and write YOLO label files.
        Uses configurable batch_size and torch.inference_mode.
        """
        success_count = 0
        total_images = len(image_paths)

        # Process in batches
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16):
            for i in range(0, total_images, self.batch_size):
                batch_paths = image_paths[i : i + self.batch_size]

                for p_idx, image_path in enumerate(batch_paths):
                    global_idx = i + p_idx + 1
                    try:
                        self._process_single_image(image_path)
                        success_count += 1
                    except Exception:
                        logger.exception(
                            "Failed to process image [%d/%d]: %s",
                            global_idx,
                            total_images,
                            image_path,
                        )

                    if global_idx % max(1, self.batch_size) == 0:
                        logger.info(
                            "SAM2 progress: %d / %d images processed.",
                            global_idx,
                            total_images,
                        )

        logger.info(
            "SAM2 finished: %d / %d images processed successfully.",
            success_count,
            total_images,
        )
        return success_count

    def _process_single_image(self, image_path: str) -> None:
        """Generate masks for a single image and save as YOLO labels."""
        image = cv2.imread(image_path)
        if image is None:
            logger.warning("Could not read image (corrupt or missing): %s", image_path)
            return

        h, w = image.shape[:2]
        if h == 0 or w == 0:
            logger.warning("Image has zero dimension: %s", image_path)
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image_rgb)

        if not masks:
            logger.info("SAM2 produced no masks for %s", image_path)
            return

        label_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        label_path = os.path.join(LABELS_DIR, label_filename)

        with open(label_path, "w") as f:
            for mask in masks:
                bbox = mask["bbox"]  # [x, y, w_box, h_box] (absolute pixels)

                # Skip degenerate boxes
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue

                # Convert to YOLO normalized format:
                #   class  x_center  y_center  width  height
                class_id = 0
                x_center = (bbox[0] + bbox[2] / 2) / w
                y_center = (bbox[1] + bbox[3] / 2) / h
                norm_w = bbox[2] / w
                norm_h = bbox[3] / h

                # Clamp to [0, 1] for safety
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                norm_w = max(0.0, min(1.0, norm_w))
                norm_h = max(0.0, min(1.0, norm_h))

                f.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} "
                    f"{norm_w:.6f} {norm_h:.6f}\n"
                )

        logger.debug("Generated labels for %s", image_path)
