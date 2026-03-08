import os
import logging
from typing import Optional
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image

from config import (
    LABELS_DIR,
    SAM2_CHECKPOINT,
    SAM2_MODEL_CFG,
    SAM2_BATCH_SIZE,
    MATERIAL_CLASSES,
    MATERIAL_CLASS_TO_ID,
    POSTMORTEM_ANNOTATED_DIR,
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

try:
    import open_clip

    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning(
        "open_clip_torch is not installed. Install it to classify SAM2 masks."
    )


class SAMProcessor:
    """Generate YOLO-format labels using SAM2 masks + CLIP material classification."""

    def __init__(self, batch_size=SAM2_BATCH_SIZE):
        if not SAM2_AVAILABLE:
            raise RuntimeError("SAM2 is not installed.")
        if not CLIP_AVAILABLE:
            raise RuntimeError("open_clip_torch is not installed.")

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
        self.material_classes = MATERIAL_CLASSES
        self.clip_min_conf = float(os.getenv("CLIP_MIN_CONFIDENCE", "0.40"))
        self._class_colors = {
            "paper": (60, 180, 75),
            "plastic": (0, 130, 200),
            "metal": (230, 25, 75),
        }
        self.last_run_stats = self._create_empty_run_stats()
        self._init_clip_model()

    def _init_clip_model(self):
        model_name = os.getenv("CLIP_MODEL_NAME", "ViT-B-32")
        pretrained = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")
        logger.info(
            "Loading CLIP model for material classification: %s (%s)",
            model_name,
            pretrained,
        )
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        self.clip_tokenizer = open_clip.get_tokenizer(model_name)

        text_prompts = [f"a photo of {name} waste" for name in self.material_classes]
        with torch.inference_mode():
            tokens = self.clip_tokenizer(text_prompts).to(self.device)
            text_features = self.clip_model.encode_text(tokens)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logger.info("CLIP classes: %s", ", ".join(self.material_classes))

    def _empty_class_counts(self) -> dict:
        return {name: 0 for name in self.material_classes}

    def _create_empty_run_stats(self) -> dict:
        return {
            "started_at": None,
            "finished_at": None,
            "total_images": 0,
            "processed_images": 0,
            "failed_images": [],
            "images_with_masks": 0,
            "images_with_labels": 0,
            "images_without_labels": 0,
            "total_masks": 0,
            "labels_written": 0,
            "labels_rejected_low_conf": 0,
            "class_counts": self._empty_class_counts(),
            "image_summaries": [],
            "clip_min_confidence": self.clip_min_conf,
        }

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

        self.last_run_stats = self._create_empty_run_stats()
        self.last_run_stats["started_at"] = datetime.utcnow().isoformat() + "Z"
        self.last_run_stats["total_images"] = total_images

        # Process in batches
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16):
            for i in range(0, total_images, self.batch_size):
                batch_paths = image_paths[i : i + self.batch_size]

                for p_idx, image_path in enumerate(batch_paths):
                    global_idx = i + p_idx + 1
                    try:
                        summary = self._process_single_image(image_path)
                        self.last_run_stats["processed_images"] += 1
                        if summary["mask_count"] > 0:
                            self.last_run_stats["images_with_masks"] += 1
                        if summary["accepted_labels"] > 0:
                            self.last_run_stats["images_with_labels"] += 1
                        else:
                            self.last_run_stats["images_without_labels"] += 1
                        self.last_run_stats["total_masks"] += summary["mask_count"]
                        self.last_run_stats["labels_written"] += summary["accepted_labels"]
                        self.last_run_stats["labels_rejected_low_conf"] += summary["rejected_low_conf"]
                        for class_name, count in summary["class_counts"].items():
                            self.last_run_stats["class_counts"][class_name] += count
                        self.last_run_stats["image_summaries"].append(summary)
                        success_count += 1
                    except Exception:
                        self.last_run_stats["failed_images"].append(os.path.basename(image_path))
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
        self.last_run_stats["finished_at"] = datetime.utcnow().isoformat() + "Z"
        logger.info(
            "SAM2 postmortem summary: images_with_masks=%d, images_with_labels=%d, "
            "images_without_labels=%d, labels_written=%d, rejected_low_conf=%d",
            self.last_run_stats["images_with_masks"],
            self.last_run_stats["images_with_labels"],
            self.last_run_stats["images_without_labels"],
            self.last_run_stats["labels_written"],
            self.last_run_stats["labels_rejected_low_conf"],
        )
        logger.info("Class totals: %s", self.last_run_stats["class_counts"])
        return success_count

    def _process_single_image(self, image_path: str) -> dict:
        """Generate masks for a single image and save labels + annotated preview."""
        image = cv2.imread(image_path)
        if image is None:
            logger.warning("Could not read image (corrupt or missing): %s", image_path)
            return self._build_image_summary(image_path, 0, 0, 0, [])

        h, w = image.shape[:2]
        if h == 0 or w == 0:
            logger.warning("Image has zero dimension: %s", image_path)
            return self._build_image_summary(image_path, 0, 0, 0, [])

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image_rgb)
        mask_count = len(masks)

        if not masks:
            logger.info("SAM2 produced no masks for %s", image_path)
            return self._build_image_summary(image_path, 0, 0, 0, [])

        label_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        label_path = os.path.join(LABELS_DIR, label_filename)

        detections = []
        written = 0
        rejected_low_conf = 0
        with open(label_path, "w") as f:
            for mask in masks:
                bbox = mask["bbox"]  # [x, y, w_box, h_box] (absolute pixels)

                # Skip degenerate boxes
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue

                # Convert to YOLO normalized format:
                #   class  x_center  y_center  width  height
                cls_result = self._classify_material(image_rgb, bbox)
                if cls_result is None:
                    rejected_low_conf += 1
                    continue
                class_id, class_name, class_conf = cls_result
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
                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(float(class_conf), 4),
                    "bbox": [float(v) for v in bbox],
                })
                written += 1

        if written == 0:
            os.remove(label_path)
            logger.info("No confident material labels for %s", image_path)
            logger.info(
                "Image %s summary: masks=%d accepted=0 rejected_low_conf=%d",
                os.path.basename(image_path),
                mask_count,
                rejected_low_conf,
            )
            return self._build_image_summary(
                image_path,
                mask_count,
                0,
                rejected_low_conf,
                detections,
            )

        annotated_path = self._write_annotated_image(image, image_path, detections)
        class_counts = self._class_counts_for_detections(detections)
        logger.info(
            "Image %s summary: masks=%d accepted=%d rejected_low_conf=%d class_counts=%s annotated=%s",
            os.path.basename(image_path),
            mask_count,
            written,
            rejected_low_conf,
            class_counts,
            annotated_path,
        )
        return self._build_image_summary(
            image_path,
            mask_count,
            written,
            rejected_low_conf,
            detections,
            annotated_path=annotated_path,
            label_path=label_path,
        )

    def _build_image_summary(
        self,
        image_path: str,
        mask_count: int,
        accepted_labels: int,
        rejected_low_conf: int,
        detections: list,
        annotated_path: Optional[str] = None,
        label_path: Optional[str] = None,
    ) -> dict:
        return {
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
            "mask_count": mask_count,
            "accepted_labels": accepted_labels,
            "rejected_low_conf": rejected_low_conf,
            "class_counts": self._class_counts_for_detections(detections),
            "detections": detections,
            "label_path": label_path,
            "annotated_path": annotated_path,
        }

    def _class_counts_for_detections(self, detections: list) -> dict:
        counts = self._empty_class_counts()
        for det in detections:
            counts[det["class_name"]] += 1
        return counts

    def _write_annotated_image(self, image_bgr: np.ndarray, image_path: str, detections: list) -> str:
        rendered = image_bgr.copy()
        for det in detections:
            x, y, w_box, h_box = det["bbox"]
            x1 = int(max(0, x))
            y1 = int(max(0, y))
            x2 = int(min(rendered.shape[1] - 1, x + w_box))
            y2 = int(min(rendered.shape[0] - 1, y + h_box))
            class_name = det["class_name"]
            conf = det["confidence"]
            color = self._class_colors.get(class_name, (255, 255, 255))
            label = f"{class_name}:{conf:.2f}"

            cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                rendered,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        annotated_path = os.path.join(POSTMORTEM_ANNOTATED_DIR, f"{base_name}_annotated.jpg")
        cv2.imwrite(annotated_path, rendered)
        return annotated_path

    def _classify_material(self, image_rgb: np.ndarray, bbox: list[float]) -> Optional[tuple[int, str, float]]:
        """Classify a SAM2 mask crop into paper/plastic/metal using CLIP."""
        x, y, w_box, h_box = bbox
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(image_rgb.shape[1], int(x + w_box))
        y2 = min(image_rgb.shape[0], int(y + h_box))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        pil_crop = Image.fromarray(crop)
        image_tensor = self.clip_preprocess(pil_crop).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            image_features = self.clip_model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            best_prob, best_idx = torch.max(probs[0], dim=0)

        if float(best_prob.item()) < self.clip_min_conf:
            return None

        class_name = self.material_classes[int(best_idx.item())]
        return MATERIAL_CLASS_TO_ID[class_name], class_name, float(best_prob.item())

    def get_run_stats(self) -> dict:
        return self.last_run_stats
