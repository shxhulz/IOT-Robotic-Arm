import logging
import os
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import torch

from config import (
    LABELS_DIR,
    MATERIAL_CLASSES,
    MIN_MASK_AREA,
    POSTMORTEM_ANNOTATED_DIR,
    POSTMORTEM_FEATURES_DIR,
    SAM2_BATCH_SIZE,
    SAM2_CHECKPOINT,
    SAM2_MODEL_CFG,
    SAM2_POINTS_PER_SIDE,
    SAM2_PRED_IOU_THRESH,
    SAM2_STABILITY_SCORE_THRESH,
)
from material_classifier import MATERIAL_CLASSIFIER_AVAILABLE, MaterialClassifier

logger = logging.getLogger(__name__)

try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    logger.warning("SAM2 library not found. Install it to use segmentation features.")


class SAMProcessor:
    """Generate YOLO labels using SAM2 regions and zero-shot material classification."""

    def __init__(self, batch_size: int = SAM2_BATCH_SIZE):
        if not SAM2_AVAILABLE:
            raise RuntimeError("SAM2 is not installed.")
        if not MATERIAL_CLASSIFIER_AVAILABLE:
            raise RuntimeError("DINOv2/CLIP dependencies are not installed.")

        if not os.path.exists(SAM2_CHECKPOINT):
            self._download_checkpoint(SAM2_CHECKPOINT)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.min_mask_area = MIN_MASK_AREA
        self.material_classes = MATERIAL_CLASSES
        self.classifier = MaterialClassifier(material_classes=self.material_classes)
        self._class_colors = self._build_class_colors()
        self.last_run_stats = self._create_empty_run_stats()

        logger.info(
            "Loading SAM2 on %s with points_per_side=%d pred_iou_thresh=%.2f stability_thresh=%.2f",
            self.device,
            SAM2_POINTS_PER_SIDE,
            SAM2_PRED_IOU_THRESH,
            SAM2_STABILITY_SCORE_THRESH,
        )
        self.sam2_model = build_sam2(
            SAM2_MODEL_CFG,
            SAM2_CHECKPOINT,
            device=self.device,
            apply_postprocessing=True,
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(
            self.sam2_model,
            points_per_side=SAM2_POINTS_PER_SIDE,
            pred_iou_thresh=SAM2_PRED_IOU_THRESH,
            stability_score_thresh=SAM2_STABILITY_SCORE_THRESH,
        )

    def _build_class_colors(self) -> dict[str, tuple[int, int, int]]:
        palette = [
            (60, 180, 75),
            (0, 130, 200),
            (230, 25, 75),
            (245, 130, 48),
            (145, 30, 180),
            (70, 240, 240),
        ]
        return {
            class_name: palette[idx % len(palette)]
            for idx, class_name in enumerate(self.material_classes)
        }

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
            "min_mask_area": self.min_mask_area,
            "classification_batch_size": self.classifier.batch_size,
            "clip_min_confidence": self.classifier.min_confidence,
        }

    def _download_checkpoint(self, checkpoint_path: str) -> None:
        import urllib.request

        url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        logger.info("Downloading SAM2 checkpoint from %s to %s", url, checkpoint_path)
        try:
            urllib.request.urlretrieve(url, checkpoint_path)
        except Exception as exc:
            raise RuntimeError(f"Could not download SAM2 checkpoint from {url}") from exc

    def process_images(self, image_paths: list[str]) -> int:
        success_count = 0
        total_images = len(image_paths)

        self.last_run_stats = self._create_empty_run_stats()
        self.last_run_stats["started_at"] = datetime.utcnow().isoformat() + "Z"
        self.last_run_stats["total_images"] = total_images

        with torch.no_grad():
            for start_idx in range(0, total_images, self.batch_size):
                batch_paths = image_paths[start_idx : start_idx + self.batch_size]
                for offset, image_path in enumerate(batch_paths, start=1):
                    global_idx = start_idx + offset
                    try:
                        summary = self._process_single_image(image_path)
                        self._update_run_stats(summary)
                        success_count += 1
                    except Exception:
                        self.last_run_stats["failed_images"].append(os.path.basename(image_path))
                        logger.exception(
                            "Failed to process image [%d/%d]: %s",
                            global_idx,
                            total_images,
                            image_path,
                        )

                    if global_idx % max(1, self.batch_size) == 0 or global_idx == total_images:
                        logger.info("SAM2 progress: %d / %d images processed.", global_idx, total_images)

        self.last_run_stats["finished_at"] = datetime.utcnow().isoformat() + "Z"
        logger.info(
            "Labeling finished: processed=%d/%d images labels_written=%d rejected_low_conf=%d",
            success_count,
            total_images,
            self.last_run_stats["labels_written"],
            self.last_run_stats["labels_rejected_low_conf"],
        )
        logger.info("Class totals: %s", self.last_run_stats["class_counts"])
        return success_count

    def _update_run_stats(self, summary: dict) -> None:
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

    def _process_single_image(self, image_path: str) -> dict:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            logger.warning("Could not read image: %s", image_path)
            return self._build_image_summary(image_path, 0, 0, 0, [])

        height, width = image_bgr.shape[:2]
        if height == 0 or width == 0:
            logger.warning("Image has zero dimension: %s", image_path)
            return self._build_image_summary(image_path, 0, 0, 0, [])

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image_rgb)
        candidate_regions = self._extract_candidate_regions(image_rgb, masks)
        if not candidate_regions:
            logger.info("No usable SAM2 masks for %s", image_path)
            return self._build_image_summary(image_path, len(masks), 0, 0, [])

        crops = [candidate["crop"] for candidate in candidate_regions]
        predictions = self.classifier.classify_crops(crops)

        detections = []
        rejected_low_conf = 0
        embeddings = []

        label_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        label_path = os.path.join(LABELS_DIR, label_filename)
        with open(label_path, "w", encoding="utf-8") as handle:
            for candidate, prediction in zip(candidate_regions, predictions):
                if prediction is None:
                    rejected_low_conf += 1
                    continue

                bbox = candidate["bbox"]
                x_center = (bbox[0] + bbox[2] / 2.0) / width
                y_center = (bbox[1] + bbox[3] / 2.0) / height
                norm_w = bbox[2] / width
                norm_h = bbox[3] / height

                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                norm_w = max(0.0, min(1.0, norm_w))
                norm_h = max(0.0, min(1.0, norm_h))

                handle.write(
                    f"{prediction.class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"
                )

                detections.append(
                    {
                        "class_id": prediction.class_id,
                        "class_name": prediction.class_name,
                        "confidence": round(prediction.confidence, 4),
                        "bbox": [float(v) for v in bbox],
                        "mask_area": candidate["mask_area"],
                        "predicted_iou": candidate["predicted_iou"],
                        "stability_score": candidate["stability_score"],
                        "clip_scores": {
                            key: round(value, 4)
                            for key, value in prediction.clip_scores.items()
                        },
                    }
                )
                embeddings.append(prediction.dino_embedding)

        if not detections:
            os.remove(label_path)
            logger.info(
                "No confident material labels for %s. masks=%d rejected_low_conf=%d",
                os.path.basename(image_path),
                len(masks),
                rejected_low_conf,
            )
            return self._build_image_summary(
                image_path,
                len(masks),
                0,
                rejected_low_conf,
                [],
            )

        annotated_path = self._write_annotated_image(image_bgr, image_path, detections)
        feature_path = self._write_feature_archive(image_path, detections, embeddings)
        class_counts = self._class_counts_for_detections(detections)
        logger.info(
            "Image %s summary: masks=%d candidates=%d accepted=%d rejected_low_conf=%d class_counts=%s",
            os.path.basename(image_path),
            len(masks),
            len(candidate_regions),
            len(detections),
            rejected_low_conf,
            class_counts,
        )
        return self._build_image_summary(
            image_path,
            len(masks),
            len(detections),
            rejected_low_conf,
            detections,
            annotated_path=annotated_path,
            feature_path=feature_path,
            label_path=label_path,
        )

    def _extract_candidate_regions(self, image_rgb: np.ndarray, masks: list[dict]) -> list[dict]:
        candidates = []
        for mask in masks:
            bbox = [int(v) for v in mask["bbox"]]
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue

            mask_area = int(mask.get("area", 0))
            if mask_area < self.min_mask_area:
                continue

            x1 = max(0, bbox[0])
            y1 = max(0, bbox[1])
            x2 = min(image_rgb.shape[1], bbox[0] + bbox[2])
            y2 = min(image_rgb.shape[0], bbox[1] + bbox[3])
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            candidates.append(
                {
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "crop": crop,
                    "mask_area": mask_area,
                    "predicted_iou": float(mask.get("predicted_iou", 0.0)),
                    "stability_score": float(mask.get("stability_score", 0.0)),
                }
            )
        return candidates

    def _build_image_summary(
        self,
        image_path: str,
        mask_count: int,
        accepted_labels: int,
        rejected_low_conf: int,
        detections: list,
        annotated_path: Optional[str] = None,
        feature_path: Optional[str] = None,
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
            "feature_path": feature_path,
        }

    def _class_counts_for_detections(self, detections: list) -> dict:
        counts = self._empty_class_counts()
        for det in detections:
            counts[det["class_name"]] += 1
        return counts

    def _write_feature_archive(
        self,
        image_path: str,
        detections: list[dict],
        embeddings: list[np.ndarray],
    ) -> str:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        feature_path = os.path.join(POSTMORTEM_FEATURES_DIR, f"{base_name}_dinov2.npz")
        np.savez_compressed(
            feature_path,
            image_name=np.array([os.path.basename(image_path)]),
            labels=np.array([det["class_name"] for det in detections]),
            class_ids=np.array([det["class_id"] for det in detections], dtype=np.int64),
            confidences=np.array([det["confidence"] for det in detections], dtype=np.float32),
            bboxes=np.array([det["bbox"] for det in detections], dtype=np.float32),
            embeddings=np.stack(embeddings).astype(np.float32),
        )
        return feature_path

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

    def get_run_stats(self) -> dict:
        return self.last_run_stats
