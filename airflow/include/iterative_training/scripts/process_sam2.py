#!/usr/bin/env python3
"""Process images with SAM2 to generate YOLO-format labels."""

import os
import sys
import logging
import json
import csv
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    IMAGES_DIR,
    POSTMORTEM_DIR,
    POSTMORTEM_REPORTS_DIR,
    POSTMORTEM_ANNOTATED_DIR,
    MINIO_POSTMORTEM_PREFIX,
    MINIO_BUCKET_NAME,
)
from sam_processor import SAMProcessor, SAM2_AVAILABLE, CLIP_AVAILABLE
from minio_handler import MinioHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _write_reports(stats: dict) -> dict:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(POSTMORTEM_REPORTS_DIR, exist_ok=True)

    json_path = os.path.join(POSTMORTEM_REPORTS_DIR, f"sam2_postmortem_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    csv_path = os.path.join(POSTMORTEM_REPORTS_DIR, f"sam2_image_summary_{timestamp}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_name",
                "mask_count",
                "accepted_labels",
                "rejected_low_conf",
                "paper",
                "plastic",
                "metal",
                "label_path",
                "annotated_path",
            ]
        )
        for item in stats.get("image_summaries", []):
            class_counts = item.get("class_counts", {})
            writer.writerow(
                [
                    item.get("image_name"),
                    item.get("mask_count", 0),
                    item.get("accepted_labels", 0),
                    item.get("rejected_low_conf", 0),
                    class_counts.get("paper", 0),
                    class_counts.get("plastic", 0),
                    class_counts.get("metal", 0),
                    item.get("label_path"),
                    item.get("annotated_path"),
                ]
            )

    txt_path = os.path.join(POSTMORTEM_REPORTS_DIR, f"sam2_postmortem_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("SAM2 + CLIP Postmortem\n")
        f.write(f"started_at: {stats.get('started_at')}\n")
        f.write(f"finished_at: {stats.get('finished_at')}\n")
        f.write(f"total_images: {stats.get('total_images')}\n")
        f.write(f"processed_images: {stats.get('processed_images')}\n")
        f.write(f"images_with_masks: {stats.get('images_with_masks')}\n")
        f.write(f"images_with_labels: {stats.get('images_with_labels')}\n")
        f.write(f"images_without_labels: {stats.get('images_without_labels')}\n")
        f.write(f"total_masks: {stats.get('total_masks')}\n")
        f.write(f"labels_written: {stats.get('labels_written')}\n")
        f.write(f"labels_rejected_low_conf: {stats.get('labels_rejected_low_conf')}\n")
        f.write(f"class_counts: {stats.get('class_counts')}\n")
        f.write(f"failed_images: {stats.get('failed_images')}\n")

    return {
        "json": json_path,
        "csv": csv_path,
        "txt": txt_path,
        "timestamp": timestamp,
    }


def _upload_postmortem_to_minio(timestamp: str) -> None:
    object_prefix = f"{MINIO_POSTMORTEM_PREFIX.rstrip('/')}/{timestamp}"
    handler = MinioHandler()
    logger.info(
        "Uploading postmortem artifacts to MinIO bucket=%s prefix=%s",
        MINIO_BUCKET_NAME,
        object_prefix,
    )
    annotated_result = handler.upload_directory(
        POSTMORTEM_ANNOTATED_DIR,
        f"{object_prefix}/annotated_images",
        bucket_name=MINIO_BUCKET_NAME,
    )
    reports_result = handler.upload_directory(
        POSTMORTEM_REPORTS_DIR,
        f"{object_prefix}/reports",
        bucket_name=MINIO_BUCKET_NAME,
    )
    logger.info(
        "MinIO upload complete: annotated(uploaded=%d, failed=%d), reports(uploaded=%d, failed=%d)",
        annotated_result["uploaded"],
        annotated_result["failed"],
        reports_result["uploaded"],
        reports_result["failed"],
    )


def main():
    if not SAM2_AVAILABLE:
        logger.error("SAM2 is not installed in this environment.")
        sys.exit(1)
    if not CLIP_AVAILABLE:
        logger.error("open_clip_torch is not installed in this environment.")
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
    logger.info(
        "Postmortem output directories: root=%s, annotated=%s, reports=%s",
        POSTMORTEM_DIR,
        POSTMORTEM_ANNOTATED_DIR,
        POSTMORTEM_REPORTS_DIR,
    )

    processor = SAMProcessor()
    processed = processor.process_images(image_paths)
    stats = processor.get_run_stats()
    reports = _write_reports(stats)

    logger.info("SAM2 processing complete: %d / %d images.", processed, len(image_paths))
    logger.info("Class distribution: %s", stats.get("class_counts"))
    logger.info(
        "Postmortem reports generated: json=%s csv=%s txt=%s",
        reports["json"],
        reports["csv"],
        reports["txt"],
    )

    should_upload = os.getenv("UPLOAD_POSTMORTEM_TO_MINIO", "True").lower() == "true"
    if should_upload:
        _upload_postmortem_to_minio(reports["timestamp"])
    else:
        logger.info("UPLOAD_POSTMORTEM_TO_MINIO is false; skipping MinIO upload.")


if __name__ == "__main__":
    main()
