import os
import logging
import mimetypes

from config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_BUCKET_NAME,
    MINIO_EXCLUDE_PREFIXES,
    MINIO_SECURE,
    IMAGES_DIR,
)

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ANNOTATED_FILE_MARKER = "_annotated"


class MinioHandler:
    """Download images from a MinIO bucket into the local staging directory."""

    def __init__(self):
        logger.info("Connecting to MinIO at %s (secure=%s)", MINIO_ENDPOINT, MINIO_SECURE)
        self.client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )

    def download_images(self, limit: int = 100) -> list[str]:
        """Download up to *limit* images from the configured bucket.

        Returns:
            List of local file paths for successfully downloaded images.
        """
        if not self.client.bucket_exists(MINIO_BUCKET_NAME):
            logger.warning("Bucket '%s' does not exist.", MINIO_BUCKET_NAME)
            return []

        objects = self.client.list_objects(MINIO_BUCKET_NAME, recursive=True)
        downloaded_files: list[str] = []
        skipped = 0
        skipped_non_images = 0
        discovered = 0

        for obj in objects:
            discovered += 1
            if len(downloaded_files) >= limit:
                break

            ext = os.path.splitext(obj.object_name)[1].lower()
            if ext not in VALID_IMAGE_EXTENSIONS:
                skipped_non_images += 1
                continue

            normalized_name = obj.object_name.replace("\\", "/").lstrip("/")
            if any(
                normalized_name.startswith(f"{prefix}/") or normalized_name == prefix
                for prefix in MINIO_EXCLUDE_PREFIXES
            ):
                skipped_non_images += 1
                continue

            if ANNOTATED_FILE_MARKER in os.path.basename(normalized_name):
                skipped_non_images += 1
                continue

            local_name = os.path.basename(obj.object_name)
            file_path = os.path.join(IMAGES_DIR, local_name)

            # Skip if already downloaded (idempotent re-runs)
            if os.path.exists(file_path):
                downloaded_files.append(file_path)
                skipped += 1
                continue

            try:
                self.client.fget_object(MINIO_BUCKET_NAME, obj.object_name, file_path)
                downloaded_files.append(file_path)
            except S3Error:
                logger.exception(
                    "Failed to download %s from bucket '%s'.",
                    obj.object_name,
                    MINIO_BUCKET_NAME,
                )

        logger.info(
            "MinIO scan summary: discovered=%d, skipped_non_images=%d, selected=%d (limit=%d).",
            discovered,
            skipped_non_images,
            len(downloaded_files),
            limit,
        )
        logger.info(
            "Downloaded %d images (%d already existed) from bucket '%s'.",
            len(downloaded_files) - skipped,
            skipped,
            MINIO_BUCKET_NAME,
        )
        return downloaded_files

    def upload_file(self, local_path: str, object_name: str, bucket_name: str = MINIO_BUCKET_NAME) -> bool:
        """Upload one local file to MinIO."""
        if not os.path.isfile(local_path):
            logger.warning("Upload skipped: local file missing: %s", local_path)
            return False

        content_type, _ = mimetypes.guess_type(local_path)
        content_type = content_type or "application/octet-stream"

        try:
            self.client.fput_object(
                bucket_name,
                object_name,
                local_path,
                content_type=content_type,
            )
            logger.info("Uploaded to MinIO: %s -> %s/%s", local_path, bucket_name, object_name)
            return True
        except S3Error:
            logger.exception("Failed to upload %s to %s/%s", local_path, bucket_name, object_name)
            return False

    def upload_directory(self, local_dir: str, object_prefix: str, bucket_name: str = MINIO_BUCKET_NAME) -> dict:
        """Upload all files in a directory tree into a MinIO object prefix."""
        if not os.path.isdir(local_dir):
            logger.warning("Upload directory skipped: missing directory: %s", local_dir)
            return {"uploaded": 0, "failed": 0}

        uploaded = 0
        failed = 0
        for root, _, files in os.walk(local_dir):
            for file_name in files:
                local_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(local_path, local_dir).replace("\\", "/")
                object_name = f"{object_prefix.rstrip('/')}/{rel_path}"
                if self.upload_file(local_path, object_name, bucket_name=bucket_name):
                    uploaded += 1
                else:
                    failed += 1

        logger.info(
            "Directory upload summary: source=%s, prefix=%s, uploaded=%d, failed=%d",
            local_dir,
            object_prefix,
            uploaded,
            failed,
        )
        return {"uploaded": uploaded, "failed": failed}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    handler = MinioHandler()
    handler.download_images()
