import os
import logging

from config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_BUCKET_NAME,
    MINIO_SECURE,
    IMAGES_DIR,
)

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


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

        for obj in objects:
            if len(downloaded_files) >= limit:
                break

            ext = os.path.splitext(obj.object_name)[1].lower()
            if ext not in VALID_IMAGE_EXTENSIONS:
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
            "Downloaded %d images (%d already existed) from bucket '%s'.",
            len(downloaded_files) - skipped,
            skipped,
            MINIO_BUCKET_NAME,
        )
        return downloaded_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    handler = MinioHandler()
    handler.download_images()
