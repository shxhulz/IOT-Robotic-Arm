"""Download images from MinIO into the staging directory."""

import os
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from minio_handler import MinioHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    handler = MinioHandler()
    images = handler.download_images(limit=100_000)
    count = len(images)
    logger.info("Downloaded %d images from MinIO.", count)

    if count == 0:
        logger.error(
            "No images were downloaded from MinIO. "
            "Check bucket existence and connectivity."
        )
        sys.exit(1)

    logger.info("Download step completed successfully: %d images.", count)


if __name__ == "__main__":
    main()
