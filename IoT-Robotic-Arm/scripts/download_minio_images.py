import argparse
import os
import logging
from minio import Minio
from minio.error import S3Error

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_images(endpoint, access_key, secret_key, bucket_name, output_dir, secure, limit, recursive):
    """Download images from a MinIO bucket into a local directory."""
    try:
        # Initialize MinIO client
        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

        # Check if bucket exists
        if not client.bucket_exists(bucket_name):
            logger.error(f"Bucket '{bucket_name}' does not exist.")
            return

        # List objects in the bucket
        objects = client.list_objects(bucket_name, recursive=recursive)
        
        count = 0
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}

        logger.info(f"Starting download from bucket '{bucket_name}' to '{output_dir}'...")

        for obj in objects:
            if limit and count >= limit:
                logger.info(f"Reached limit of {limit} images. Stopping.")
                break

            # Skip directories
            if obj.is_dir:
                continue

            # Check if object is an image by extension
            ext = os.path.splitext(obj.object_name)[1].lower()
            if ext not in valid_extensions:
                logger.debug(f"Skipping non-image object: {obj.object_name}")
                continue

            # Define local file path
            # For objects in subdirectories, flatten or preserve structure?
            # Usually vision thread saves to root, but let's handle subdirectories if needed
            local_file_path = os.path.join(output_dir, os.path.basename(obj.object_name))
            
            # If preserving structure is needed (if there's subdirectories):
            # local_file_path = os.path.join(output_dir, obj.object_name.replace("/", os.sep))
            # os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            try:
                client.fget_object(bucket_name, obj.object_name, local_file_path)
                count += 1
                if count % 10 == 0:
                    logger.info(f"Downloaded {count} images...")
            except S3Error as e:
                logger.error(f"Error downloading {obj.object_name}: {e}")

        logger.info(f"Successfully downloaded {count} images to '{output_dir}'.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download images from a MinIO bucket saved by the vision thread.")
    
    parser.add_argument("--endpoint", default="localhost:9000", help="MinIO server endpoint (default: localhost:9000)")
    parser.add_argument("--access-key", default="minioadmin", help="MinIO access key (default: minioadmin)")
    parser.add_argument("--secret-key", default="minioadmin", help="MinIO secret key (default: minioadmin)")
    parser.add_argument("--bucket", default="camera-frames", help="MinIO bucket name (default: camera-frames)")
    parser.add_argument("--output", default="downloaded_images", help="Local folder to save images (default: downloaded_images)")
    parser.add_argument("--secure", action="store_true", help="Use secure connection (HTTPS)")
    parser.add_argument("--limit", type=int, help="Maximum number of images to download")
    parser.add_argument("--no-recursive", action="store_false", dest="recursive", default=True, help="Do not search subdirectories recursively")

    args = parser.parse_args()

    download_images(
        endpoint=args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        bucket_name=args.bucket,
        output_dir=args.output,
        secure=args.secure,
        limit=args.limit,
        recursive=args.recursive
    )

if __name__ == "__main__":
    main()
