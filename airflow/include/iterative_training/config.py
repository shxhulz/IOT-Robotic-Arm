import os

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "host.docker.internal:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "camera-frames")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == "true"

_PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))

TRAINING_DATA_DIR = os.getenv(
    "TRAINING_DATA_DIR",
    os.path.join(_PACKAGE_DIR, "data"),
)
IMAGES_DIR = os.path.join(TRAINING_DATA_DIR, "images")
LABELS_DIR = os.path.join(TRAINING_DATA_DIR, "labels")

TRAIN_IMAGES_DIR = os.path.join(TRAINING_DATA_DIR, "train", "images")
TRAIN_LABELS_DIR = os.path.join(TRAINING_DATA_DIR, "train", "labels")
VAL_IMAGES_DIR = os.path.join(TRAINING_DATA_DIR, "val", "images")
VAL_LABELS_DIR = os.path.join(TRAINING_DATA_DIR, "val", "labels")

YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    os.path.join(_PACKAGE_DIR, "weights", "best.pt"),
)

YOLO_BASE_MODEL = os.getenv("YOLO_BASE_MODEL", "yolov8n.pt")
YOLO_TRAIN_EPOCHS = int(os.getenv("YOLO_TRAIN_EPOCHS", "5"))
YOLO_IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE", "640"))
USE_CUDA = os.getenv("USE_CUDA", "False").lower() == "true"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "yolo-iterative-training")

SAM2_CHECKPOINT = os.getenv("SAM2_CHECKPOINT", "sam2_hiera_large.pt")
SAM2_MODEL_CFG = os.getenv("SAM2_MODEL_CFG", "sam2_hiera_l.yaml")
SAM2_BATCH_SIZE = int(os.getenv("SAM2_BATCH_SIZE", "8"))

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)
