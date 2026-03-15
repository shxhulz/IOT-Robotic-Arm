import os


def _parse_material_classes() -> list[str]:
    raw = os.getenv("MATERIAL_CLASSES", "paper,plastic,metal")
    classes = [item.strip() for item in raw.split(",") if item.strip()]
    return classes or ["paper", "plastic", "metal"]


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
POSTMORTEM_DIR = os.path.join(TRAINING_DATA_DIR, "postmortem")
POSTMORTEM_ANNOTATED_DIR = os.path.join(POSTMORTEM_DIR, "annotated_images")
POSTMORTEM_REPORTS_DIR = os.path.join(POSTMORTEM_DIR, "reports")
POSTMORTEM_FEATURES_DIR = os.path.join(POSTMORTEM_DIR, "features")
MINIO_POSTMORTEM_PREFIX = os.getenv("MINIO_POSTMORTEM_PREFIX", "postmortem")
MINIO_EXCLUDE_PREFIXES = tuple(
    prefix.strip().strip("/")
    for prefix in os.getenv("MINIO_EXCLUDE_PREFIXES", MINIO_POSTMORTEM_PREFIX).split(",")
    if prefix.strip()
)

MATERIAL_CLASSES = _parse_material_classes()
MATERIAL_CLASS_TO_ID = {name: idx for idx, name in enumerate(MATERIAL_CLASSES)}
MATERIAL_PROMPT_TEMPLATE = os.getenv(
    "MATERIAL_PROMPT_TEMPLATE",
    "a close-up industrial photo of {material} material",
)

TRAIN_IMAGES_DIR = os.path.join(TRAINING_DATA_DIR, "train", "images")
TRAIN_LABELS_DIR = os.path.join(TRAINING_DATA_DIR, "train", "labels")
VAL_IMAGES_DIR = os.path.join(TRAINING_DATA_DIR, "val", "images")
VAL_LABELS_DIR = os.path.join(TRAINING_DATA_DIR, "val", "labels")

YOLO_MODEL_PATH = os.getenv(
    "YOLO_MODEL_PATH",
    os.path.join(_PACKAGE_DIR, "weights", "best.pt"),
)

YOLO_BASE_MODEL = os.getenv("YOLO_BASE_MODEL", "yolov8n.pt")
YOLO_TRAIN_EPOCHS = int(os.getenv("YOLO_TRAIN_EPOCHS", "1000"))
YOLO_TRAIN_PATIENCE = int(os.getenv("YOLO_TRAIN_PATIENCE", "50"))
YOLO_IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE", "640"))
USE_CUDA = os.getenv("USE_CUDA", "False").lower() == "true"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "yolo-iterative-training")

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/models")
SAM2_CHECKPOINT = os.getenv(
    "SAM2_CHECKPOINT",
    os.path.join(MODEL_CACHE_DIR, "sam2", "sam2_hiera_large.pt"),
)
SAM2_MODEL_CFG = os.getenv("SAM2_MODEL_CFG", "sam2_hiera_l.yaml")
SAM2_BATCH_SIZE = int(os.getenv("SAM2_BATCH_SIZE", "8"))
SAM2_POINTS_PER_SIDE = int(os.getenv("SAM2_POINTS_PER_SIDE", "32"))
SAM2_PRED_IOU_THRESH = float(os.getenv("SAM2_PRED_IOU_THRESH", "0.86"))
SAM2_STABILITY_SCORE_THRESH = float(os.getenv("SAM2_STABILITY_SCORE_THRESH", "0.92"))

MIN_MASK_AREA = int(os.getenv("MIN_MASK_AREA", "1200"))
DINO_MODEL_ID = os.getenv("DINO_MODEL_ID", "facebook/dinov2-base")
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "ViT-B-32")
CLIP_PRETRAINED = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")
CLASSIFICATION_BATCH_SIZE = int(os.getenv("CLASSIFICATION_BATCH_SIZE", "16"))
CLIP_MIN_CONFIDENCE = float(os.getenv("CLIP_MIN_CONFIDENCE", "0.40"))

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(POSTMORTEM_ANNOTATED_DIR, exist_ok=True)
os.makedirs(POSTMORTEM_REPORTS_DIR, exist_ok=True)
os.makedirs(POSTMORTEM_FEATURES_DIR, exist_ok=True)
