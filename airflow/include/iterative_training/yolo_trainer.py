import os
import yaml
import logging

import torch
import mlflow
from ultralytics import YOLO, settings
from config import (
    TRAINING_DATA_DIR,
    YOLO_MODEL_PATH,
    YOLO_BASE_MODEL,
    YOLO_TRAIN_EPOCHS,
    YOLO_IMG_SIZE,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MATERIAL_CLASSES,
)

logger = logging.getLogger(__name__)


class YOLOTrainer:
    def __init__(self, model_path=YOLO_MODEL_PATH):

        if os.path.exists(model_path):
            logger.info("Fine-tuning from existing weights: %s", model_path)
            self.model = YOLO(model_path)
        else:
            logger.info(
                "No existing weights at %s — starting from base model: %s",
                model_path,
                YOLO_BASE_MODEL,
            )
            self.model = YOLO(YOLO_BASE_MODEL)
        
        settings.update({"mlflow": True})
        logger.info("MLflow enabled for YOLO training.")
        
        self.data_yaml_path = os.path.join(TRAINING_DATA_DIR, "data.yaml")

    def create_data_yaml(self):
        """Creates data.yaml with proper train/val split directories."""
        train_images = os.path.abspath(
            os.path.join(TRAINING_DATA_DIR, "train", "images")
        )
        val_images = os.path.abspath(
            os.path.join(TRAINING_DATA_DIR, "val", "images")
        )

        for split_name, split_dir in [("train", train_images), ("val", val_images)]:
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(
                    f"{split_name} images directory not found: {split_dir}"
                )
            image_count = len([
                f for f in os.listdir(split_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            if image_count == 0:
                raise ValueError(
                    f"No images found in {split_name} directory: {split_dir}"
                )
            logger.info("%s split: %d images in %s", split_name, image_count, split_dir)

        train_label_counts = self._count_split_labels("train")
        val_label_counts = self._count_split_labels("val")
        logger.info("Train label distribution: %s", train_label_counts)
        logger.info("Val label distribution: %s", val_label_counts)

        data = {
            "train": train_images,
            "val": val_images,
            "nc": len(MATERIAL_CLASSES),
            "names": MATERIAL_CLASSES,
        }

        with open(self.data_yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info("Created data.yaml at %s", self.data_yaml_path)
        logger.info("YOLO classes configured: nc=%d names=%s", len(MATERIAL_CLASSES), MATERIAL_CLASSES)

    def _count_split_labels(self, split_name: str) -> dict:
        label_dir = os.path.join(TRAINING_DATA_DIR, split_name, "labels")
        counts = {name: 0 for name in MATERIAL_CLASSES}
        counts["unknown"] = 0

        if not os.path.isdir(label_dir):
            return counts

        class_map = {str(idx): name for idx, name in enumerate(MATERIAL_CLASSES)}
        label_files = [f for f in os.listdir(label_dir) if f.lower().endswith(".txt")]
        for label_file in label_files:
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_name = class_map.get(parts[0], "unknown")
                    counts[class_name] += 1
        return counts

    def train(self, epochs=YOLO_TRAIN_EPOCHS, img_size=YOLO_IMG_SIZE):
        """Trains the YOLO model with MLflow tracking."""
        self.create_data_yaml()

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info("MLflow tracking URI: %s", MLFLOW_TRACKING_URI)
        logger.info("MLflow experiment: %s", MLFLOW_EXPERIMENT_NAME)

        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
        os.environ["MLFLOW_EXPERIMENT_NAME"] = MLFLOW_EXPERIMENT_NAME

        cuda_requested = os.getenv("USE_CUDA", "True") == "True"
        cuda_available = torch.cuda.is_available()

        if cuda_requested and cuda_available:
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info("GPU SELECTED: %s (%.2f GB)", gpu_name, gpu_mem)
            logger.info(
                "GPU memory before training: %.2f MB allocated, %.2f MB cached",
                torch.cuda.memory_allocated(0) / (1024 ** 2),
                torch.cuda.memory_reserved(0) / (1024 ** 2),
            )
        elif cuda_requested and not cuda_available:
            device = "cpu"
            logger.warning(
                "USE_CUDA=True but CUDA is NOT available — falling back to CPU."
            )
        else:
            device = "cpu"
            logger.info("CPU training mode selected (USE_CUDA != True).")

        logger.info(
            "Starting YOLO training — epochs=%d, imgsz=%d, device=%s",
            epochs,
            img_size,
            device,
        )

        with mlflow.start_run(run_name="yolo_train") as run:
            mlflow.log_params({
                "epochs": epochs,
                "img_size": img_size,
                "device": device,
                "base_model": str(self.model.model_name)
                if hasattr(self.model, "model_name")
                else "unknown",
            })

            results = self.model.train(
                data=self.data_yaml_path,
                epochs=epochs,
                imgsz=img_size,
                device=device,
                project=os.path.join(TRAINING_DATA_DIR, "runs"),
                name="train",
                exist_ok=False,
            )

            if device == "cuda" and torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
                logger.info("GPU peak memory during training: %.2f MB", peak_mem)
                mlflow.log_metric("gpu_peak_memory_mb", peak_mem)

            best_weights = self.get_latest_weights()
            if best_weights:
                mlflow.log_artifact(best_weights, artifact_path="weights")

            logger.info("MLflow run ID: %s", run.info.run_id)

        logger.info("YOLO training completed.")
        return results

    def get_latest_weights(self):
        """Returns the path to the best weights from the latest training run."""
        runs_dir = os.path.join(TRAINING_DATA_DIR, "runs")
        if not os.path.exists(runs_dir):
            logger.warning("Runs directory not found: %s", runs_dir)
            return None

        train_dirs = sorted(
            [
                d
                for d in os.listdir(runs_dir)
                if d.startswith("train") and os.path.isdir(os.path.join(runs_dir, d))
            ],
            key=lambda x: os.path.getctime(os.path.join(runs_dir, x)),
        )

        if not train_dirs:
            logger.warning("No training run directories found in %s", runs_dir)
            return None

        latest_train = train_dirs[-1]
        best_pt = os.path.join(runs_dir, latest_train, "weights", "best.pt")

        if os.path.exists(best_pt):
            logger.info("Found best weights: %s", best_pt)
            return best_pt

        logger.warning("best.pt not found at %s", best_pt)
        return None
