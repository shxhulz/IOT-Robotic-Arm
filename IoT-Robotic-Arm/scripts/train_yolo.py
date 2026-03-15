import os
import shutil
import random
import yaml
from pathlib import Path
from collections import Counter
import mlflow
from mlflow import log_param, log_metric, log_artifact
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_DIR = PROJECT_ROOT / "downloaded_images"
DATA_DIR = PROJECT_ROOT / "data" / "waste_dataset"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_images"
MODELS_DIR = PROJECT_ROOT / "models"
TRAIN_RATIO = 0.9

CLASS_NAMES = ["plastic", "paper", "metal"]
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "yolo11n_waste_detection"


def analyze_annotations():
    """Analyze annotation files to get class distribution"""
    txt_files = [f for f in SOURCE_DIR.glob("*.txt") if f.name != "labels.txt"]

    class_counts = Counter()
    total_objects = 0

    for txt_file in txt_files:
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts and len(parts) >= 5:
                    try:
                        class_id = int(parts[0])
                        if 0 <= class_id < len(CLASS_NAMES):
                            class_counts[class_id] += 1
                            total_objects += 1
                    except ValueError:
                        continue

    class_distribution = {CLASS_NAMES[k]: v for k, v in sorted(class_counts.items())}

    return len(txt_files), class_distribution, total_objects


def copy_images_to_processed():
    """Copy images and annotations to a new processed location"""
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    image_files = list(SOURCE_DIR.glob("*.jpg"))

    for img_file in image_files:
        shutil.copy2(img_file, PROCESSED_DIR / img_file.name)
        lbl_file = img_file.with_suffix(".txt")
        if lbl_file.exists() and lbl_file.name != "labels.txt":
            shutil.copy2(lbl_file, PROCESSED_DIR / lbl_file.name)

    print(f"  Copied {len(image_files)} images to {PROCESSED_DIR}")
    return PROCESSED_DIR


def split_data():
    """Split data into train and validation sets"""
    image_files = sorted([f for f in PROCESSED_DIR.glob("*.jpg")])
    random.seed(42)
    random.shuffle(image_files)

    split_idx = int(len(image_files) * TRAIN_RATIO)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    for split, files in [("train", train_files), ("val", val_files)]:
        img_dir = DATA_DIR / "images" / split
        lbl_dir = DATA_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_file in files:
            shutil.copy2(img_file, img_dir / img_file.name)
            lbl_file = img_file.with_suffix(".txt")
            if lbl_file.exists():
                shutil.copy2(lbl_file, lbl_dir / lbl_file.name)

    return len(train_files), len(val_files)


def create_data_yaml():
    """Create data.yaml for YOLO training"""
    data_config = {
        "path": str(DATA_DIR.absolute()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
        "nc": len(CLASS_NAMES),
    }

    yaml_path = DATA_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    return yaml_path


def get_next_model_version():
    """Determine next model version number"""
    MODELS_DIR.mkdir(exist_ok=True)
    existing = list(MODELS_DIR.glob("yolo11n_waste_v*.pt"))

    versions = []
    for f in existing:
        try:
            v = int(f.stem.split("v")[-1])
            versions.append(v)
        except:
            pass

    next_version = max(versions) + 1 if versions else 1
    return f"yolo11n_waste_v{next_version}.pt"


def train_model(data_yaml_path, model_version):
    """Train YOLO model with extensive augmentations"""
    model_path = PROJECT_ROOT / "src" / "yolo" / "weights" / "best.pt"
    model = YOLO(str(model_path))

    results = model.train(
        data=str(data_yaml_path),
        epochs=300,
        batch=16,
        imgsz=640,
        project=str(DATA_DIR),
        name="train",
        exist_ok=True,
        pretrained=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        augment=True,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0001,
        flipud=0.0,
        fliplr=0.5,
        amp=True,
        patience=50,
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        val=True,
        plots=True,
    )

    return results


def log_metrics_to_mlflow(results):
    """Log training metrics to MLflow"""
    if results and hasattr(results, "results_dict"):
        metrics = results.results_dict
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                log_metric(key, value)


def main():
    print("=" * 60)
    print("YOLO11n Waste Detection Training Script")
    print("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        print("\n[1/6] Analyzing dataset...")
        total_images, class_dist, total_objects = analyze_annotations()

        print(f"  Total images: {total_images}")
        print(f"  Total annotated objects: {total_objects}")
        print(f"  Class distribution:")
        for cls, count in class_dist.items():
            print(f"    - {cls}: {count} ({count / total_objects * 100:.1f}%)")

        log_param("total_images", total_images)
        log_param("total_objects", total_objects)
        log_param("train_ratio", TRAIN_RATIO)
        log_param("val_ratio", 1 - TRAIN_RATIO)
        for cls, count in class_dist.items():
            log_param(f"class_{cls}", count)

        print("\n[2/6] Copying images to processed location...")
        copy_images_to_processed()

        print("\n[3/6] Creating train/val split (90/10)...")
        if DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)

        train_count, val_count = split_data()
        print(f"  Train images: {train_count}")
        print(f"  Validation images: {val_count}")

        log_param("train_images", train_count)
        log_param("val_images", val_count)

        print("\n[4/6] Creating data.yaml...")
        data_yaml_path = create_data_yaml()
        print(f"  Saved to: {data_yaml_path}")

        print("\n[5/6] Training model (300 epochs with augmentations)...")
        model_version = get_next_model_version()
        print(f"  Model will be saved as: {model_version}")

        log_param("epochs", 300)
        log_param("batch_size", "auto")
        log_param("imgsz", 640)
        log_param("model_version", model_version)
        log_param("augmentations", "mosaic,mixup,hsv,affine,flip,perspective")

        train_results = train_model(data_yaml_path, model_version)

        if train_results:
            log_metrics_to_mlflow(train_results)

        print("\n[6/6] Saving model...")
        best_model_path = DATA_DIR / "train" / "weights" / "best.pt"
        final_model_path = MODELS_DIR / model_version

        if best_model_path.exists():
            shutil.copy2(best_model_path, final_model_path)
            print(f"  Model saved to: {final_model_path}")
            log_artifact(str(final_model_path))
        else:
            print(f"  ERROR: Best model not found at {best_model_path}")

        log_metric("training_completed", 1)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {final_model_path}")
    print("View MLflow UI: uv run mlflow ui")
    print("=" * 60)


if __name__ == "__main__":
    main()
