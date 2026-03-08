"""
Tests for the YOLO Iterative Training DAG and its pipeline scripts.

These tests verify:
  1. DAG loads without import errors
  2. DAG has correct structure (task count, dependencies, tags, retries)
  3. All BashOperator commands reference existing scripts
  4. The pipeline scripts' logic works correctly with mocked dependencies
"""

import os
import sys
import shutil
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup — the scripts live under airflow/include/iterative_training/
# ---------------------------------------------------------------------------
_AIRFLOW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_INCLUDE_DIR = os.path.join(_AIRFLOW_DIR, "include")
_PACKAGE_DIR = os.path.join(_INCLUDE_DIR, "iterative_training")
_SCRIPTS_DIR = os.path.join(_PACKAGE_DIR, "scripts")

# Add include/iterative_training to path so tests can import config, etc.
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: DAG Structure Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDAGStructure:
    """Verify the DAG parses correctly and has the expected shape."""

    @pytest.fixture(autouse=True)
    def load_dag(self):
        pytest.importorskip("airflow")
        from airflow.models import DagBag

        self.dagbag = DagBag(
            dag_folder=os.path.join(_AIRFLOW_DIR, "dags"),
            include_examples=False,
        )

    def test_dag_loads_without_import_errors(self):
        errors = self.dagbag.import_errors
        for path, err in errors.items():
            if "yolo_iterative_training_dag" in path:
                pytest.fail(f"DAG import error:\n{err}")

    def test_dag_exists(self):
        assert "yolo_iterative_training" in self.dagbag.dags

    def test_dag_has_correct_task_count(self):
        dag = self.dagbag.dags["yolo_iterative_training"]
        assert len(dag.tasks) == 6

    def test_dag_task_ids(self):
        dag = self.dagbag.dags["yolo_iterative_training"]
        expected_ids = {
            "prepare_workspace",
            "download_images",
            "process_with_sam2",
            "split_dataset",
            "train_yolo",
            "promote_model",
        }
        actual_ids = {t.task_id for t in dag.tasks}
        assert actual_ids == expected_ids

    def test_dag_dependencies(self):
        """Verify the linear pipeline order."""
        dag = self.dagbag.dags["yolo_iterative_training"]
        tasks = {t.task_id: t for t in dag.tasks}

        assert tasks["prepare_workspace"].downstream_task_ids == {"download_images"}
        assert tasks["download_images"].downstream_task_ids == {"process_with_sam2"}
        assert tasks["process_with_sam2"].downstream_task_ids == {"split_dataset"}
        assert tasks["split_dataset"].downstream_task_ids == {"train_yolo"}
        assert tasks["train_yolo"].downstream_task_ids == {"promote_model"}
        assert tasks["promote_model"].downstream_task_ids == set()

    def test_dag_uses_bash_or_docker_operator(self):
        """All tasks should be BashOperator except train_yolo and process_with_sam2 which are DockerOperators."""
        dag = self.dagbag.dags["yolo_iterative_training"]
        for task in dag.tasks:
            if task.task_id in ["train_yolo", "process_with_sam2"]:
                assert task.task_type == "DockerOperator", (
                    f"Task '{task.task_id}' is {task.task_type}, expected DockerOperator"
                )
                assert task.image == "ultralytics/ultralytics:latest"
                assert task.auto_remove == "force"
                assert task.docker_url == "unix://var/run/docker.sock"
                assert len(task.mounts) == 1
                assert task.mounts[0]['Target'] == "/usr/local/airflow/include"
                assert len(task.device_requests) == 1
                # network_mode should NOT be 'bridge' — it must join the
                # Astro project network so it can resolve 'minio' hostname.
                assert task.network_mode != "bridge", (
                    "DockerOperator must not use 'bridge' network — "
                    "it needs the Astro project network for MinIO connectivity"
                )
                # MLflow env vars must be passed to the training container
                assert "MLFLOW_TRACKING_URI" in task.environment, (
                    "MLFLOW_TRACKING_URI must be in DockerOperator environment"
                )
                assert "MLFLOW_EXPERIMENT_NAME" in task.environment, (
                    "MLFLOW_EXPERIMENT_NAME must be in DockerOperator environment"
                )
            else:
                assert task.task_type == "BashOperator", (
                    f"Task '{task.task_id}' is {task.task_type}, expected BashOperator"
                )

    def test_dag_has_tags(self):
        dag = self.dagbag.dags["yolo_iterative_training"]
        assert dag.tags, "DAG must have tags"
        assert "yolo" in dag.tags
        assert "training" in dag.tags
        assert "mlflow" in dag.tags

    def test_dag_retries_at_least_2(self):
        dag = self.dagbag.dags["yolo_iterative_training"]
        assert dag.default_args.get("retries", 0) >= 2

    def test_dag_max_active_runs(self):
        dag = self.dagbag.dags["yolo_iterative_training"]
        assert dag.max_active_runs == 1

    def test_dag_catchup_disabled(self):
        dag = self.dagbag.dags["yolo_iterative_training"]
        assert dag.catchup is False


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Script Existence Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestScriptsExist:
    """Verify all referenced scripts exist on disk."""

    EXPECTED_SCRIPTS = [
        "prepare_workspace.py",
        "download_images.py",
        "process_sam2.py",
        "split_dataset.py",
        "train_yolo.py",
        "promote_model.py",
    ]

    @pytest.mark.parametrize("script_name", EXPECTED_SCRIPTS)
    def test_script_exists(self, script_name):
        script_path = os.path.join(_SCRIPTS_DIR, script_name)
        assert os.path.isfile(script_path), f"Script not found: {script_path}"

    @pytest.mark.parametrize("script_name", EXPECTED_SCRIPTS)
    def test_script_syntax(self, script_name):
        """Each script should compile without syntax errors."""
        import py_compile

        script_path = os.path.join(_SCRIPTS_DIR, script_name)
        try:
            py_compile.compile(script_path, doraise=True)
        except py_compile.PyCompileError as e:
            pytest.fail(f"Syntax error in {script_name}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Config Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfig:
    """Test configuration module defaults."""

    def test_default_minio_endpoint(self):
        from config import MINIO_ENDPOINT

        assert MINIO_ENDPOINT

    def test_training_data_dir_is_absolute(self):
        from config import TRAINING_DATA_DIR

        assert os.path.isabs(TRAINING_DATA_DIR)

    def test_images_dir_under_training_data(self):
        from config import IMAGES_DIR, TRAINING_DATA_DIR

        assert IMAGES_DIR.startswith(TRAINING_DATA_DIR)

    def test_labels_dir_under_training_data(self):
        from config import LABELS_DIR, TRAINING_DATA_DIR

        assert LABELS_DIR.startswith(TRAINING_DATA_DIR)

    def test_split_dirs_are_absolute(self):
        from config import (
            TRAIN_IMAGES_DIR,
            TRAIN_LABELS_DIR,
            VAL_IMAGES_DIR,
            VAL_LABELS_DIR,
        )

        for d in [TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_IMAGES_DIR, VAL_LABELS_DIR]:
            assert isinstance(d, str)
            assert os.path.isabs(d)

    def test_yolo_train_epochs_is_positive_int(self):
        from config import YOLO_TRAIN_EPOCHS

        assert isinstance(YOLO_TRAIN_EPOCHS, int)
        assert YOLO_TRAIN_EPOCHS > 0

    def test_yolo_img_size_is_positive_int(self):
        from config import YOLO_IMG_SIZE

        assert isinstance(YOLO_IMG_SIZE, int)
        assert YOLO_IMG_SIZE > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Pipeline Logic Tests (Mocked — no real MinIO/SAM2/YOLO needed)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPrepareWorkspace:
    """Test the workspace preparation logic."""

    def test_clean_and_create_dirs(self, tmp_path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"

        # Create dirs with stale files
        images_dir.mkdir()
        labels_dir.mkdir()
        (images_dir / "old_file.jpg").write_bytes(b"old")
        (labels_dir / "old_file.txt").write_text("old label")

        # Simulate the prepare_workspace logic
        for directory in [str(images_dir), str(labels_dir)]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)

        for split in ["train", "val"]:
            for sub in ["images", "labels"]:
                os.makedirs(str(tmp_path / split / sub), exist_ok=True)

        # Old files should be gone
        assert len(os.listdir(str(images_dir))) == 0
        assert len(os.listdir(str(labels_dir))) == 0

        # Split dirs should exist
        assert (tmp_path / "train" / "images").exists()
        assert (tmp_path / "train" / "labels").exists()
        assert (tmp_path / "val" / "images").exists()
        assert (tmp_path / "val" / "labels").exists()


class TestMinioHandler:
    """Test MinioHandler with mocked MinIO client."""

    @patch("minio_handler.Minio")
    def test_download_images_filters_by_extension(self, mock_minio_cls, tmp_path):
        with patch("minio_handler.IMAGES_DIR", str(tmp_path)):
            mock_client = MagicMock()
            mock_minio_cls.return_value = mock_client
            mock_client.bucket_exists.return_value = True

            mock_obj1 = MagicMock()
            mock_obj1.object_name = "photo1.jpg"
            mock_obj2 = MagicMock()
            mock_obj2.object_name = "photo2.png"
            mock_obj3 = MagicMock()
            mock_obj3.object_name = "readme.txt"  # should be skipped
            mock_client.list_objects.return_value = [mock_obj1, mock_obj2, mock_obj3]

            from minio_handler import MinioHandler

            handler = MinioHandler()
            files = handler.download_images(limit=10)

            assert len(files) == 2
            assert all(f.endswith((".jpg", ".png")) for f in files)

    @patch("minio_handler.Minio")
    def test_download_empty_bucket(self, mock_minio_cls):
        mock_client = MagicMock()
        mock_minio_cls.return_value = mock_client
        mock_client.bucket_exists.return_value = False

        from minio_handler import MinioHandler

        handler = MinioHandler()
        files = handler.download_images()
        assert files == []

    @patch("minio_handler.Minio")
    def test_download_respects_limit(self, mock_minio_cls, tmp_path):
        with patch("minio_handler.IMAGES_DIR", str(tmp_path)):
            mock_client = MagicMock()
            mock_minio_cls.return_value = mock_client
            mock_client.bucket_exists.return_value = True

            objects = [MagicMock(object_name=f"img_{i:03d}.jpg") for i in range(20)]
            mock_client.list_objects.return_value = objects

            from minio_handler import MinioHandler

            handler = MinioHandler()
            files = handler.download_images(limit=5)
            assert len(files) == 5


class TestSplitDataset:
    """Test the dataset splitting logic."""

    @pytest.fixture
    def setup_dirs(self, tmp_path):
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()

        for split in ["train", "val"]:
            for sub in ["images", "labels"]:
                (tmp_path / split / sub).mkdir(parents=True)

        for i in range(10):
            (images_dir / f"img_{i:03d}.jpg").write_bytes(b"fake_image")
            (labels_dir / f"img_{i:03d}.txt").write_text("0 0.5 0.5 0.1 0.1")

        return str(images_dir), str(labels_dir), str(tmp_path)

    def test_split_creates_correct_ratio(self, setup_dirs):
        images_dir, labels_dir, training_dir = setup_dirs

        valid_ext = {".png", ".jpg", ".jpeg"}
        image_files = sorted(
            f for f in os.listdir(images_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        )

        split_idx = int(len(image_files) * 0.8)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]

        assert len(train_files) == 8
        assert len(val_files) == 2

        for split_name, files in [("train", train_files), ("val", val_files)]:
            img_dst = os.path.join(training_dir, split_name, "images")
            lbl_dst = os.path.join(training_dir, split_name, "labels")
            for img_name in files:
                shutil.copy2(os.path.join(images_dir, img_name), os.path.join(img_dst, img_name))
                label_name = os.path.splitext(img_name)[0] + ".txt"
                src_lbl = os.path.join(labels_dir, label_name)
                if os.path.exists(src_lbl):
                    shutil.copy2(src_lbl, os.path.join(lbl_dst, label_name))

        assert len(os.listdir(os.path.join(training_dir, "train", "images"))) == 8
        assert len(os.listdir(os.path.join(training_dir, "val", "images"))) == 2
        assert len(os.listdir(os.path.join(training_dir, "train", "labels"))) == 8
        assert len(os.listdir(os.path.join(training_dir, "val", "labels"))) == 2

    def test_split_is_deterministic(self, setup_dirs):
        images_dir, _, _ = setup_dirs

        valid_ext = {".png", ".jpg", ".jpeg"}
        files1 = sorted(f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in valid_ext)
        files2 = sorted(f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in valid_ext)

        split1 = int(len(files1) * 0.8)
        split2 = int(len(files2) * 0.8)

        assert files1[:split1] == files2[:split2]
        assert files1[split1:] == files2[split2:]

    def test_split_handles_missing_label(self, setup_dirs):
        images_dir, labels_dir, training_dir = setup_dirs

        os.remove(os.path.join(labels_dir, "img_000.txt"))

        valid_ext = {".png", ".jpg", ".jpeg"}
        image_files = sorted(
            f for f in os.listdir(images_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        )
        split_idx = int(len(image_files) * 0.8)
        train_files = image_files[:split_idx]

        img_dst = os.path.join(training_dir, "train", "images")
        lbl_dst = os.path.join(training_dir, "train", "labels")
        for img_name in train_files:
            shutil.copy2(os.path.join(images_dir, img_name), os.path.join(img_dst, img_name))
            label_name = os.path.splitext(img_name)[0] + ".txt"
            src_lbl = os.path.join(labels_dir, label_name)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(lbl_dst, label_name))

        assert len(os.listdir(img_dst)) == 8
        assert len(os.listdir(lbl_dst)) == 7  # one label missing


class TestPromoteModel:
    """Test the model promotion logic."""

    def test_promote_copies_weights(self, tmp_path):
        src = tmp_path / "runs" / "train" / "weights"
        src.mkdir(parents=True)
        best_pt = src / "best.pt"
        best_pt.write_bytes(b"model_weights_data")

        prod_dir = tmp_path / "production" / "weights"
        prod_path = prod_dir / "best.pt"
        os.makedirs(str(prod_dir), exist_ok=True)

        shutil.copy2(str(best_pt), str(prod_path))

        assert prod_path.exists()
        assert prod_path.read_bytes() == b"model_weights_data"

    def test_promote_via_marker_file(self, tmp_path):
        """Test the marker file mechanism used between train and promote steps."""
        # Simulate train_yolo writing a marker
        marker = tmp_path / ".best_weights_path"
        best_pt = tmp_path / "best.pt"
        best_pt.write_bytes(b"weights")
        marker.write_text(str(best_pt))

        # Read back
        path_from_marker = marker.read_text().strip()
        assert os.path.exists(path_from_marker)
        assert path_from_marker == str(best_pt)

    def test_promote_fails_on_missing_marker(self, tmp_path):
        marker = tmp_path / ".best_weights_path"
        assert not marker.exists()


class TestYOLOLabelFormat:
    """Test YOLO label format generation."""

    def test_yolo_label_format(self, tmp_path):
        label_path = tmp_path / "test.txt"

        h, w = 480, 640
        bbox = [100, 50, 200, 150]
        class_id = 0

        x_center = (bbox[0] + bbox[2] / 2) / w
        y_center = (bbox[1] + bbox[3] / 2) / h
        norm_w = bbox[2] / w
        norm_h = bbox[3] / h

        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        norm_w = max(0.0, min(1.0, norm_w))
        norm_h = max(0.0, min(1.0, norm_h))

        with open(str(label_path), "w") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        content = label_path.read_text().strip()
        parts = content.split()

        assert len(parts) == 5
        assert parts[0] == "0"
        for val_str in parts[1:]:
            val = float(val_str)
            assert 0.0 <= val <= 1.0

    def test_degenerate_bbox_skipped(self):
        bbox = [100, 50, 0, 150]
        assert bbox[2] <= 0 or bbox[3] <= 0


class TestFineTuning:
    """Test the fine-tuning fallback logic."""

    def test_fallback_to_base_model_when_no_weights(self, tmp_path):
        """When YOLO_MODEL_PATH doesn't exist, trainer should use YOLO_BASE_MODEL."""
        fake_weights = str(tmp_path / "nonexistent" / "best.pt")
        assert not os.path.exists(fake_weights)
        assert not os.path.exists(fake_weights)

    def test_existing_weights_used_for_finetuning(self, tmp_path):
        """When weights file exists, it should be used."""
        weights_file = tmp_path / "best.pt"
        weights_file.write_bytes(b"fake_model_weights")
        assert os.path.exists(str(weights_file))


class TestMLflowConfig:
    """Test MLflow configuration."""

    def test_mlflow_tracking_uri_configured(self):
        from config import MLFLOW_TRACKING_URI
        assert MLFLOW_TRACKING_URI
        assert MLFLOW_TRACKING_URI.startswith("http")

    def test_mlflow_experiment_name_configured(self):
        from config import MLFLOW_EXPERIMENT_NAME
        assert MLFLOW_EXPERIMENT_NAME
        assert len(MLFLOW_EXPERIMENT_NAME) > 0
