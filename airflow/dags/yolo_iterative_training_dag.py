"""
YOLO Iterative Training DAG

Runs every 12 hours:
  1. Prepare workspace — clean staging dirs for this run
  2. Download images from MinIO into a staging directory
  3. Process images with SAM2 to generate YOLO labels
  4. Organize dataset into train/val splits
  5. Train YOLO model
  6. Promote the best weights to the production model path

Uses BashOperator to invoke standalone Python scripts from
include/iterative_training/scripts/. Each script manages its own
imports via sys.path — no external package resolution needed.
"""
from datetime import timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount, DeviceRequest
from pendulum import datetime
import os

SCRIPTS_DIR = "/usr/local/airflow/include/iterative_training/scripts"
HOST_INCLUDE_DIR = os.getenv("HOST_INCLUDE_DIR", "d:/RoboticArm/IOT-Robotic-Arm/airflow/include")
DOCKER_NETWORK = os.getenv("ASTRO_DOCKER_NETWORK", "airflow_918bcf_airflow")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "yolo-iterative-training")

default_args = {
    "owner": "RoboticArm",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=4),
}

with DAG(
    dag_id="yolo_iterative_training",
    default_args=default_args,
    description="12-hour iterative training loop: MinIO → SAM2 → train/val split → YOLO → promote weights",
    schedule=timedelta(hours=12),
    start_date=datetime(2025, 3, 7),
    catchup=False,
    tags=["yolo", "sam2", "training", "iterative", "mlflow"],
    max_active_runs=1,  
) as dag:

    env = {
        "MINIO_ENDPOINT": "host.docker.internal:9000",
        "USE_CUDA": "True",
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
        "MLFLOW_EXPERIMENT_NAME": MLFLOW_EXPERIMENT_NAME,
    }

    t_prepare = BashOperator(
        task_id="prepare_workspace",
        bash_command=f"python {SCRIPTS_DIR}/prepare_workspace.py",
        env=env,
    )

    t_download = BashOperator(
        task_id="download_images",
        bash_command=f"python {SCRIPTS_DIR}/download_images.py",
        env=env,
    )

    t_sam2 = DockerOperator(
        task_id="process_with_sam2",
        image="ultralytics/ultralytics:latest",
        command=(
            f"bash -c 'pip install git+https://github.com/facebookresearch/sam2.git "
            "open_clip_torch minio && "
            f"python {SCRIPTS_DIR}/process_sam2.py'"
        ),
        mounts=[
            Mount(
                source=HOST_INCLUDE_DIR, 
                target="/usr/local/airflow/include", 
                type="bind"
            )
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        auto_remove="force",
        mount_tmp_dir=False,
        environment=env,
        device_requests=[DeviceRequest(count=-1, capabilities=[['gpu']])],
        shm_size="8g",
    )

    t_split = BashOperator(
        task_id="split_dataset",
        bash_command=f"python {SCRIPTS_DIR}/split_dataset.py",
        env=env,
    )

    t_train = DockerOperator(
        task_id="train_yolo",
        image="ultralytics/ultralytics:latest",
        command=f"bash -c 'pip install mlflow boto3 && python {SCRIPTS_DIR}/train_yolo.py'",
        mounts=[
            Mount(
                source=HOST_INCLUDE_DIR, 
                target="/usr/local/airflow/include", 
                type="bind"
            )
        ],
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        auto_remove="force",
        mount_tmp_dir=False,
        environment=env,
        device_requests=[DeviceRequest(count=-1, capabilities=[['gpu']])],
        force_pull=True,
        shm_size="8g", 
    )

    t_promote = BashOperator(
        task_id="promote_model",
        bash_command=f"python {SCRIPTS_DIR}/promote_model.py",
        env=env,
    )

    t_prepare >> t_download >> t_sam2 >> t_split >> t_train >> t_promote
