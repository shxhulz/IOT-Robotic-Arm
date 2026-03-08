#!/usr/bin/env python3
"""Train the YOLO model on the split dataset.

This script runs inside the DockerOperator GPU container.
It logs comprehensive GPU diagnostics before and after training.
"""

import os
import sys
import subprocess
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from yolo_trainer import YOLOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def log_gpu_diagnostics():
    """Log comprehensive GPU information for debugging and verification."""
    logger.info("=" * 60)
    logger.info("GPU DIAGNOSTICS")
    logger.info("=" * 60)

    try:
        import torch

        logger.info("PyTorch version: %s", torch.__version__)
        logger.info("CUDA available: %s", torch.cuda.is_available())
        logger.info("CUDA compiled version: %s", torch.version.cuda)

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info("CUDA device count: %d", device_count)
            for i in range(device_count):
                logger.info("  Device %d: %s", i, torch.cuda.get_device_name(i))
                props = torch.cuda.get_device_properties(i)
                logger.info("    Compute capability: %d.%d", props.major, props.minor)
                logger.info(
                    "    Total memory: %.2f GB",
                    props.total_memory / (1024 ** 3),
                )
                logger.info(
                    "    Allocated memory: %.2f MB",
                    torch.cuda.memory_allocated(i) / (1024 ** 2),
                )
                logger.info(
                    "    Cached memory: %.2f MB",
                    torch.cuda.memory_reserved(i) / (1024 ** 2),
                )

            logger.info("cuDNN enabled: %s", torch.backends.cudnn.enabled)
            logger.info("cuDNN version: %s", torch.backends.cudnn.version())
        else:
            logger.warning(
                "*** CUDA IS NOT AVAILABLE — training will fall back to CPU. ***"
            )
            logger.warning(
                "Check: (1) NVIDIA drivers, (2) nvidia-container-toolkit, "
                "(3) device_requests in DockerOperator"
            )
    except Exception as e:
        logger.error("Failed to query PyTorch CUDA info: %s", e)

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            logger.info("nvidia-smi output:\n%s", result.stdout)
        else:
            logger.warning("nvidia-smi failed (rc=%d): %s", result.returncode, result.stderr)
    except FileNotFoundError:
        logger.warning("nvidia-smi not found in PATH.")
    except Exception as e:
        logger.warning("Could not run nvidia-smi: %s", e)

    logger.info("=" * 60)


def log_post_training_gpu():
    """Log GPU memory usage after training for performance analysis."""
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("=" * 60)
            logger.info("POST-TRAINING GPU STATS")
            logger.info("=" * 60)
            for i in range(torch.cuda.device_count()):
                logger.info(
                    "  Device %d peak memory: %.2f MB",
                    i,
                    torch.cuda.max_memory_allocated(i) / (1024 ** 2),
                )
                logger.info(
                    "  Device %d peak cached: %.2f MB",
                    i,
                    torch.cuda.max_memory_reserved(i) / (1024 ** 2),
                )
            logger.info("=" * 60)
    except Exception as e:
        logger.warning("Could not log post-training GPU stats: %s", e)


def main():
    log_gpu_diagnostics()
    trainer = YOLOTrainer()
    results = trainer.train()
    log_post_training_gpu()
    best_weights = trainer.get_latest_weights()
    if best_weights:
        marker_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            "data",
            ".best_weights_path",
        )
        with open(marker_path, "w") as f:
            f.write(best_weights)
        logger.info("Training complete. Best weights at: %s", best_weights)
    else:
        logger.error("Training completed but no best.pt was found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
