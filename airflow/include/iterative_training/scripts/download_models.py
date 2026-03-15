#!/usr/bin/env python3
"""Warm model caches under /models for SAM2, DINOv2, and CLIP."""

import argparse
import logging
import os
import sys
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    DINO_MODEL_ID,
    MODEL_CACHE_DIR,
    SAM2_CHECKPOINT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"


def _configure_cache_env() -> None:
    os.environ.setdefault("TORCH_HOME", os.path.join(MODEL_CACHE_DIR, "torch"))
    os.environ.setdefault("HF_HOME", os.path.join(MODEL_CACHE_DIR, "hf"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(MODEL_CACHE_DIR, "hf"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", os.path.join(MODEL_CACHE_DIR, "clip"))


def _download_sam2() -> None:
    os.makedirs(os.path.dirname(SAM2_CHECKPOINT), exist_ok=True)
    if os.path.exists(SAM2_CHECKPOINT):
        logger.info("SAM2 checkpoint already cached: %s", SAM2_CHECKPOINT)
        return
    logger.info("Downloading SAM2 checkpoint to %s", SAM2_CHECKPOINT)
    urllib.request.urlretrieve(SAM2_URL, SAM2_CHECKPOINT)


def _warm_dinov2() -> None:
    from transformers import AutoImageProcessor, AutoModel

    cache_dir = os.path.join(MODEL_CACHE_DIR, "hf")
    AutoImageProcessor.from_pretrained(DINO_MODEL_ID, cache_dir=cache_dir)
    AutoModel.from_pretrained(DINO_MODEL_ID, cache_dir=cache_dir)
    logger.info("DINOv2 cached: %s", DINO_MODEL_ID)


def _warm_clip() -> None:
    import open_clip

    open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
        cache_dir=os.path.join(MODEL_CACHE_DIR, "clip"),
    )
    logger.info("CLIP cached: %s (%s)", CLIP_MODEL_NAME, CLIP_PRETRAINED)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-effort", action="store_true")
    args = parser.parse_args()

    _configure_cache_env()
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    for step in (_download_sam2, _warm_dinov2, _warm_clip):
        try:
            step()
        except Exception as exc:
            if not args.best_effort:
                raise
            logger.warning("Model warmup step failed: %s (%s)", step.__name__, exc)


if __name__ == "__main__":
    main()
