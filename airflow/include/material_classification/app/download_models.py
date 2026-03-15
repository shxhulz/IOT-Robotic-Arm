import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve

from transformers import AutoImageProcessor, AutoModel
import open_clip


SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_sam2(models_dir: Path) -> None:
    target = models_dir / "sam2" / "sam2_hiera_large.pt"
    ensure_dir(target.parent)
    if target.exists():
        return
    print(f"Downloading SAM2 checkpoint to {target}...")
    urlretrieve(SAM2_URL, target)


def warm_dino(models_dir: Path) -> None:
    hf_dir = models_dir / "hf"
    ensure_dir(hf_dir)
    AutoImageProcessor.from_pretrained("facebook/dinov2-base", cache_dir=str(hf_dir))
    AutoModel.from_pretrained("facebook/dinov2-base", cache_dir=str(hf_dir))


def warm_clip(models_dir: Path) -> None:
    clip_dir = models_dir / "clip"
    ensure_dir(clip_dir)
    open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        cache_dir=str(clip_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and cache model artifacts under /models")
    parser.add_argument("--model-dir", default=os.getenv("MODEL_DIR", "/models"))
    parser.add_argument("--best-effort", action="store_true", help="Do not fail the process if any model download fails")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    os.environ.setdefault("HF_HOME", str(model_dir / "hf"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(model_dir / "hf"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(model_dir / "clip"))

    steps = [download_sam2, warm_dino, warm_clip]
    for step in steps:
        try:
            step(model_dir)
            print(f"[OK] {step.__name__}")
        except Exception as exc:
            if not args.best_effort:
                raise
            print(f"[WARN] {step.__name__} failed: {exc}")


if __name__ == "__main__":
    main()
