import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import open_clip


@dataclass
class PipelineConfig:
    model_dir: Path = Path("/models")
    sam2_checkpoint: Path = Path("/models/sam2/sam2_hiera_large.pt")
    sam2_model_cfg: str = "sam2_hiera_l.yaml"
    dino_model_id: str = "facebook/dinov2-base"
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    material_prompts: tuple[str, ...] = (
        "metal",
        "plastic",
        "wood",
        "glass",
        "fabric",
        "paper",
        "ceramic",
        "rubber",
    )
    batch_size: int = 16
    min_mask_area: int = 1200
    clip_weight: float = 0.8
    sam_points_per_side: int = 32
    sam_pred_iou_thresh: float = 0.86
    sam_stability_score_thresh: float = 0.92


class MaterialClassifierPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_float32_matmul_precision("high")
        self._load_models()

    def _load_models(self) -> None:
        os.environ.setdefault("TORCH_HOME", str(self.config.model_dir / "torch"))
        os.environ.setdefault("HF_HOME", str(self.config.model_dir / "hf"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self.config.model_dir / "hf"))
        os.environ.setdefault("OPENCLIP_CACHE_DIR", str(self.config.model_dir / "clip"))

        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        sam_model = build_sam2(
            self.config.sam2_model_cfg,
            str(self.config.sam2_checkpoint),
            device=self.device,
            apply_postprocessing=True,
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(
            sam_model,
            points_per_side=self.config.sam_points_per_side,
            pred_iou_thresh=self.config.sam_pred_iou_thresh,
            stability_score_thresh=self.config.sam_stability_score_thresh,
        )

        self.dino_processor = AutoImageProcessor.from_pretrained(
            self.config.dino_model_id,
            cache_dir=str(self.config.model_dir / "hf"),
        )
        self.dino_model = AutoModel.from_pretrained(
            self.config.dino_model_id,
            cache_dir=str(self.config.model_dir / "hf"),
        ).to(self.device)
        self.dino_model.eval()

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            self.config.clip_model_name,
            pretrained=self.config.clip_pretrained,
            device=self.device,
            cache_dir=str(self.config.model_dir / "clip"),
        )
        self.clip_model.eval()
        self.clip_tokenizer = open_clip.get_tokenizer(self.config.clip_model_name)

        with torch.no_grad():
            text_tokens = self.clip_tokenizer(list(self.config.material_prompts)).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            self.text_features = torch.nn.functional.normalize(text_features, dim=-1)

    @staticmethod
    def _project_dino_to_clip_space(dino_feats: torch.Tensor, clip_dim: int) -> torch.Tensor:
        dino_dim = dino_feats.shape[-1]
        if dino_dim == clip_dim:
            projected = dino_feats
        elif dino_dim > clip_dim:
            projected = dino_feats[:, :clip_dim]
        else:
            pad = torch.zeros(
                dino_feats.shape[0],
                clip_dim - dino_dim,
                device=dino_feats.device,
                dtype=dino_feats.dtype,
            )
            projected = torch.cat([dino_feats, pad], dim=-1)
        return torch.nn.functional.normalize(projected, dim=-1)

    def _extract_regions(self, image_rgb: np.ndarray, masks: list[dict[str, Any]]) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        regions: list[np.ndarray] = []
        metas: list[dict[str, Any]] = []

        for mask in masks:
            area = int(mask.get("area", 0))
            if area < self.config.min_mask_area:
                continue

            x, y, w, h = [int(v) for v in mask["bbox"]]
            crop = image_rgb[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            regions.append(crop)
            metas.append(
                {
                    "bbox": [x, y, w, h],
                    "area": area,
                    "stability_score": float(mask.get("stability_score", 0.0)),
                    "predicted_iou": float(mask.get("predicted_iou", 0.0)),
                    "segmentation": mask.get("segmentation"),
                }
            )

        return regions, metas

    def _classify_batch(self, crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        dino_inputs = self.dino_processor(
            images=[Image.fromarray(c) for c in crops],
            return_tensors="pt",
        )
        dino_inputs = {k: v.to(self.device) for k, v in dino_inputs.items()}

        clip_inputs = torch.stack([self.clip_preprocess(Image.fromarray(c)) for c in crops]).to(self.device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
                dino_out = self.dino_model(**dino_inputs)
                dino_feats = dino_out.last_hidden_state[:, 0, :]

                clip_image_feats = self.clip_model.encode_image(clip_inputs)

        dino_feats = torch.nn.functional.normalize(dino_feats, dim=-1)
        clip_image_feats = torch.nn.functional.normalize(clip_image_feats, dim=-1)

        clip_sim = clip_image_feats @ self.text_features.T
        dino_proxy = self._project_dino_to_clip_space(dino_feats, self.text_features.shape[-1])
        dino_sim = dino_proxy @ self.text_features.T

        fused_sim = (self.config.clip_weight * clip_sim) + ((1.0 - self.config.clip_weight) * dino_sim)

        conf, idx = torch.max(fused_sim, dim=-1)
        return idx.detach().cpu().numpy(), conf.detach().cpu().numpy()

    def run(self, input_image: Path, output_image: Path) -> dict[str, Any]:
        image_bgr = cv2.imread(str(input_image), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {input_image}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image_rgb)

        crops, metas = self._extract_regions(image_rgb, masks)
        if not crops:
            output_image.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_image), image_bgr)
            result = {"input": str(input_image), "output": str(output_image), "regions": []}
            (output_image.with_suffix(".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result

        labels: list[str] = []
        scores: list[float] = []

        for i in range(0, len(crops), self.config.batch_size):
            batch_crops = crops[i : i + self.config.batch_size]
            pred_idx, conf = self._classify_batch(batch_crops)
            labels.extend([self.config.material_prompts[int(x)] for x in pred_idx])
            scores.extend([float(x) for x in conf])

        vis = image_bgr.copy()
        regions = []
        for i, meta in enumerate(metas):
            x, y, w, h = meta["bbox"]
            color = (0, int(100 + (i * 37) % 155), int(100 + (i * 73) % 155))
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            text = f"{labels[i]} ({scores[i]:.2f})"
            cv2.putText(vis, text, (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            regions.append(
                {
                    "bbox": meta["bbox"],
                    "label": labels[i],
                    "confidence": round(scores[i], 4),
                    "area": meta["area"],
                    "stability_score": meta["stability_score"],
                    "predicted_iou": meta["predicted_iou"],
                }
            )

        output_image.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_image), vis)

        result = {
            "input": str(input_image),
            "output": str(output_image),
            "num_masks_total": len(masks),
            "num_regions_classified": len(regions),
            "regions": regions,
        }
        (output_image.with_suffix(".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


def _parse_prompts(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return PipelineConfig.material_prompts
    prompts = [p.strip() for p in raw.split(",") if p.strip()]
    return tuple(prompts) if prompts else PipelineConfig.material_prompts


def main() -> None:
    image_path = os.getenv("IMAGE_PATH") or os.getenv("INPUT_IMAGE")
    if not image_path:
        raise ValueError("Set IMAGE_PATH or INPUT_IMAGE.")

    output_path = os.getenv("OUTPUT_IMAGE", "/output/annotated.png")
    model_dir = Path(os.getenv("MODEL_DIR", "/models"))

    config = PipelineConfig(
        model_dir=model_dir,
        sam2_checkpoint=Path(os.getenv("SAM2_CHECKPOINT", str(model_dir / "sam2/sam2_hiera_large.pt"))),
        sam2_model_cfg=os.getenv("SAM2_MODEL_CFG", "sam2_hiera_l.yaml"),
        material_prompts=_parse_prompts(os.getenv("MATERIAL_PROMPTS")),
        batch_size=int(os.getenv("BATCH_SIZE", "16")),
        min_mask_area=int(os.getenv("MIN_MASK_AREA", "1200")),
        clip_weight=float(os.getenv("CLIP_WEIGHT", "0.8")),
    )

    pipeline = MaterialClassifierPipeline(config)
    result = pipeline.run(Path(image_path), Path(output_path))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
