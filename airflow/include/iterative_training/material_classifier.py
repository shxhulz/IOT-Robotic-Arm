import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from config import (
    CLASSIFICATION_BATCH_SIZE,
    CLIP_MIN_CONFIDENCE,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    DINO_MODEL_ID,
    MATERIAL_CLASSES,
    MATERIAL_PROMPT_TEMPLATE,
    MODEL_CACHE_DIR,
)

logger = logging.getLogger(__name__)

try:
    import open_clip

    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    logger.warning("open_clip_torch is not installed.")

try:
    from transformers import AutoImageProcessor, AutoModel

    DINO_AVAILABLE = True
except ImportError:
    DINO_AVAILABLE = False
    logger.warning("transformers is not installed.")

MATERIAL_CLASSIFIER_AVAILABLE = OPEN_CLIP_AVAILABLE and DINO_AVAILABLE


@dataclass
class CropPrediction:
    class_id: int
    class_name: str
    confidence: float
    clip_scores: dict[str, float]
    dino_embedding: np.ndarray


class MaterialClassifier:
    """Zero-shot material classifier using CLIP text prompts plus DINOv2 features."""

    def __init__(
        self,
        material_classes: Optional[list[str]] = None,
        batch_size: int = CLASSIFICATION_BATCH_SIZE,
        min_confidence: float = CLIP_MIN_CONFIDENCE,
    ):
        if not MATERIAL_CLASSIFIER_AVAILABLE:
            raise RuntimeError("Material classifier dependencies are not installed.")

        self.material_classes = material_classes or MATERIAL_CLASSES
        self.batch_size = batch_size
        self.min_confidence = min_confidence
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.set_float32_matmul_precision("high")
        self._configure_model_cache()
        self._load_models()

    def _configure_model_cache(self) -> None:
        model_dir = Path(MODEL_CACHE_DIR)
        os.environ.setdefault("TORCH_HOME", str(model_dir / "torch"))
        os.environ.setdefault("HF_HOME", str(model_dir / "hf"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(model_dir / "hf"))
        os.environ.setdefault("OPENCLIP_CACHE_DIR", str(model_dir / "clip"))

    def _load_models(self) -> None:
        hf_cache = Path(MODEL_CACHE_DIR) / "hf"
        clip_cache = Path(MODEL_CACHE_DIR) / "clip"

        logger.info("Loading DINOv2 model: %s", DINO_MODEL_ID)
        self.dino_processor = AutoImageProcessor.from_pretrained(
            DINO_MODEL_ID,
            cache_dir=str(hf_cache),
        )
        self.dino_model = AutoModel.from_pretrained(
            DINO_MODEL_ID,
            cache_dir=str(hf_cache),
        ).to(self.device)
        self.dino_model.eval()

        logger.info("Loading CLIP model: %s (%s)", CLIP_MODEL_NAME, CLIP_PRETRAINED)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED,
            device=self.device,
            cache_dir=str(clip_cache),
        )
        self.clip_model.eval()
        self.clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)

        text_prompts = [
            MATERIAL_PROMPT_TEMPLATE.format(material=name) for name in self.material_classes
        ]
        with torch.no_grad():
            tokens = self.clip_tokenizer(text_prompts).to(self.device)
            text_features = self.clip_model.encode_text(tokens)
            self.text_features = torch.nn.functional.normalize(text_features, dim=-1)

        logger.info("Material classes configured: %s", ", ".join(self.material_classes))

    def classify_crops(self, crops: list[np.ndarray]) -> list[Optional[CropPrediction]]:
        if not crops:
            return []

        predictions: list[Optional[CropPrediction]] = []
        for start_idx in range(0, len(crops), self.batch_size):
            batch = crops[start_idx : start_idx + self.batch_size]
            predictions.extend(self._classify_batch(batch))
        return predictions

    def _classify_batch(self, crops: list[np.ndarray]) -> list[Optional[CropPrediction]]:
        pil_crops = [Image.fromarray(crop) for crop in crops]
        dino_inputs = self.dino_processor(images=pil_crops, return_tensors="pt")
        dino_inputs = {name: value.to(self.device) for name, value in dino_inputs.items()}
        clip_inputs = torch.stack(
            [self.clip_preprocess(image) for image in pil_crops]
        ).to(self.device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
                dino_outputs = self.dino_model(**dino_inputs)
                dino_embeddings = dino_outputs.last_hidden_state[:, 0, :]
                dino_embeddings = torch.nn.functional.normalize(dino_embeddings, dim=-1)

                clip_embeddings = self.clip_model.encode_image(clip_inputs)
                clip_embeddings = torch.nn.functional.normalize(clip_embeddings, dim=-1)
                clip_scores = clip_embeddings @ self.text_features.T
                clip_probabilities = (100.0 * clip_scores).softmax(dim=-1)

        batch_predictions: list[Optional[CropPrediction]] = []
        for idx in range(len(crops)):
            per_class_scores = {
                class_name: float(clip_probabilities[idx][class_idx].item())
                for class_idx, class_name in enumerate(self.material_classes)
            }
            best_score, best_class_idx = torch.max(clip_probabilities[idx], dim=0)
            confidence = float(best_score.item())

            if confidence < self.min_confidence:
                batch_predictions.append(None)
                continue

            class_name = self.material_classes[int(best_class_idx.item())]
            batch_predictions.append(
                CropPrediction(
                    class_id=int(best_class_idx.item()),
                    class_name=class_name,
                    confidence=confidence,
                    clip_scores=per_class_scores,
                    dino_embedding=dino_embeddings[idx].detach().cpu().float().numpy(),
                )
            )

        return batch_predictions
