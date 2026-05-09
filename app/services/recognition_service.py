import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from app.config import settings
from app.services.image_utils import np_to_pil_rgb, cosine_similarity, softmax
from app.services.metadata_service import MetadataService


class RecognitionService:
    def __init__(self, metadata_service: MetadataService) -> None:
        self.metadata_service = metadata_service
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.model = AutoModel.from_pretrained("facebook/dinov2-small").to(self.device)
        self.model.eval()

        self.embeddings = None
        self.artifact_ids = []
        self._load_index()

    def _load_index(self) -> None:
        if os.path.exists(settings.ARTIFACT_EMBEDDINGS_PATH) and os.path.exists(settings.ARTIFACT_IDS_PATH):
            self.embeddings = np.load(settings.ARTIFACT_EMBEDDINGS_PATH)
            with open(settings.ARTIFACT_IDS_PATH, "r", encoding="utf-8") as f:
                self.artifact_ids = json.load(f)
        else:
            self.embeddings = np.empty((0, 384), dtype=np.float32)
            self.artifact_ids = []

    @torch.no_grad()
    def extract_embedding(self, image_bgr: np.ndarray) -> np.ndarray:
        pil_image = np_to_pil_rgb(image_bgr)
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy().astype(np.float32)
        return cls_embedding

    def recognize(self, image_bgr: np.ndarray, top_k: int = 5) -> dict:
        if self.embeddings is None or len(self.artifact_ids) == 0:
            return {
                "artifact_id": None,
                "name": None,
                "artifact_type": None,
                "museum": None,
                "era": None,
                "dynasty": None,
                "material": None,
                "confidence": 0.0,
                "similar_artifacts": [],
                "raw_scores": [],
            }

        query_embedding = self.extract_embedding(image_bgr)

        scored = []
        for idx, artifact_id in enumerate(self.artifact_ids):
            score = cosine_similarity(query_embedding, self.embeddings[idx])
            scored.append((artifact_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        top_scores = np.array([score for _, score in top], dtype=np.float32)
        probs = softmax(top_scores)

        similar_artifacts = []
        raw_scores = []

        for i, ((artifact_id, score), prob) in enumerate(zip(top, probs)):
            meta = self.metadata_service.get(artifact_id) or {}
            raw_scores.append({
                "rank": i + 1,
                "artifact_id": artifact_id,
                "score": float(score),
                "probability": float(prob),
                "name": meta.get("name"),
            })
            similar_artifacts.append({
                "artifact_id": artifact_id,
                "name": meta.get("name", "Unknown"),
                "museum": meta.get("museum", "Unknown"),
                "era": meta.get("era", "Unknown"),
                "confidence": round(float(prob), 4),
            })

        best_id = top[0][0]
        best_prob = float(probs[0])
        best_meta = self.metadata_service.get(best_id) or {}

        return {
            "artifact_id": best_id,
            "name": best_meta.get("name"),
            "artifact_type": best_meta.get("artifact_type"),
            "museum": best_meta.get("museum"),
            "era": best_meta.get("era"),
            "dynasty": best_meta.get("dynasty"),
            "material": best_meta.get("material"),
            "confidence": round(best_prob, 4),
            "similar_artifacts": similar_artifacts,
            "raw_scores": raw_scores,
        }