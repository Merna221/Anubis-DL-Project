from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from app.services.metadata_service import MetadataService
from app.services.recognition_service import RecognitionService
from app.services.restoration_service import RestorationService


class AnalyzeService:
    def __init__(self) -> None:
        self.metadata_service = MetadataService()
        self.recognition_service = RecognitionService(self.metadata_service)
        self.restoration_service = RestorationService()

    def _normalize_metadata(self, metadata: Optional[dict], artifact_id: Optional[str]) -> dict:
        if not metadata:
            return {
                "artifact_id": artifact_id,
                "name": None,
                "artifact_type": None,
                "museum": None,
                "era": None,
                "dynasty": None,
                "material": None,
                "location": None,
                "description_en": None,
                "description_ar": None,
                "story_en": None,
                "story_ar": None,
                "audio_script_en": None,
                "audio_script_ar": None,
            }

        return {
            "artifact_id": metadata.get("artifact_id", artifact_id),
            "name": metadata.get("name"),
            "artifact_type": metadata.get("artifact_type") or metadata.get("type"),
            "museum": metadata.get("museum"),
            "era": metadata.get("era"),
            "dynasty": metadata.get("dynasty"),
            "material": metadata.get("material"),
            "location": metadata.get("location"),
            "description_en": metadata.get("description_en"),
            "description_ar": metadata.get("description_ar"),
            "story_en": metadata.get("story_en"),
            "story_ar": metadata.get("story_ar"),
            "audio_script_en": metadata.get("audio_script_en"),
            "audio_script_ar": metadata.get("audio_script_ar"),
        }

    def _normalize_recognition(self, result: Dict[str, Any]) -> Dict[str, Any]:
        artifact_id = result.get("artifact_id")
        return {
            "artifact_id": artifact_id,
            "name": result.get("name"),
            "artifact_type": result.get("artifact_type") or result.get("type"),
            "museum": result.get("museum"),
            "era": result.get("era"),
            "dynasty": result.get("dynasty"),
            "material": result.get("material"),
            "confidence": float(result.get("confidence", 0.0)),
            "similar_artifacts": result.get("similar_artifacts", []),
            "raw_scores": result.get("raw_scores", []),
        }

    def analyze_image(
        self,
        image_bgr: np.ndarray,
        scanned_filename: str,
        base_url: Optional[str] = None,
    ) -> dict:
        recognition_result = self.recognition_service.recognize(image_bgr)
        recognition = self._normalize_recognition(recognition_result)

        artifact_id = recognition.get("artifact_id")
        metadata = self.metadata_service.get(artifact_id) if artifact_id else None
        metadata = self._normalize_metadata(metadata, artifact_id)

        restoration = self.restoration_service.get_saved_restoration(artifact_id)

        final_image_url = None
        if restoration.get("available") and restoration.get("final_image_path") and artifact_id and base_url:
            final_image_name = restoration["final_image_path"].replace("\\", "/").split("/")[-1]
            final_image_url = f"{base_url.rstrip('/')}/static/restoration/{artifact_id}/{final_image_name}"

        return {
            "success": True,
            "scanned_filename": scanned_filename,
            "recognition": recognition,
            "metadata": metadata,
            "restoration": {
                "available": restoration.get("available", False),
                "artifact_id": restoration.get("artifact_id"),
                "artifact_name": restoration.get("artifact_name"),
                "final_image_path": restoration.get("final_image_path"),
                "final_image_url": final_image_url,
                "bundle_path": restoration.get("bundle_path"),
                "workspace_path": restoration.get("workspace_path"),
            },
            "debug": recognition_result,
        }