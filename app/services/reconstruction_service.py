import os
from typing import Dict, List

from app.config import settings
from app.services.metadata_service import MetadataService


class ReconstructionService:
    def __init__(self, metadata_service: MetadataService) -> None:
        self.metadata_service = metadata_service

    def get_reference_images(self, artifact_id: str) -> List[str]:
        artifact_dir = os.path.join(settings.ARTIFACT_IMAGES_DIR, artifact_id)
        if not os.path.isdir(artifact_dir):
            return []

        image_paths = []
        for file_name in os.listdir(artifact_dir):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_paths.append(os.path.join(artifact_dir, file_name))

        image_paths.sort()
        return image_paths

    def get_training_pairs(self, artifact_id: str) -> Dict:
        pair_root = os.path.join(settings.DATA_DIR, "reconstruction_pairs", artifact_id)
        input_dir = os.path.join(pair_root, "input")
        target_dir = os.path.join(pair_root, "target")

        input_images = []
        target_images = []

        if os.path.isdir(input_dir):
            input_images = sorted(
                os.path.join(input_dir, f)
                for f in os.listdir(input_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            )

        if os.path.isdir(target_dir):
            target_images = sorted(
                os.path.join(target_dir, f)
                for f in os.listdir(target_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            )

        return {
            "artifact_id": artifact_id,
            "input_images": input_images,
            "target_images": target_images,
            "num_inputs": len(input_images),
            "num_targets": len(target_images),
        }

    def prepare_reconstruction_bundle(self, artifact_id: str) -> Dict:
        metadata = self.metadata_service.get(artifact_id)
        reference_images = self.get_reference_images(artifact_id)
        training_pairs = self.get_training_pairs(artifact_id)

        return {
            "artifact_id": artifact_id,
            "metadata": metadata,
            "reference_images": reference_images,
            "num_reference_images": len(reference_images),
            "training_pairs": training_pairs,
            "status": "ready_for_reconstruction_model" if metadata and reference_images else "incomplete",
        }