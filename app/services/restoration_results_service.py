from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app.config import settings


class RestorationResultsService:
    def __init__(self, results_dir: Path | str | None = None) -> None:
        self.results_dir = Path(results_dir or settings.RESTORATION_RESULTS_DIR)

    def get_artifact_result_dir(self, artifact_id: str) -> Path:
        return self.results_dir / artifact_id

    def _find_first_existing_file(
        self,
        artifact_dir: Path,
        candidate_names: list[str],
    ) -> Optional[Path]:
        for name in candidate_names:
            path = artifact_dir / name
            if path.exists():
                return path
        return None

    def get_final_image_path(self, artifact_id: Optional[str]) -> Optional[Path]:
        if not artifact_id:
            return None

        artifact_dir = self.get_artifact_result_dir(artifact_id)

        # First: look directly inside the artifact folder
        direct_match = self._find_first_existing_file(
            artifact_dir,
            [
                "final_result.png",
                f"final_{artifact_id}_result_stage3.png",
                f"{artifact_id}_final_result.png",
                "final_artifact_result.png",
            ],
        )
        if direct_match:
            return direct_match

        # Second: look inside full_workspace recursively
        workspace = artifact_dir / "full_workspace"
        if workspace.exists():
            png_candidates = list(workspace.rglob("final*.png"))
            stage3_candidates = list(workspace.rglob("*stage3*.png"))
            all_candidates = png_candidates + stage3_candidates

            if all_candidates:
                return all_candidates[0]

        return None

    def get_bundle_path(self, artifact_id: Optional[str]) -> Optional[Path]:
        if not artifact_id:
            return None

        artifact_dir = self.get_artifact_result_dir(artifact_id)
        return self._find_first_existing_file(
            artifact_dir,
            [
                f"{artifact_id}_bundle_final.json",
                f"{artifact_id}_bundle_initial.json",
                "artifact_bundle.json",
            ],
        )

    def get_workspace_path(self, artifact_id: Optional[str]) -> Optional[Path]:
        if not artifact_id:
            return None

        artifact_dir = self.get_artifact_result_dir(artifact_id)
        workspace = artifact_dir / "full_workspace"
        return workspace if workspace.exists() else None

    def load_bundle(self, artifact_id: Optional[str]) -> Optional[dict]:
        bundle_path = self.get_bundle_path(artifact_id)
        if not bundle_path:
            return None

        with open(bundle_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_artifact_name(self, artifact_id: Optional[str]) -> Optional[str]:
        bundle = self.load_bundle(artifact_id)
        if not bundle:
            return None
        return bundle.get("artifact_name")

    def has_result(self, artifact_id: Optional[str]) -> bool:
        return self.get_final_image_path(artifact_id) is not None

    def get_result_info(self, artifact_id: Optional[str]) -> dict:
        if not artifact_id:
            return {
                "artifact_id": None,
                "artifact_name": None,
                "final_image_path": None,
                "bundle_path": None,
                "workspace_path": None,
                "available": False,
            }

        artifact_dir = self.get_artifact_result_dir(artifact_id)

        final_image = self.get_final_image_path(artifact_id)
        bundle = self.get_bundle_path(artifact_id)
        workspace = self.get_workspace_path(artifact_id)

        return {
            "artifact_id": artifact_id,
            "artifact_name": self.get_artifact_name(artifact_id),
            "final_image_path": str(final_image) if final_image else None,
            "bundle_path": str(bundle) if bundle else None,
            "workspace_path": str(workspace) if workspace else None,
            "available": artifact_dir.exists() and final_image is not None,
        }