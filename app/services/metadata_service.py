import json
from typing import Dict, Optional, List, Any

from app.config import settings


class MetadataService:
    def __init__(self) -> None:
        self.metadata_by_id: Dict[str, dict] = {}
        self._raw_data: Any = None
        self._load()

    def _load(self) -> None:
        try:
            with open(settings.ARTIFACT_METADATA_PATH, "r", encoding="utf-8") as f:
                self._raw_data = json.load(f)

            if isinstance(self._raw_data, list):
                self.metadata_by_id = {
                    item["artifact_id"]: item
                    for item in self._raw_data
                    if isinstance(item, dict) and "artifact_id" in item
                }
                return

            if isinstance(self._raw_data, dict):
                if "artifacts" in self._raw_data and isinstance(self._raw_data["artifacts"], list):
                    self.metadata_by_id = {
                        item["artifact_id"]: item
                        for item in self._raw_data["artifacts"]
                        if isinstance(item, dict) and "artifact_id" in item
                    }
                    return

                self.metadata_by_id = {
                    key: value
                    for key, value in self._raw_data.items()
                    if isinstance(value, dict)
                }

                for key, value in self.metadata_by_id.items():
                    value.setdefault("artifact_id", key)
                return

            self.metadata_by_id = {}

        except FileNotFoundError:
            self.metadata_by_id = {}
            self._raw_data = None

    def get(self, artifact_id: str) -> Optional[dict]:
        return self.metadata_by_id.get(artifact_id)

    def get_by_artifact_id(self, artifact_id: str) -> Optional[dict]:
        return self.get(artifact_id)

    def all(self) -> List[dict]:
        return list(self.metadata_by_id.values())