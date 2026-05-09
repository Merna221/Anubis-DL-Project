from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SimilarArtifact(BaseModel):
    artifact_id: str
    name: str
    museum: str
    era: str
    confidence: float


class RecognitionResponse(BaseModel):
    artifact_id: Optional[str] = None
    name: Optional[str] = None
    artifact_type: Optional[str] = None
    museum: Optional[str] = None
    era: Optional[str] = None
    dynasty: Optional[str] = None
    material: Optional[str] = None
    confidence: float = 0.0
    similar_artifacts: List[SimilarArtifact] = Field(default_factory=list)
    raw_scores: List[Dict[str, Any]] = Field(default_factory=list)


class KnowledgeResponse(BaseModel):
    artifact_id: str
    name: str
    artifact_type: str
    museum: str
    era: str
    dynasty: str
    material: str
    location: str
    description_en: str
    description_ar: str
    story_en: str
    story_ar: str
    audio_script_en: str
    audio_script_ar: str


class RestorationResponse(BaseModel):
    message: str
    used_lama: bool
    used_super_resolution: bool
    damage_ratio: float
    output_image_base64: str


class ScanResponse(BaseModel):
    recognition: RecognitionResponse
    knowledge: Optional[KnowledgeResponse] = None
    restoration: RestorationResponse


class HieroglyphResponse(BaseModel):
    detected_symbols: List[str]
    transliteration: str
    translated_en: str
    translated_ar: str
    note: str


# New models for the combined analyze-artifact flow
class AnalyzeRecognitionData(BaseModel):
    artifact_id: Optional[str] = None
    name: Optional[str] = None
    artifact_type: Optional[str] = None
    museum: Optional[str] = None
    era: Optional[str] = None
    dynasty: Optional[str] = None
    material: Optional[str] = None
    confidence: float = 0.0
    similar_artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    raw_scores: List[Dict[str, Any]] = Field(default_factory=list)


class AnalyzeMetadataData(BaseModel):
    artifact_id: Optional[str] = None
    name: Optional[str] = None
    artifact_type: Optional[str] = None
    museum: Optional[str] = None
    era: Optional[str] = None
    dynasty: Optional[str] = None
    material: Optional[str] = None
    location: Optional[str] = None
    description_en: Optional[str] = None
    description_ar: Optional[str] = None
    story_en: Optional[str] = None
    story_ar: Optional[str] = None
    audio_script_en: Optional[str] = None
    audio_script_ar: Optional[str] = None


class AnalyzeRestorationData(BaseModel):
    available: bool = False
    artifact_id: Optional[str] = None
    artifact_name: Optional[str] = None
    final_image_path: Optional[str] = None
    final_image_url: Optional[str] = None
    bundle_path: Optional[str] = None
    workspace_path: Optional[str] = None


class AnalyzeArtifactResponse(BaseModel):
    success: bool
    scanned_filename: str
    recognition: AnalyzeRecognitionData
    metadata: AnalyzeMetadataData
    restoration: AnalyzeRestorationData
    debug: Optional[Any] = None