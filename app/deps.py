from functools import lru_cache

from app.services.audio_service import AudioService
from app.services.hieroglyph_service import HieroglyphService
from app.services.metadata_service import MetadataService
from app.services.recognition_service import RecognitionService
from app.services.restoration_service import RestorationService
from app.services.reconstruction_service import ReconstructionService

@lru_cache
def get_metadata_service() -> MetadataService:
    return MetadataService()


@lru_cache
def get_recognition_service() -> RecognitionService:
    return RecognitionService(get_metadata_service())


@lru_cache
def get_restoration_service() -> RestorationService:
    return RestorationService()


@lru_cache
def get_audio_service() -> AudioService:
    return AudioService()


@lru_cache
def get_hieroglyph_service() -> HieroglyphService:
    return HieroglyphService()

@lru_cache
def get_reconstruction_service() -> ReconstructionService:
    return ReconstructionService(get_metadata_service())