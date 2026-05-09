import os

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.deps import (
    get_metadata_service,
    get_recognition_service,
    get_restoration_service,
    get_audio_service,
    get_hieroglyph_service,
)
from app.schemas import (
    RecognitionResponse,
    KnowledgeResponse,
    RestorationResponse,
    ScanResponse,
    HieroglyphResponse,
    AnalyzeArtifactResponse,
)
from app.services.image_utils import read_image_bytes, resize_keep_ratio
from app.services.metadata_service import MetadataService
from app.services.recognition_service import RecognitionService
from app.services.restoration_service import RestorationService
from app.services.audio_service import AudioService
from app.services.hieroglyph_service import HieroglyphService
from app.services.analyze_service import AnalyzeService


os.makedirs(settings.TMP_DIR, exist_ok=True)
os.makedirs(settings.RESTORATION_RESULTS_DIR, exist_ok=True)

app = FastAPI(title=settings.APP_NAME)
analyze_service = AnalyzeService()

app.mount(
    "/static/restoration",
    StaticFiles(directory=settings.RESTORATION_RESULTS_DIR),
    name="restoration_static",
)


@app.get("/")
def root():
    return {"message": f"{settings.APP_NAME} is running."}


@app.get("/health")
def health():
    return {"status": "ok"}


# OLD ROUTES - kept
@app.post("/recognize", response_model=RecognitionResponse)
async def recognize_artifact(
    file: UploadFile = File(...),
    recognition_service: RecognitionService = Depends(get_recognition_service),
):
    try:
        file_bytes = await file.read()
        image_bgr = read_image_bytes(file_bytes)
        image_bgr = resize_keep_ratio(image_bgr, 1024)
        result = recognition_service.recognize(image_bgr)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/knowledge/{artifact_id}", response_model=KnowledgeResponse)
def get_knowledge(
    artifact_id: str,
    metadata_service: MetadataService = Depends(get_metadata_service),
):
    item = metadata_service.get(artifact_id)
    if not item:
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return item


@app.post("/restore", response_model=RestorationResponse)
async def restore_artifact(
    file: UploadFile = File(...),
    restoration_service: RestorationService = Depends(get_restoration_service),
):
    try:
        file_bytes = await file.read()
        image_bgr = read_image_bytes(file_bytes)
        image_bgr = resize_keep_ratio(image_bgr, 1024)
        result = restoration_service.restore(image_bgr)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/scan", response_model=ScanResponse)
async def scan_artifact(
    file: UploadFile = File(...),
    recognition_service: RecognitionService = Depends(get_recognition_service),
    metadata_service: MetadataService = Depends(get_metadata_service),
    restoration_service: RestorationService = Depends(get_restoration_service),
):
    try:
        file_bytes = await file.read()
        image_bgr = read_image_bytes(file_bytes)
        image_bgr = resize_keep_ratio(image_bgr, 1024)

        recognition = recognition_service.recognize(image_bgr)
        knowledge = None

        artifact_id = recognition.get("artifact_id")
        if artifact_id:
            item = metadata_service.get(artifact_id)
            if item:
                knowledge = item

        restoration = restoration_service.restore(image_bgr)

        return {
            "recognition": recognition,
            "knowledge": knowledge,
            "restoration": restoration,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/audio/status")
def audio_status(audio_service: AudioService = Depends(get_audio_service)):
    return {"message": audio_service.generate_note()}


@app.post("/hieroglyphs/demo", response_model=HieroglyphResponse)
def hieroglyph_demo(
    symbols: list[str],
    hieroglyph_service: HieroglyphService = Depends(get_hieroglyph_service),
):
    return hieroglyph_service.translate_known_sequence(symbols)


# NEW ROUTE - added
@app.post("/analyze-artifact", response_model=AnalyzeArtifactResponse)
async def analyze_artifact(
    request: Request,
    file: UploadFile = File(...),
):
    try:
        file_bytes = await file.read()
        image_bgr = read_image_bytes(file_bytes)
        image_bgr = resize_keep_ratio(image_bgr, 1024)

        result = analyze_service.analyze_image(
            image_bgr=image_bgr,
            scanned_filename=file.filename or "uploaded_image.jpg",
            base_url=str(request.base_url),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))