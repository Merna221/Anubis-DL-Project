from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "Anubis AI Service"
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True

    DATA_DIR: str = "app/data"
    ARTIFACTS_DIR: str = "app/data/artifacts"
    ARTIFACT_IMAGES_DIR: str = "app/data/artifacts/images"
    ARTIFACT_METADATA_PATH: str = "app/data/artifacts/metadata.json"
    ARTIFACT_EMBEDDINGS_PATH: str = "app/data/artifacts/embeddings.npy"
    ARTIFACT_IDS_PATH: str = "app/data/artifacts/artifact_ids.json"

    # Match your project structure
    RECONSTRUCTION_PAIRS_DIR: str = "app/data/artifacts/reconstruction_pairs"
    RECONSTRUCTION_EXPERIMENTS_DIR: str = "app/data/experiments/reconstruction"
    RECONSTRUCTION_TRACKING_CSV: str = "app/data/experiments/reconstruction/tracking.csv"

    RECONSTRUCTION_MODEL_ID: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    RECONSTRUCTION_IMAGE_SIZE: int = 512
    RECONSTRUCTION_DEFAULT_STRENGTHS: str = "0.24,0.30,0.36,0.42"
    RECONSTRUCTION_DEFAULT_GUIDANCE: float = 6.8
    RECONSTRUCTION_DEFAULT_STEPS: int = 28

    OPENAI_API_KEY: str = ""
    OPENAI_TTS_MODEL: str = "gpt-4o-mini-tts"
    OPENAI_TTS_VOICE: str = "alloy"

    HIEROGLYPHS_DIR: str = "app/data/hieroglyphs"
    TMP_DIR: str = "app/data/tmp"
    RESTORATION_RESULTS_DIR: str = "app/data/restoration_results"


settings = Settings()