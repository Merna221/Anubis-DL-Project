import os
from typing import Optional

from app.config import settings


class AudioService:
    def __init__(self) -> None:
        self.api_key = settings.OPENAI_API_KEY

    def available(self) -> bool:
        return bool(self.api_key)

    def generate_note(self) -> str:
        if not self.available():
            return "OpenAI API key not configured. Audio generation is disabled."
        return "Audio service key is configured. Connect this to your frontend or a save-to-file endpoint."