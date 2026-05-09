import json
import os
from typing import Dict, List

from app.config import settings


class HieroglyphService:
    def __init__(self) -> None:
        self.sign_map_path = os.path.join(settings.DATA_DIR, "hieroglyphs", "sign_map.json")
        self.sign_map = self._load_sign_map()

    def _load_sign_map(self) -> Dict[str, Dict[str, str]]:
        if os.path.exists(self.sign_map_path):
            with open(self.sign_map_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def translate_known_sequence(self, symbols: List[str]) -> dict:
        transliteration_parts = []
        english_parts = []
        arabic_parts = []

        for symbol in symbols:
            entry = self.sign_map.get(symbol, {})
            transliteration_parts.append(entry.get("transliteration", symbol))
            english_parts.append(entry.get("english", symbol))
            arabic_parts.append(entry.get("arabic", symbol))

        return {
            "detected_symbols": symbols,
            "transliteration": " ".join(transliteration_parts),
            "translated_en": " ".join(english_parts),
            "translated_ar": " ".join(arabic_parts),
            "note": (
                "This is a starter symbolic mapping pipeline. "
                "For full wall/photo hieroglyph reading, you still need a trained sign detector/classifier."
            ),
        }