from typing import Dict, Tuple, Optional

import cv2
import numpy as np

from app.services.enhancement_service import EnhancementService
from app.services.image_utils import ensure_uint8, np_image_to_base64_jpeg
from app.services.restoration_results_service import RestorationResultsService


class RestorationService:
    def __init__(self) -> None:
        self.lama_available = False
        self.realesrgan_available = False
        self.enhancement_service = EnhancementService()
        self.results_service = RestorationResultsService()

    def _build_damage_mask(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, blackhat_kernel)

        _, mask = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)

        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        mask = cv2.dilate(mask, kernel_small, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean_mask = np.zeros_like(mask)

        image_area = gray.shape[0] * gray.shape[1]
        min_area = 10
        max_area = int(image_area * 0.003)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                clean_mask[labels == i] = 255

        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=1)

        damage_ratio = float(np.sum(clean_mask > 0) / clean_mask.size)
        return clean_mask, damage_ratio

    def _adaptive_inpaint(self, image_bgr: np.ndarray, mask: np.ndarray, damage_ratio: float) -> np.ndarray:
        if damage_ratio < 0.001:
            radius = 2
        elif damage_ratio < 0.01:
            radius = 3
        else:
            radius = 4

        return cv2.inpaint(image_bgr, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)

    def _final_polish(self, image_bgr: np.ndarray) -> np.ndarray:
        bilateral = cv2.bilateralFilter(image_bgr, d=5, sigmaColor=30, sigmaSpace=30)
        blended = cv2.addWeighted(image_bgr, 0.85, bilateral, 0.15, 0)
        return ensure_uint8(blended)

    # Old functionality remains exactly here
    def restore(self, image_bgr: np.ndarray, use_super_resolution: bool = False) -> Dict:
        enhanced = self.enhancement_service.enhance(image_bgr)
        mask, damage_ratio = self._build_damage_mask(image_bgr)

        if damage_ratio > 0.0005:
            restored = self._adaptive_inpaint(enhanced, mask, damage_ratio)
        else:
            restored = enhanced

        restored = self._final_polish(restored)
        result_base64 = np_image_to_base64_jpeg(restored)

        return {
            "message": "Restoration completed.",
            "used_lama": False,
            "used_super_resolution": False,
            "damage_ratio": round(damage_ratio, 6),
            "output_image_base64": result_base64,
        }

    # Old functionality remains exactly here
    def debug_restore(self, image_bgr: np.ndarray) -> Dict:
        enhanced = self.enhancement_service.enhance(image_bgr)
        mask, damage_ratio = self._build_damage_mask(image_bgr)

        if damage_ratio > 0.0005:
            restored = self._adaptive_inpaint(enhanced, mask, damage_ratio)
        else:
            restored = enhanced

        restored = self._final_polish(restored)

        return {
            "original": ensure_uint8(image_bgr),
            "enhanced": ensure_uint8(enhanced),
            "mask": ensure_uint8(mask),
            "restored": ensure_uint8(restored),
            "damage_ratio": round(damage_ratio, 6),
        }

    # New functionality
    def get_saved_restoration(self, artifact_id: Optional[str]) -> Dict:
        return self.results_service.get_result_info(artifact_id)