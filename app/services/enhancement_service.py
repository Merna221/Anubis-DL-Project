import cv2
import numpy as np

from app.services.image_utils import ensure_uint8


class EnhancementService:
    """
    Reusable baseline enhancement service.

    This is the cleaned reusable version of the old enhancement pipeline:
    - color correction
    - denoising
    - sharpening
    - contrast enhancement

    It is NOT the final reconstruction model.
    It is the preprocessing / cleanup stage.
    """

    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        # Step 1: color correction in LAB
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_corrected = clahe.apply(l)

        color_corrected = cv2.merge((l_corrected, a, b))
        color_corrected = cv2.cvtColor(color_corrected, cv2.COLOR_LAB2BGR)

        # Step 2: denoising while preserving details
        denoised = cv2.fastNlMeansDenoisingColored(
            color_corrected,
            None,
            h=3,
            hColor=3,
            templateWindowSize=7,
            searchWindowSize=21,
        )

        # Step 3: gentle sharpening
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 1.4)
        sharpened = cv2.addWeighted(denoised, 1.2, gaussian, -0.2, 0)

        # Step 4: final adaptive contrast enhancement
        lab2 = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l2, a2, b2 = cv2.split(lab2)

        clahe2 = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
        l2_enhanced = clahe2.apply(l2)

        final = cv2.merge((l2_enhanced, a2, b2))
        final = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)

        return ensure_uint8(final)