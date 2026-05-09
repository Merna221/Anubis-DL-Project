import base64
import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def read_image_bytes(file_bytes: bytes) -> np.ndarray:
    file_array = np.frombuffer(file_bytes, np.uint8)
    image_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode image.")
    return image_bgr


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def resize_keep_ratio(image: np.ndarray, max_side: int = 1024) -> np.ndarray:
    h, w = image.shape[:2]
    largest = max(h, w)
    if largest <= max_side:
        return image
    scale = max_side / float(largest)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def np_image_to_base64_jpeg(image_bgr: np.ndarray, quality: int = 90) -> str:
    success, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Failed to encode image.")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def np_to_pil_rgb(image_bgr: np.ndarray) -> Image.Image:
    rgb = bgr_to_rgb(image_bgr)
    return Image.fromarray(rgb)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    return exp_scores / (np.sum(exp_scores) + 1e-8)


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)