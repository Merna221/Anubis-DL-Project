import csv
import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(data: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_image(input_path: str, output_path: str, size: tuple[int, int] = (512, 512), keep_as_png: bool = False) -> str:
    img = Image.open(input_path).convert("RGB")
    img = img.resize(size)

    ensure_dir(os.path.dirname(output_path))

    if keep_as_png:
        if not output_path.lower().endswith(".png"):
            output_path += ".png"
        img.save(output_path, format="PNG")
    else:
        if not output_path.lower().endswith((".jpg", ".jpeg")):
            output_path += ".jpg"
        img.save(output_path, quality=95)

    return output_path


def basic_preprocess(image_path: str, output_path: Optional[str] = None) -> str:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l = clahe.apply(l)

    img_bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, 2, 2, 7, 21)

    blur = cv2.GaussianBlur(img_bgr, (0, 0), 1.0)
    img_bgr = cv2.addWeighted(img_bgr, 1.12, blur, -0.12, 0)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if output_path is None:
        root, _ = os.path.splitext(image_path)
        output_path = f"{root}_preprocessed.jpg"

    ensure_dir(os.path.dirname(output_path))
    Image.fromarray(img_rgb).save(output_path, quality=95)
    return output_path


def crop_artifact_grabcut(
    image_path: str,
    out_path: str,
    size: int = 512,
    rect_ratio: tuple[float, float, float, float] = (0.12, 0.05, 0.76, 0.88),
    padding_ratio: float = 0.08,
    black_background: bool = True,
) -> str:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img_bgr.shape[:2]

    rx, ry, rw, rh = rect_ratio
    rect = (
        int(w * rx),
        int(h * ry),
        int(w * rw),
        int(h * rh),
    )

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    ys, xs = np.where(fg_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Could not isolate artifact from image.")

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    pad_x = int((x2 - x1) * padding_ratio)
    pad_y = int((y2 - y1) * padding_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop_bgr = img_bgr[y1:y2, x1:x2]
    crop_mask = fg_mask[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    if black_background:
        bg = np.zeros_like(crop_rgb)
    else:
        bg = np.full_like(crop_rgb, 255)

    crop_rgb = np.where(crop_mask[:, :, None] == 1, crop_rgb, bg)

    ch, cw = crop_rgb.shape[:2]
    s = max(ch, cw)
    square = np.zeros((s, s, 3), dtype=np.uint8) if black_background else np.full((s, s, 3), 255, dtype=np.uint8)

    yoff = (s - ch) // 2
    xoff = (s - cw) // 2
    square[yoff:yoff + ch, xoff:xoff + cw] = crop_rgb

    square = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)

    ensure_dir(os.path.dirname(out_path))
    Image.fromarray(square).save(out_path)
    return out_path


def compute_ssim_score(image1_path: str, image2_path: str, size: tuple[int, int] = (512, 512)) -> float:
    img1 = np.array(Image.open(image1_path).convert("RGB").resize(size))
    img2 = np.array(Image.open(image2_path).convert("RGB").resize(size))

    vals: List[float] = []
    for c in range(3):
        vals.append(ssim(img1[:, :, c], img2[:, :, c], data_range=255))
    return float(np.mean(vals))


def upsert_tracking_row(csv_path: str, row: Dict[str, str]) -> None:
    ensure_dir(os.path.dirname(csv_path))

    existing: List[Dict[str, str]] = []
    fieldnames = [
        "artifact_id",
        "artifact_name",
        "input_ready",
        "target_ready",
        "references_ready",
        "prompt_ready",
        "cpu_prep_done",
        "gpu_run_done",
        "best_result_path",
        "bundle_path",
        "notes",
    ]

    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing = list(reader)

    updated = False
    for i, item in enumerate(existing):
        if item.get("artifact_id") == row.get("artifact_id"):
            existing[i] = row
            updated = True
            break

    if not updated:
        existing.append(row)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)