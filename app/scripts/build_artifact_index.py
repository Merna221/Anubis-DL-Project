import json
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from app.config import settings


def load_images_grouped(metadata_path: str, images_root: str):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    grouped = defaultdict(list)
    valid_ids = set()

    for item in metadata:
        artifact_id = item["artifact_id"]
        valid_ids.add(artifact_id)
        folder = os.path.join(images_root, artifact_id)
        if not os.path.isdir(folder):
            continue

        for file_name in os.listdir(folder):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                grouped[artifact_id].append(os.path.join(folder, file_name))

    return grouped, valid_ids


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
    model.eval()

    grouped, valid_ids = load_images_grouped(
        settings.ARTIFACT_METADATA_PATH,
        settings.ARTIFACT_IMAGES_DIR,
    )

    artifact_ids = []
    artifact_embeddings = []

    for artifact_id, paths in grouped.items():
        embeddings = []
        for path in paths:
            image_bgr = cv2.imread(path)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            inputs = processor(images=pil_image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy().astype(np.float32)
            embeddings.append(emb)

        if embeddings:
            mean_embedding = np.mean(np.stack(embeddings, axis=0), axis=0)
            artifact_ids.append(artifact_id)
            artifact_embeddings.append(mean_embedding)

    if not artifact_embeddings:
        raise RuntimeError("No embeddings were created. Check your dataset folders and metadata.")

    embeddings_array = np.stack(artifact_embeddings, axis=0).astype(np.float32)

    np.save(settings.ARTIFACT_EMBEDDINGS_PATH, embeddings_array)
    with open(settings.ARTIFACT_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact_ids, f, ensure_ascii=False, indent=2)

    print(f"Saved embeddings: {settings.ARTIFACT_EMBEDDINGS_PATH}")
    print(f"Saved artifact IDs: {settings.ARTIFACT_IDS_PATH}")
    print(f"Artifacts indexed: {len(artifact_ids)}")


if __name__ == "__main__":
    main()