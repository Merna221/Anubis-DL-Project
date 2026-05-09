import json
import os
import cv2

from app.config import settings
from app.services.metadata_service import MetadataService
from app.services.recognition_service import RecognitionService


def main():
    metadata_service = MetadataService()
    recognition_service = RecognitionService(metadata_service)

    total = 0
    correct_top1 = 0
    correct_top3 = 0
    results = []

    for artifact_id in os.listdir(settings.ARTIFACT_IMAGES_DIR):
        folder = os.path.join(settings.ARTIFACT_IMAGES_DIR, artifact_id)
        if not os.path.isdir(folder):
            continue

        for file_name in os.listdir(folder):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            image_path = os.path.join(folder, file_name)
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                continue

            result = recognition_service.recognize(image_bgr, top_k=3)
            predicted_top1 = result.get("artifact_id")
            similar = result.get("similar_artifacts", [])
            predicted_top3_ids = [item["artifact_id"] for item in similar]

            is_top1 = predicted_top1 == artifact_id
            is_top3 = artifact_id in predicted_top3_ids

            total += 1
            correct_top1 += int(is_top1)
            correct_top3 += int(is_top3)

            results.append({
                "image": image_path,
                "expected": artifact_id,
                "predicted_top1": predicted_top1,
                "top3_ids": predicted_top3_ids,
                "top1_correct": is_top1,
                "top3_correct": is_top3,
                "confidence": result.get("confidence", 0.0),
            })

            print(f"{image_path}")
            print(f"  expected: {artifact_id}")
            print(f"  top1:     {predicted_top1}")
            print(f"  top3:     {predicted_top3_ids}")
            print(f"  top1 ok:  {is_top1}")
            print(f"  top3 ok:  {is_top3}")
            print("-" * 50)

    if total == 0:
        print("No images found for evaluation.")
        return

    summary = {
        "total_images": total,
        "top1_accuracy": correct_top1 / total,
        "top3_accuracy": correct_top3 / total,
        "details": results,
    }

    output_dir = "app/data/tmp"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "recognition_evaluation.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print(f"Total images: {total}")
    print(f"Top-1 accuracy: {(correct_top1 / total) * 100:.2f}%")
    print(f"Top-3 accuracy: {(correct_top3 / total) * 100:.2f}%")
    print(f"Saved report: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()