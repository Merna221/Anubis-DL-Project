import json
import os
import sys
import cv2
import base64

from app.services.metadata_service import MetadataService
from app.services.recognition_service import RecognitionService
from app.services.restoration_service import RestorationService


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m app.scripts.test_pipeline_direct <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError("Could not read image.")

    metadata_service = MetadataService()
    recognition_service = RecognitionService(metadata_service)
    restoration_service = RestorationService()

    print("=" * 70)
    print("RUNNING DIRECT AI PIPELINE TEST")
    print("=" * 70)

    recognition = recognition_service.recognize(image_bgr)
    print("\nRECOGNITION RESULT:")
    print(json.dumps(recognition, indent=2, ensure_ascii=False))

    artifact_id = recognition.get("artifact_id")
    if artifact_id:
        knowledge = metadata_service.get(artifact_id)
        print("\nKNOWLEDGE RESULT:")
        print(json.dumps(knowledge, indent=2, ensure_ascii=False))
    else:
        print("\nNo artifact matched.")

    restoration = restoration_service.restore(image_bgr)
    print("\nRESTORATION RESULT:")
    print(json.dumps(
        {k: v for k, v in restoration.items() if k != "output_image_base64"},
        indent=2,
        ensure_ascii=False
    ))

    output_dir = "app/data/tmp"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "direct_restored_output.jpg")

    with open(output_path, "wb") as f:
        f.write(base64.b64decode(restoration["output_image_base64"]))

    print(f"\nSaved restored image to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()