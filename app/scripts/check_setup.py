import json
import os
from collections import Counter

from app.config import settings


def main():
    print("=" * 70)
    print("ANUBIS AI SETUP CHECK")
    print("=" * 70)

    required_paths = {
        "ARTIFACT_METADATA_PATH": settings.ARTIFACT_METADATA_PATH,
        "ARTIFACT_IMAGES_DIR": settings.ARTIFACT_IMAGES_DIR,
    }

    for name, path in required_paths.items():
        exists = os.path.exists(path)
        print(f"{name}: {path} -> {'OK' if exists else 'MISSING'}")

    if not os.path.exists(settings.ARTIFACT_METADATA_PATH):
        print("\nmetadata.json is missing.")
        return

    with open(settings.ARTIFACT_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"\nArtifacts in metadata: {len(metadata)}")

    artifact_ids = [item.get("artifact_id") for item in metadata]
    duplicates = [k for k, v in Counter(artifact_ids).items() if v > 1]
    if duplicates:
        print(f"Duplicate artifact IDs found: {duplicates}")
    else:
        print("Duplicate artifact IDs: none")

    required_fields = [
        "artifact_id",
        "name",
        "artifact_type",
        "museum",
        "era",
        "dynasty",
        "material",
        "location",
        "description_en",
        "description_ar",
        "story_en",
        "story_ar",
        "audio_script_en",
        "audio_script_ar",
    ]

    missing_report = []
    total_images = 0

    for item in metadata:
        artifact_id = item["artifact_id"]
        missing_fields = [field for field in required_fields if field not in item or item[field] in [None, ""]]
        if missing_fields:
            missing_report.append((artifact_id, missing_fields))

        image_dir = os.path.join(settings.ARTIFACT_IMAGES_DIR, artifact_id)
        if not os.path.isdir(image_dir):
            print(f"Image folder missing for {artifact_id}: {image_dir}")
            continue

        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        total_images += len(image_files)
        print(f"{artifact_id}: {len(image_files)} image(s)")

    print(f"\nTotal dataset images: {total_images}")

    if missing_report:
        print("\nArtifacts with missing fields:")
        for artifact_id, fields in missing_report:
            print(f" - {artifact_id}: {fields}")
    else:
        print("\nAll metadata required fields are present.")

    print("\nSuggested minimum before testing recognition seriously:")
    print("- At least 3 images per artifact")
    print("- At least 30 total images overall")
    print("=" * 70)


if __name__ == "__main__":
    main()