import json
import os
from app.config import settings


def main():
    if not os.path.exists(settings.ARTIFACT_METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found: {settings.ARTIFACT_METADATA_PATH}")

    with open(settings.ARTIFACT_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    os.makedirs(settings.ARTIFACT_IMAGES_DIR, exist_ok=True)

    for item in metadata:
        artifact_id = item["artifact_id"]
        artifact_name = item.get("name", "Unknown Artifact")
        artifact_dir = os.path.join(settings.ARTIFACT_IMAGES_DIR, artifact_id)
        os.makedirs(artifact_dir, exist_ok=True)

        info_file = os.path.join(artifact_dir, "_artifact_info.txt")
        if not os.path.exists(info_file):
            with open(info_file, "w", encoding="utf-8") as out:
                out.write(f"Artifact ID: {artifact_id}\n")
                out.write(f"Name: {artifact_name}\n")
                out.write("Put 3 to 10 images of this exact artifact in this folder.\n")
                out.write("Use JPG/PNG/WEBP files only.\n")
                out.write("Suggested names: 1.jpg, 2.jpg, 3.jpg\n")

        print(f"Ready: {artifact_dir}")

    print("\nAll artifact folders are ready.")


if __name__ == "__main__":
    main()