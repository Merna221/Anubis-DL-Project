import argparse
import os

from app.config import settings
from app.services.local_reconstruction_utils import basic_preprocess, prepare_image, load_json, save_json


def main():
    parser = argparse.ArgumentParser(description="Add a full-size non-cropped prepared input variant to an artifact bundle.")
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--input-image", required=True)
    args = parser.parse_args()

    artifact_dir = os.path.join(settings.RECONSTRUCTION_EXPERIMENTS_DIR, args.artifact_id)
    input_dir = os.path.join(artifact_dir, "input")
    bundle_path = os.path.join(artifact_dir, "bundle", "bundle.json")

    os.makedirs(input_dir, exist_ok=True)

    full_resized_input = os.path.join(input_dir, "full_input_512.png")
    full_preprocessed_input = os.path.join(input_dir, "full_preprocessed_input.jpg")

    prepare_image(
        input_path=args.input_image,
        output_path=full_resized_input,
        size=(settings.RECONSTRUCTION_IMAGE_SIZE, settings.RECONSTRUCTION_IMAGE_SIZE),
        keep_as_png=True,
    )

    basic_preprocess(full_resized_input, full_preprocessed_input)

    bundle = load_json(bundle_path)
    bundle["full_input_variant"] = full_resized_input
    bundle["full_preprocessed_input_variant"] = full_preprocessed_input

    save_json(bundle, bundle_path)

    print("=" * 70)
    print("FULL INPUT VARIANT ADDED")
    print("=" * 70)
    print("Full resized input:", full_resized_input)
    print("Full preprocessed input:", full_preprocessed_input)
    print("Bundle updated:", bundle_path)
    print("=" * 70)


if __name__ == "__main__":
    main()