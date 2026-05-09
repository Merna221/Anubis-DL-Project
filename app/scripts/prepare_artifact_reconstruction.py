import argparse
import os

from app.config import settings
from app.services.local_reconstruction_utils import (
    basic_preprocess,
    crop_artifact_grabcut,
    prepare_image,
    save_json,
)


def main():
    parser = argparse.ArgumentParser(description="Prepare one artifact for local reconstruction experiments.")
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--target-image", required=True)
    parser.add_argument("--references", nargs="*", default=[])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    artifact_dir = os.path.join(settings.RECONSTRUCTION_EXPERIMENTS_DIR, args.artifact_id)
    input_dir = os.path.join(artifact_dir, "input")
    target_dir = os.path.join(artifact_dir, "target")
    reference_dir = os.path.join(artifact_dir, "reference")
    bundle_dir = os.path.join(artifact_dir, "bundle")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(reference_dir, exist_ok=True)
    os.makedirs(bundle_dir, exist_ok=True)

    cropped_input = os.path.join(input_dir, "cropped_input.png")
    preprocessed_input = os.path.join(input_dir, "preprocessed_input.jpg")
    clean_target = os.path.join(target_dir, "target_512.png")

    crop_artifact_grabcut(
        image_path=args.input_image,
        out_path=cropped_input,
        size=settings.RECONSTRUCTION_IMAGE_SIZE,
    )

    basic_preprocess(cropped_input, preprocessed_input)

    prepare_image(
        input_path=args.target_image,
        output_path=clean_target,
        size=(settings.RECONSTRUCTION_IMAGE_SIZE, settings.RECONSTRUCTION_IMAGE_SIZE),
        keep_as_png=True,
    )

    clean_references = []
    for i, ref_path in enumerate(args.references, start=1):
        out_ref = os.path.join(reference_dir, f"reference_{i}.jpg")
        prepare_image(
            input_path=ref_path,
            output_path=out_ref,
            size=(settings.RECONSTRUCTION_IMAGE_SIZE, settings.RECONSTRUCTION_IMAGE_SIZE),
            keep_as_png=False,
        )
        clean_references.append(out_ref)

    bundle = {
        "artifact_id": args.artifact_id,
        "artifact_name": args.artifact_name,
        "input_image": args.input_image,
        "target_image": args.target_image,
        "reference_images": args.references,
        "cropped_input": cropped_input,
        "preprocessed_input": preprocessed_input,
        "clean_target": clean_target,
        "clean_references": clean_references,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "notes": args.notes,
        "generated_candidates": [],
        "scores": {},
        "best_candidate": None,
        "best_score": None,
        "final_result": None,
    }

    bundle_path = os.path.join(bundle_dir, "bundle.json")
    save_json(bundle, bundle_path)

    print("=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)
    print("Artifact:", args.artifact_id, "-", args.artifact_name)
    print("Cropped input:", cropped_input)
    print("Preprocessed input:", preprocessed_input)
    print("Clean target:", clean_target)
    print("Clean references:", clean_references)
    print("Bundle:", bundle_path)
    print("=" * 70)


if __name__ == "__main__":
    main()