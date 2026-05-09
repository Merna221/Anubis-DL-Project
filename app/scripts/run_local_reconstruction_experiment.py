import argparse
import gc
import os

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

from app.config import settings
from app.services.local_reconstruction_utils import (
    compute_ssim_score,
    load_json,
    save_json,
)


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_strengths(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def choose_input_path(bundle: dict, input_variant: str) -> str:
    if input_variant == "full":
        path = bundle.get("full_preprocessed_input_variant")
        if not path:
            raise ValueError("Full input variant is not available in bundle.")
        return path

    if input_variant == "cropped":
        path = bundle.get("preprocessed_input")
        if not path:
            raise ValueError("Cropped input variant is not available in bundle.")
        return path

    raise ValueError("input_variant must be either 'full' or 'cropped'.")


def main():
    parser = argparse.ArgumentParser(description="Run local reconstruction experiment from a saved bundle.")
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument(
        "--input-variant",
        choices=["full", "cropped"],
        default=None,
        help="Choose which prepared input to use.",
    )
    parser.add_argument(
        "--strengths",
        default=settings.RECONSTRUCTION_DEFAULT_STRENGTHS,
        help="Comma separated strengths, e.g. 0.24,0.30,0.36,0.42",
    )
    parser.add_argument("--guidance-scale", type=float, default=settings.RECONSTRUCTION_DEFAULT_GUIDANCE)
    parser.add_argument("--num-inference-steps", type=int, default=settings.RECONSTRUCTION_DEFAULT_STEPS)
    parser.add_argument("--base-seed", type=int, default=200)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA GPU is not available on this machine.")
        print("Preparation can continue on CPU, but generation should be run only on a GPU machine.")
        return

    artifact_dir = os.path.join(settings.RECONSTRUCTION_EXPERIMENTS_DIR, args.artifact_id)
    bundle_path = os.path.join(artifact_dir, "bundle", "bundle.json")

    bundle = load_json(bundle_path)

    input_variant = args.input_variant or bundle.get("primary_input_variant", "full")
    input_path = choose_input_path(bundle, input_variant)
    target_path = bundle["clean_target"]
    prompt = bundle["prompt"]
    negative_prompt = bundle["negative_prompt"]

    generated_dir = os.path.join(artifact_dir, "generated", input_variant)
    final_dir = os.path.join(artifact_dir, "final", input_variant)

    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    clear_memory()

    device = "cuda"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        settings.RECONSTRUCTION_MODEL_ID,
        torch_dtype=torch.float16,
    ).to(device)

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    input_img = Image.open(input_path).convert("RGB").resize(
        (settings.RECONSTRUCTION_IMAGE_SIZE, settings.RECONSTRUCTION_IMAGE_SIZE)
    )

    strengths = parse_strengths(args.strengths)
    generated_paths = []
    scores = {}

    for i, strength in enumerate(strengths, start=1):
        clear_memory()

        generator = torch.Generator(device=device).manual_seed(args.base_seed + i)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_img,
            strength=strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images[0]

        out_path = os.path.join(generated_dir, f"candidate_{i:02d}.png")
        result.save(out_path)
        generated_paths.append(out_path)

        score = compute_ssim_score(out_path, target_path)
        scores[out_path] = score

        print(f"[{input_variant}] Saved: {out_path} | SSIM: {score:.4f}")

    best_path = max(scores, key=scores.get)
    best_score = scores[best_path]

    final_result_path = os.path.join(final_dir, "final_result.png")
    Image.open(best_path).convert("RGB").save(final_result_path)

    bundle.setdefault("runs", {})
    bundle["runs"][input_variant] = {
        "input_variant": input_variant,
        "input_path_used": input_path,
        "generated_candidates": generated_paths,
        "scores": scores,
        "best_candidate": best_path,
        "best_score": best_score,
        "final_result": final_result_path,
        "run_settings": {
            "strengths": strengths,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "base_seed": args.base_seed,
            "model_id": settings.RECONSTRUCTION_MODEL_ID,
        },
    }

    save_json(bundle, bundle_path)

    print("=" * 70)
    print("RECONSTRUCTION RUN COMPLETE")
    print("=" * 70)
    print("Artifact:", args.artifact_id)
    print("Input variant:", input_variant)
    print("Best candidate:", best_path)
    print("Best score:", best_score)
    print("Final result:", final_result_path)
    print("Bundle updated:", bundle_path)
    print("=" * 70)


if __name__ == "__main__":
    main()