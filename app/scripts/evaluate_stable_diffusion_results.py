import os
import json
import cv2
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skimage.metrics import structural_similarity as ssim

from app.config import settings


RESULTS_DIR = "results/restoration"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_image(path, size=(512, 512)):
    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    image = cv2.resize(image, size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    return image


def safe_mape(y_true, y_pred):
    mask = y_true > 0.05
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_psnr(y_true, y_pred):
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    if mse == 0:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


def calculate_ssim(y_true_img, y_pred_img):
    return ssim(
        y_true_img,
        y_pred_img,
        channel_axis=2,
        data_range=1.0
    )


def get_images(folder):
    if not os.path.isdir(folder):
        return []

    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])


def main():
    reconstruction_root = os.path.join(settings.DATA_DIR, "reconstruction_pairs")

    if not os.path.isdir(reconstruction_root):
        print(f"No reconstruction_pairs folder found at: {reconstruction_root}")
        return

    details = []
    all_true = []
    all_pred = []

    for artifact_id in sorted(os.listdir(reconstruction_root)):
        artifact_dir = os.path.join(reconstruction_root, artifact_id)

        target_dir = os.path.join(artifact_dir, "target")
        sd_dir = os.path.join(artifact_dir, "stable_diffusion_output")

        target_images = get_images(target_dir)
        sd_images = get_images(sd_dir)

        pair_count = min(len(target_images), len(sd_images))

        if pair_count == 0:
            continue

        for i in range(pair_count):
            target_path = target_images[i]
            sd_path = sd_images[i]

            target_img = load_image(target_path)
            sd_img = load_image(sd_path)

            y_true = target_img.flatten()
            y_pred = sd_img.flatten()

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape_value = safe_mape(y_true, y_pred)
            psnr_value = calculate_psnr(y_true, y_pred)
            ssim_value = calculate_ssim(target_img, sd_img)

            all_true.append(y_true)
            all_pred.append(y_pred)

            details.append({
                "artifact_id": artifact_id,
                "target_image": target_path,
                "stable_diffusion_output": sd_path,
                "mae": float(mae),
                "safe_mape": float(mape_value),
                "rmse": float(rmse),
                "r2": float(r2),
                "psnr": float(psnr_value),
                "ssim": float(ssim_value)
            })

    if not details:
        print("No Stable Diffusion outputs found.")
        print("Make sure each artifact has a stable_diffusion_output folder.")
        return

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    summary = {
        "model": "Stable Diffusion Img2Img",
        "total_pairs": len(details),
        "average_mae": float(mean_absolute_error(all_true, all_pred)),
        "average_safe_mape": float(safe_mape(all_true, all_pred)),
        "average_rmse": float(np.sqrt(mean_squared_error(all_true, all_pred))),
        "average_r2": float(r2_score(all_true, all_pred)),
        "average_psnr": float(np.mean([r["psnr"] for r in details])),
        "average_ssim": float(np.mean([r["ssim"] for r in details])),
        "details": details
    }

    output_path = os.path.join(RESULTS_DIR, "stable_diffusion_metrics.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("Stable Diffusion Restoration Metrics")
    print("=" * 70)
    print(f"Total pairs: {summary['total_pairs']}")
    print(f"MAE:       {summary['average_mae']:.6f}")
    print(f"Safe MAPE: {summary['average_safe_mape']:.6f}")
    print(f"RMSE:      {summary['average_rmse']:.6f}")
    print(f"R2:        {summary['average_r2']:.6f}")
    print(f"PSNR:      {summary['average_psnr']:.6f}")
    print(f"SSIM:      {summary['average_ssim']:.6f}")
    print(f"Saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()