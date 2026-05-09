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


def main():
    reconstruction_root = os.path.join(settings.DATA_DIR, "reconstruction_pairs")

    if not os.path.isdir(reconstruction_root):
        print(f"No reconstruction_pairs folder found at: {reconstruction_root}")
        return

    all_results = []
    all_true = []
    all_pred = []

    for artifact_id in sorted(os.listdir(reconstruction_root)):
        artifact_dir = os.path.join(reconstruction_root, artifact_id)
        input_dir = os.path.join(artifact_dir, "input")
        target_dir = os.path.join(artifact_dir, "target")

        if not os.path.isdir(input_dir) or not os.path.isdir(target_dir):
            continue

        input_files = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ])

        target_files = sorted([
            f for f in os.listdir(target_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ])

        pair_count = min(len(input_files), len(target_files))

        for i in range(pair_count):
            input_path = os.path.join(input_dir, input_files[i])
            target_path = os.path.join(target_dir, target_files[i])

            input_img = load_image(input_path)
            target_img = load_image(target_path)

            y_pred = input_img.flatten()
            y_true = target_img.flatten()

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            mape_value = safe_mape(y_true, y_pred)
            psnr_value = calculate_psnr(y_true, y_pred)
            ssim_value = calculate_ssim(target_img, input_img)

            all_true.append(y_true)
            all_pred.append(y_pred)

            all_results.append({
                "artifact_id": artifact_id,
                "input_image": input_path,
                "target_image": target_path,
                "mae": float(mae),
                "safe_mape": float(mape_value),
                "rmse": float(rmse),
                "r2": float(r2),
                "psnr": float(psnr_value),
                "ssim": float(ssim_value)
            })

    if not all_results:
        print("No restoration pairs found.")
        return

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    summary = {
        "total_pairs": len(all_results),
        "average_mae": float(mean_absolute_error(all_true, all_pred)),
        "average_safe_mape": float(safe_mape(all_true, all_pred)),
        "average_rmse": float(np.sqrt(mean_squared_error(all_true, all_pred))),
        "average_r2": float(r2_score(all_true, all_pred)),
        "average_psnr": float(np.mean([r["psnr"] for r in all_results])),
        "average_ssim": float(np.mean([r["ssim"] for r in all_results])),
        "details": all_results
    }

    with open(os.path.join(RESULTS_DIR, "restoration_regression_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("Restoration Regression Metrics")
    print("=" * 70)
    print(f"Total pairs: {summary['total_pairs']}")
    print(f"MAE:       {summary['average_mae']:.6f}")
    print(f"Safe MAPE: {summary['average_safe_mape']:.6f}")
    print(f"RMSE:      {summary['average_rmse']:.6f}")
    print(f"R2:        {summary['average_r2']:.6f}")
    print(f"PSNR:      {summary['average_psnr']:.6f}")
    print(f"SSIM:      {summary['average_ssim']:.6f}")
    print(f"Saved to: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()