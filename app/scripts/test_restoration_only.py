import os
import sys
import cv2

from app.services.restoration_service import RestorationService


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m app.scripts.test_restoration_only <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError("Could not read image.")

    service = RestorationService()
    result = service.debug_restore(image_bgr)

    output_dir = "app/data/tmp/restoration_debug"
    os.makedirs(output_dir, exist_ok=True)

    original_path = os.path.join(output_dir, "01_original.jpg")
    enhanced_path = os.path.join(output_dir, "02_enhanced.jpg")
    mask_path = os.path.join(output_dir, "03_mask.jpg")
    restored_path = os.path.join(output_dir, "04_restored.jpg")

    cv2.imwrite(original_path, result["original"])
    cv2.imwrite(enhanced_path, result["enhanced"])
    cv2.imwrite(mask_path, result["mask"])
    cv2.imwrite(restored_path, result["restored"])

    print("=" * 70)
    print("RESTORATION DEBUG COMPLETE")
    print("=" * 70)
    print(f"Damage ratio: {result['damage_ratio']}")
    print(f"Saved: {original_path}")
    print(f"Saved: {enhanced_path}")
    print(f"Saved: {mask_path}")
    print(f"Saved: {restored_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()