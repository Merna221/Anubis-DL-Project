import os
from app.config import settings


RECONSTRUCTION_ROOT = os.path.join(settings.DATA_DIR, "reconstruction_pairs")


def main():
    print("=" * 70)
    print("RECONSTRUCTION PAIRS CHECK")
    print("=" * 70)

    if not os.path.exists(RECONSTRUCTION_ROOT):
        print(f"Folder not found: {RECONSTRUCTION_ROOT}")
        return

    artifact_folders = sorted(
        f for f in os.listdir(RECONSTRUCTION_ROOT)
        if os.path.isdir(os.path.join(RECONSTRUCTION_ROOT, f))
    )

    if not artifact_folders:
        print("No reconstruction pair folders found.")
        return

    total_inputs = 0
    total_targets = 0

    for artifact_id in artifact_folders:
        artifact_dir = os.path.join(RECONSTRUCTION_ROOT, artifact_id)
        input_dir = os.path.join(artifact_dir, "input")
        target_dir = os.path.join(artifact_dir, "target")

        input_files = []
        target_files = []

        if os.path.isdir(input_dir):
            input_files = [
                f for f in os.listdir(input_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]

        if os.path.isdir(target_dir):
            target_files = [
                f for f in os.listdir(target_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]

        total_inputs += len(input_files)
        total_targets += len(target_files)

        print(f"{artifact_id}")
        print(f"  input images : {len(input_files)}")
        print(f"  target images: {len(target_files)}")

        if len(input_files) != len(target_files):
            print("  WARNING: input/target counts do not match")
        print("-" * 50)

    print(f"Total input images : {total_inputs}")
    print(f"Total target images: {total_targets}")
    print("=" * 70)


if __name__ == "__main__":
    main()