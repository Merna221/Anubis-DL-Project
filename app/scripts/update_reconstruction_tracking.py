import argparse

from app.config import settings
from app.services.local_reconstruction_utils import upsert_tracking_row


def main():
    parser = argparse.ArgumentParser(description="Update reconstruction tracking CSV.")
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--input-ready", default="yes")
    parser.add_argument("--target-ready", default="yes")
    parser.add_argument("--references-ready", default="yes")
    parser.add_argument("--prompt-ready", default="yes")
    parser.add_argument("--cpu-prep-done", default="yes")
    parser.add_argument("--gpu-run-done", default="no")
    parser.add_argument("--best-result-path", default="")
    parser.add_argument("--bundle-path", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    row = {
        "artifact_id": args.artifact_id,
        "artifact_name": args.artifact_name,
        "input_ready": args.input_ready,
        "target_ready": args.target_ready,
        "references_ready": args.references_ready,
        "prompt_ready": args.prompt_ready,
        "cpu_prep_done": args.cpu_prep_done,
        "gpu_run_done": args.gpu_run_done,
        "best_result_path": args.best_result_path,
        "bundle_path": args.bundle_path,
        "notes": args.notes,
    }

    upsert_tracking_row(settings.RECONSTRUCTION_TRACKING_CSV, row)

    print("Tracking updated:", settings.RECONSTRUCTION_TRACKING_CSV)
    print("Artifact:", args.artifact_id)


if __name__ == "__main__":
    main()