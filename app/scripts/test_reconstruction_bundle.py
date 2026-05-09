import json
import sys

from app.deps import get_reconstruction_service


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m app.scripts.test_reconstruction_bundle <artifact_id>")
        sys.exit(1)

    artifact_id = sys.argv[1]
    service = get_reconstruction_service()
    result = service.prepare_reconstruction_bundle(artifact_id)

    print("=" * 70)
    print("RECONSTRUCTION BUNDLE TEST")
    print("=" * 70)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 70)


if __name__ == "__main__":
    main()