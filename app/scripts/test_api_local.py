import base64
import requests
import sys
from pathlib import Path


BASE_URL = "http://127.0.0.1:8000"


def test_health():
    r = requests.get(f"{BASE_URL}/health", timeout=30)
    print("GET /health")
    print(r.status_code, r.text)
    print("-" * 50)


def test_recognize(image_path: str):
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        r = requests.post(f"{BASE_URL}/recognize", files=files, timeout=120)

    print("POST /recognize")
    print("Status:", r.status_code)
    print(r.json())
    print("-" * 50)


def test_knowledge(artifact_id: str):
    r = requests.get(f"{BASE_URL}/knowledge/{artifact_id}", timeout=30)
    print(f"GET /knowledge/{artifact_id}")
    print("Status:", r.status_code)
    print(r.json())
    print("-" * 50)


def test_restore(image_path: str):
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        r = requests.post(f"{BASE_URL}/restore", files=files, timeout=120)

    print("POST /restore")
    print("Status:", r.status_code)
    data = r.json()
    print({k: v for k, v in data.items() if k != "output_image_base64"})

    base64_data = data.get("output_image_base64")
    if base64_data:
        output_path = "restored_test_output.jpg"
        with open(output_path, "wb") as out:
            out.write(base64.b64decode(base64_data))
        print(f"Saved restored image to: {output_path}")
    print("-" * 50)


def test_scan(image_path: str):
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        r = requests.post(f"{BASE_URL}/scan", files=files, timeout=120)

    print("POST /scan")
    print("Status:", r.status_code)
    data = r.json()

    if "restoration" in data and "output_image_base64" in data["restoration"]:
        base64_data = data["restoration"]["output_image_base64"]
        with open("scan_restored_output.jpg", "wb") as out:
            out.write(base64.b64decode(base64_data))
        data["restoration"]["output_image_base64"] = "[base64 omitted]"
        print("Saved scan restored image to: scan_restored_output.jpg")

    print(data)
    print("-" * 50)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m app.scripts.test_api_local <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    test_health()
    test_recognize(image_path)
    test_restore(image_path)
    test_scan(image_path)
    test_knowledge("artifact_001")