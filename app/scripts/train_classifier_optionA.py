import os
import json
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from codecarbon import EmissionsTracker
from transformers import AutoImageProcessor, AutoModel

from app.config import settings


RESULTS_DIR = "results/classification"
CARBON_DIR = "results/carbon"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CARBON_DIR, exist_ok=True)


class ArtifactClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


def load_image_paths():
    image_paths = []
    labels = []

    for artifact_id in sorted(os.listdir(settings.ARTIFACT_IMAGES_DIR)):
        folder = os.path.join(settings.ARTIFACT_IMAGES_DIR, artifact_id)

        if not os.path.isdir(folder):
            continue

        for file_name in os.listdir(folder):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_paths.append(os.path.join(folder, file_name))
                labels.append(artifact_id)

    return image_paths, labels


@torch.no_grad()
def extract_dino_embeddings(image_paths, device):
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
    dino_model.eval()

    embeddings = []

    for path in tqdm(image_paths, desc="Extracting DINOv2 embeddings"):
        image_bgr = cv2.imread(path)

        if image_bgr is None:
            raise ValueError(f"Could not read image: {path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        outputs = dino_model(**inputs)

        emb = outputs.last_hidden_state[:, 0, :].squeeze(0)
        embeddings.append(emb.cpu().numpy())

    return np.array(embeddings, dtype=np.float32)


def plot_learning_curves(history):
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_curve.png"), dpi=300)
    plt.close()


def calculate_accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    image_paths, labels = load_image_paths()

    if len(image_paths) == 0:
        raise RuntimeError("No images found. Check app/data/artifacts/images")

    print(f"Total images: {len(image_paths)}")
    print(f"Total classes: {len(set(labels))}")

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    embeddings = extract_dino_embeddings(image_paths, device)

    X_train, X_val, y_train, y_val = train_test_split(
        embeddings,
        encoded_labels,
        test_size=0.2,
        random_state=42,
        stratify=encoded_labels
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    input_dim = embeddings.shape[1]
    num_classes = len(label_encoder.classes_)

    model = ArtifactClassifier(input_dim=input_dim, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    tracker = EmissionsTracker(
        output_dir=CARBON_DIR,
        output_file="training_emissions.csv",
        project_name="ANUBIS_DL_Classifier"
    )

    tracker.start()
    start_time = time.time()

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        train_outputs = model(X_train)
        train_loss = criterion(train_outputs, y_train)
        train_loss.backward()
        optimizer.step()

        train_acc = calculate_accuracy(train_outputs, y_train)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_acc = calculate_accuracy(val_outputs, y_val)

        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss.item())
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss.item():.4f} "
            f"Val Loss: {val_loss.item():.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

    emissions = tracker.stop()
    training_time = time.time() - start_time

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "artifact_classifier.pt"))

    np.save(os.path.join(RESULTS_DIR, "embeddings.npy"), embeddings)
    np.save(os.path.join(RESULTS_DIR, "labels.npy"), encoded_labels)

    with open(os.path.join(RESULTS_DIR, "label_classes.json"), "w", encoding="utf-8") as f:
        json.dump(label_encoder.classes_.tolist(), f, indent=2)

    with open(os.path.join(RESULTS_DIR, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "total_images": len(image_paths),
        "num_classes": num_classes,
        "input_dim": input_dim,
        "epochs": epochs,
        "final_train_accuracy": history["train_acc"][-1],
        "final_validation_accuracy": history["val_acc"][-1],
        "final_train_loss": history["train_loss"][-1],
        "final_validation_loss": history["val_loss"][-1],
        "training_time_seconds": training_time,
        "carbon_emissions_kg": emissions
    }

    with open(os.path.join(RESULTS_DIR, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_learning_curves(history)

    print("=" * 70)
    print("Training completed.")
    print(f"Model saved to: {os.path.join(RESULTS_DIR, 'artifact_classifier.pt')}")
    print(f"Learning curves saved to: {RESULTS_DIR}")
    print(f"Carbon report saved to: {CARBON_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()