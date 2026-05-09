import os
import json

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


RESULTS_DIR = "results/classification"
os.makedirs(RESULTS_DIR, exist_ok=True)


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


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = np.load(os.path.join(RESULTS_DIR, "embeddings.npy"))
    labels = np.load(os.path.join(RESULTS_DIR, "labels.npy"))

    with open(os.path.join(RESULTS_DIR, "label_classes.json"), "r", encoding="utf-8") as f:
        class_names = json.load(f)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    input_dim = embeddings.shape[1]
    num_classes = len(class_names)

    model = ArtifactClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "artifact_classifier.pt"), map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(X_test_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        y_pred = np.argmax(probabilities, axis=1)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    try:
        y_test_bin = label_binarize(y_test, classes=list(range(num_classes)))
        roc_auc = roc_auc_score(y_test_bin, probabilities, average="weighted", multi_class="ovr")
    except Exception:
        roc_auc = None

    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0
    )

    metrics = {
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "cohen_kappa": kappa,
        "matthews_correlation_coefficient": mcc,
        "roc_auc_weighted_ovr": roc_auc,
        "test_samples": len(y_test)
    }

    with open(os.path.join(RESULTS_DIR, "classification_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    plot_confusion_matrix(cm, class_names)

    print("=" * 70)
    print("Classification Metrics")
    print("=" * 70)
    print(json.dumps(metrics, indent=2))
    print()
    print(report)
    print(f"Saved metrics to: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()