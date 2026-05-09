import os
import json
import torch
import torch.nn as nn
from torchinfo import summary


RESULTS_DIR = "results/flops"
CLASSIFICATION_DIR = "results/classification"
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


def main():
    with open(os.path.join(CLASSIFICATION_DIR, "label_classes.json"), "r", encoding="utf-8") as f:
        class_names = json.load(f)

    input_dim = 384
    num_classes = len(class_names)

    model = ArtifactClassifier(input_dim=input_dim, num_classes=num_classes)

    model.load_state_dict(
        torch.load(
            os.path.join(CLASSIFICATION_DIR, "artifact_classifier.pt"),
            map_location="cpu"
        )
    )

    model.eval()

    model_summary = summary(
        model,
        input_size=(1, input_dim),
        verbose=0
    )

    total_params = model_summary.total_params
    trainable_params = model_summary.trainable_params
    estimated_mult_adds = model_summary.total_mult_adds

    report = {
        "model": "DINOv2 embeddings + Neural Network Classifier",
        "input_dimension": input_dim,
        "num_classes": num_classes,
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "estimated_mult_adds": int(estimated_mult_adds),
        "note": "FLOPs are estimated for the classifier head only. DINOv2 is used as pretrained feature extractor."
    }

    with open(os.path.join(RESULTS_DIR, "flops_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 70)
    print("FLOPs / Complexity Report")
    print("=" * 70)
    print(json.dumps(report, indent=2))
    print(f"Saved to: {os.path.join(RESULTS_DIR, 'flops_report.json')}")
    print("=" * 70)


if __name__ == "__main__":
    main()