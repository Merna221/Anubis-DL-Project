# Anubis-DL-Project

Deep Learning project for Egyptian Artifact Recognition and Reconstruction using DINOv2 and Stable Diffusion.

---

# Project Overview

This project focuses on applying Deep Learning techniques in the field of Cultural Heritage and Egyptian Artifact Analysis.

The system performs two major tasks:

1. Egyptian Artifact Classification
2. Artifact Reconstruction and Restoration

The project combines:
- Self-supervised visual feature extraction using DINOv2
- Deep Neural Network classification
- Stable Diffusion Img2Img reconstruction
- Image preprocessing and augmentation
- Regression and classification evaluation metrics
- FLOPs analysis
- Energy consumption and carbon emission tracking

---

# Main Objectives

The project aims to:

- Recognize Egyptian artifacts automatically
- Extract deep visual embeddings using DINOv2
- Train a neural classifier for artifact identification
- Restore damaged artifacts using Stable Diffusion
- Evaluate both classification and reconstruction performance
- Analyze computational complexity and sustainability metrics

---

# Technologies Used

## Deep Learning
- PyTorch
- Torchvision
- HuggingFace Transformers
- Diffusers
- Stable Diffusion v1.5

## Image Processing
- OpenCV
- PIL
- NumPy

## Evaluation and Visualization
- Scikit-learn
- Matplotlib
- Seaborn

## Sustainability Tracking
- CodeCarbon

---

# Project Structure

```text
AI_Enhancement/
│
├── app/
│   ├── data/
│   │   ├── artifacts/
│   │   ├── reconstruction_pairs/
│   │   ├── restoration_results/
│   │   └── tmp/
│   │
│   ├── results/
│   │   ├── classification/
│   │   ├── restoration/
│   │   ├── flops/
│   │   └── carbon/
│   │
│   ├── scripts/
│   ├── services/
│   ├── config.py
│   ├── deps.py
│   ├── main.py
│   └── schemas.py
│
├── requirements.txt
├── README.md
└── .env
```

---

# Dataset Description

## Dataset 1 — Classification Dataset

The classification dataset contains:

- 10 Egyptian artifacts
- Approximately 80 artifact images
- Multiple views per artifact
- Different lighting and angles

Examples include:
- Tutankhamun Mask
- Bust of Nefertiti
- Anubis Statue
- Ramses II Statue
- Canopic Jars

Each artifact represents one class.

---

## Dataset 2 — Reconstruction Dataset

The reconstruction dataset contains:

### Input Images
Damaged or low-quality artifact images.

### Target Images
Clean and high-quality reference images.

### Stable Diffusion Outputs
AI-generated reconstructed artifact images.

---

# Data Preprocessing

Several preprocessing techniques were applied:

## Image Resizing
All images were resized to standard dimensions.

## RGB Conversion
Images were converted into RGB format.

## Normalization
Pixel values were normalized before model processing.

## Feature Extraction
DINOv2 embeddings were extracted from artifact images.

## Dataset Organization
Images were grouped into:
- input
- target
- stable_diffusion_output

---

# Model 1 — DINOv2 Artifact Classification

## Overview

DINOv2 is used as a pretrained visual feature extractor.

The model generates high-dimensional embeddings representing artifact features.

These embeddings are passed into a neural network classifier.

---

## Classification Pipeline

1. Load artifact image
2. Extract DINOv2 embedding
3. Feed embedding into classifier
4. Predict artifact class

---

## Neural Network Classifier

The classifier includes:
- Fully connected layers
- ReLU activation
- Dropout regularization
- Softmax output

---

# Model 2 — Stable Diffusion Reconstruction

## Overview

Stable Diffusion Img2Img is used to reconstruct and restore Egyptian artifacts.

The model receives:
- an input artifact image
- restoration prompts

and generates:
- visually enhanced reconstructed images

---

## Reconstruction Workflow

1. Upload artifact image
2. Apply Stable Diffusion Img2Img
3. Generate reconstructed artifact
4. Compare output against target image

---

## Prompt Engineering

Prompts included:
- museum quality
- restored artifact
- ancient Egyptian artifact
- realistic texture
- clean surface

Negative prompts removed:
- blur
- distortions
- watermark
- low quality

---

# Training Process

## Classification Training

The classifier was trained using:
- CrossEntropyLoss
- Adam optimizer
- Multiple epochs

Training curves were generated for:
- Accuracy
- Loss

---

## Reconstruction Evaluation

Stable Diffusion outputs were evaluated using:
- MAE
- RMSE
- MAPE
- R²
- PSNR
- SSIM

---

# Evaluation Metrics

## Classification Metrics

The project evaluated:

- Accuracy
- Precision
- Recall
- F1-score
- Cohen Kappa
- Matthews Correlation Coefficient (MCC)
- ROC-AUC

---

## Reconstruction Regression Metrics

The project evaluated:

- MAE
- MAPE
- RMSE
- R² Score
- PSNR
- SSIM

---

# FLOPs and Complexity Analysis

The project includes computational complexity analysis.

Measured:
- Total parameters
- Trainable parameters
- Estimated multiply-add operations

The classifier head complexity was analyzed separately from DINOv2.

---

# Energy Consumption and Carbon Emissions

CodeCarbon was used to measure:

- Energy consumption
- Carbon emissions
- CPU usage
- GPU usage

This provides sustainability analysis for Deep Learning workloads.

---

# Results

## Classification Performance

The DINOv2 classifier achieved high classification accuracy across artifact classes.

Generated outputs include:
- Confusion matrix
- Accuracy curves
- Loss curves
- Classification report

---

## Reconstruction Performance

Stable Diffusion generated visually plausible artifact reconstructions.

Some generated images deviated from target references because:
- the model was not fine-tuned specifically for Egyptian artifacts
- reconstruction relied on general pretrained diffusion knowledge

Despite this, the model successfully demonstrated generative restoration capabilities.

---

# Output Files

## Classification Outputs

Located in:

```text
results/classification/
```

Includes:
- accuracy_curve.png
- loss_curve.png
- confusion_matrix.png
- classification_metrics.json
- classification_report.txt
- artifact_classifier.pt

---

## Reconstruction Outputs

Located in:

```text
results/restoration/
```

Includes:
- stable_diffusion_metrics.json
- restoration_regression_metrics.json

---

## FLOPs Outputs

Located in:

```text
results/flops/
```

Includes:
- flops_report.json

---

## Sustainability Outputs

Located in:

```text
results/carbon/
```

Includes:
- training_emissions.csv

---

# How to Run the Project

## 1. Clone Repository

```bash
git clone https://github.com/Merna221/Anubis-DL-Project.git
```

---

## 2. Navigate to Project

```bash
cd Anubis-DL-Project/AI_Enhancement
```

---

## 3. Create Virtual Environment

```bash
python -m venv .venv
```

---

## 4. Activate Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

---

## 5. Install Requirements

```bash
pip install -r requirements.txt
```

---

# Main Scripts

## Test Recognition Pipeline

This script tests the complete Egyptian artifact recognition pipeline on a real artifact image.

The script:
- loads the artifact image
- extracts DINOv2 embeddings
- compares image embeddings with stored artifact embeddings
- predicts the most similar artifact
- displays similarity confidence results

Run:

```bash
python -m app.scripts.test_pipeline_direct app/data/artifacts/images/artifact_001/<image_name>
```

Example:

```bash
python -m app.scripts.test_pipeline_direct app/data/artifacts/images/artifact_001/01.jpg
```

---

## Evaluate Recognition System

This script evaluates the overall artifact recognition system performance.

The script:
- tests artifact retrieval performance
- computes Top-1 accuracy
- computes Top-3 accuracy
- evaluates embedding similarity matching
- generates recognition evaluation summaries

Run:

```bash
python -m app.scripts.evaluate_recognition
```

---

## Train Classifier

This script trains the neural network classifier using DINOv2 embeddings.

The script:
- loads artifact embeddings
- trains the classification model
- saves trained model weights
- generates learning curves
- stores classification outputs

Run:

```bash
python -m app.scripts.train_classifier_optionA
```

---

## Evaluate Classification Metrics

This script evaluates the trained classification model performance.

The script computes:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Cohen Kappa
- Matthews Correlation Coefficient (MCC)

It also generates:
- confusion matrix
- classification report
- metric summaries

Run:

```bash
python -m app.scripts.evaluate_classifier_metrics
```

---

## Evaluate Stable Diffusion Results

This script evaluates the generated Stable Diffusion reconstruction outputs against target images.

The script computes:
- MAE
- MAPE
- RMSE
- R²
- PSNR
- SSIM

Run:

```bash
python -m app.scripts.evaluate_stable_diffusion_results
```

---

## Evaluate Restoration Regression

This script evaluates the overall restoration and reconstruction regression performance.

The script compares:
- restored artifact images
- reference target images

The script computes:
- MAE
- MAPE
- RMSE
- R²
- PSNR
- SSIM

Run:

```bash
python -m app.scripts.evaluate_restoration_regression
```

---

## Measure FLOPs

This script measures the computational complexity of the neural classifier.

The script computes:
- total parameters
- trainable parameters
- estimated multiply-add operations (FLOPs)

Run:

```bash
python -m app.scripts.measure_flops
```

---

# Conclusion

This project successfully demonstrates a multi-model Deep Learning system for:

- Egyptian artifact recognition
- Feature extraction
- Artifact reconstruction
- Sustainability analysis

The project combines discriminative and generative AI models to support digital cultural heritage preservation.

---
# Done by
Egyptian Chinese University (ECU)  
Faculty of Engineering  
Software Engineering Department
