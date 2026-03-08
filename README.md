<div align="center">

# Emotion Detection Lab

**Binary classification of facial expressions (happy vs neutral)**

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Cloud Run](https://img.shields.io/badge/Cloud%20Run-Deployed-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com/run)
[![License](https://img.shields.io/badge/License-Academic-gray)]()

*An interactive web application for training, evaluating, and comparing 51 emotion classification models on the SMILE_PLUS dataset.*

**CSI 5386 — Affective Computing | University of Ottawa**

---

**Best Model:** CLIP Linear Probe | **F1 = 0.9758** | **51 Models Trained** | **10-Fold CV**

</div>

---

## Overview

Emotion Detection Lab is a full-stack ML application that lets you explore a facial expression dataset, engineer features, train classifiers, compare results on a leaderboard, and run batch predictions — all through a clean Streamlit interface with Google OAuth.

### Pages

| Page | Description |
|------|-------------|
| **Dataset Explorer** | Browse class distribution, sample images, and image statistics |
| **Feature Engineering** | Select, extract, and cache 7 feature methods with status tracking |
| **Model Training** | Train any model/feature combination with hyperparameter tuning |
| **Results** | Confusion matrices, ROC curves, and per-fold breakdowns |
| **Model Comparison** | Leaderboard, bar charts, and radar chart across all models |
| **Batch Prediction** | Upload face images and classify them with any trained model |

---

## Project Structure

```
.
├── app.py                        # Home page & navigation
├── pages/
│   ├── 1_Dataset_Explorer.py     # Data visualization
│   ├── 2_Feature_Engineering.py  # Feature extraction pipeline
│   ├── 3_Model_Training.py       # Interactive training with 10-fold CV
│   ├── 4_Results.py              # Per-model metrics & charts
│   ├── 5_Model_Comparison.py     # Cross-model leaderboard
│   └── 6_Batch_Prediction.py     # Inference on uploaded images
├── src/
│   ├── auth.py                   # Google OAuth 2.0 with PKCE
│   ├── config.py                 # Paths, constants, registries
│   ├── data_loader.py            # SMILE_PLUS dataset loading
│   ├── features.py               # 7 feature extractors with caching
│   ├── models.py                 # Model factories (traditional + deep)
│   ├── evaluation.py             # 10-fold stratified cross-validation
│   └── logger.py                 # Loguru logging setup
├── scripts/
│   └── train_all.py              # Offline batch training of all 51 models
├── artifacts/
│   ├── metadata.json             # Leaderboard & best model reference
│   ├── models/                   # Trained .joblib files (51 models)
│   ├── features/                 # Cached .npy feature matrices
│   └── results/                  # Per-model CV result .json files
├── data/                         # SMILE_PLUS dataset (500 images)
├── Dockerfile                    # Production container (CPU-only PyTorch)
├── Makefile                      # Build, deploy, and manage commands
├── requirements.txt              # Python dependencies
└── pyproject.toml                # Project metadata
```

---

## Dataset

**SMILE_PLUS Training Set** — 500 grayscale JPEG images (162 x 193 px)

| Class | Count | Split |
|-------|-------|-------|
| Happy | 250 | 50% |
| Neutral | 250 | 50% |

---

## Pipeline

### Feature Extractors

| Method | Type | Dims | Description |
|--------|------|------|-------------|
| HOG | Traditional | — | Histogram of Oriented Gradients — edge & shape info |
| LBP | Traditional | — | Local Binary Patterns — multi-scale texture encoding |
| Gabor | Traditional | — | Gabor filter bank — multi-orientation frequency features |
| Landmarks | Traditional | — | MediaPipe face landmarks — geometric distances |
| ConvNeXt-Tiny | Deep | 768 | ImageNet-pretrained visual features |
| CLIP-ViT-B/32 | Deep | 512 | Semantically rich vision-language embeddings |
| InsightFace | Deep | 512 | ArcFace face-specific embeddings |

### Models

**6 Traditional Classifiers** trained on **8 feature combinations** = **48 models**

| Classifier | Key Hyperparameters |
|------------|-------------------|
| SVM | C, kernel, gamma |
| Random Forest | n_estimators, max_depth |
| XGBoost | n_estimators, learning_rate, max_depth |
| LightGBM | n_estimators, num_leaves, learning_rate |
| Logistic Regression | C, penalty |
| KNN | n_neighbors, weights, metric |

**3 Deep Linear Probes** — 2-layer MLP on frozen embeddings

| Model | Embedding Source |
|-------|-----------------|
| CLIP Linear Probe | CLIP-ViT-B/32 |
| ConvNeXt-Tiny (Fine-tune) | ConvNeXt-Tiny |
| InsightFace + FC | InsightFace |

<details>
<summary><strong>Feature combinations (8)</strong></summary>

| # | Features |
|---|----------|
| 1 | InsightFace |
| 2 | CLIP-ViT-B/32 |
| 3 | ConvNeXt-Tiny |
| 4 | InsightFace + CLIP-ViT-B/32 |
| 5 | InsightFace + CLIP-ViT-B/32 + ConvNeXt-Tiny |
| 6 | HOG + LBP + Gabor |
| 7 | HOG + LBP + Gabor + Landmarks |
| 8 | HOG + LBP + Gabor + InsightFace |

</details>

### Evaluation

All models evaluated with **10-fold stratified cross-validation** (seed 42):

- **Metrics:** F1, Precision, Recall, FPR, Confusion Matrix, ROC/AUC
- **Scaling:** StandardScaler per fold for traditional models

---

## Results

### Top 5 Models

| Rank | Model | F1 |
|------|-------|----|
| 1 | CLIP Linear Probe | **0.9758** |
| 2 | SVM (InsightFace + CLIP) | 0.9720 |
| 3 | Logistic Regression (InsightFace + CLIP) | 0.9700 |
| 4 | LightGBM (InsightFace + CLIP + ConvNeXt) | 0.9680 |
| 5 | XGBoost (InsightFace + CLIP) | 0.9660 |

---

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

### Installation

```bash
git clone <repo-url>
cd "Project 1"

# With uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

### Dataset Setup

Place the SMILE_PLUS dataset in the `data/` directory:

```
data/
  annotations.csv
  SMILE PLUS Training Set/
    1a.jpg
    1b.jpg
    ...
```

### Running Locally

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. If Google OAuth is not configured, it runs in **dev mode** with a default user.

### Training All Models

```bash
python -m scripts.train_all
```

Trains all 51 model combinations (~40 min) and saves artifacts.

<details>
<summary><strong>Google OAuth Setup (optional)</strong></summary>

1. Create a project in [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the **Google People API**
3. Create **OAuth 2.0 credentials** (Web application type)
4. Add `http://localhost:8501` to authorized redirect URIs and JavaScript origins
5. Download credentials as `client_secret.json` in the project root
6. Create `.streamlit/secrets.toml`:

```toml
[google_auth]
cookie_secret = "<random-hex-string>"
redirect_uri = "http://localhost:8501"
```

</details>

---

## Deployment

Deploy to **Google Cloud Run** with a single command.

### Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- A GCP project with billing enabled

### Deploy

```bash
# One-time setup: enable APIs and upload secrets
make setup

# Build and deploy
make deploy
```

After deploying, add the Cloud Run URL to your OAuth authorized redirect URIs, update `.streamlit/secrets.toml`, then run `make update-secrets`.

### Make Commands

| Command | Description |
|---------|-------------|
| `make setup` | One-time GCP setup (APIs + secrets) |
| `make deploy` | Build and deploy to Cloud Run |
| `make update-secrets` | Push updated secrets to GCP |
| `make logs` | View Cloud Run logs |
| `make status` | Show service status |
| `make url` | Print the deployed URL |
| `make docker-run` | Test locally with Docker |
| `make delete` | Delete the Cloud Run service |

---

## Tech Stack

| | Technology |
|---|-----------|
| **UI** | Streamlit |
| **Auth** | Google OAuth 2.0 with PKCE |
| **ML** | scikit-learn, XGBoost, LightGBM, PyTorch |
| **Features** | OpenCV, MediaPipe, CLIP, ConvNeXt, InsightFace |
| **Visualization** | Plotly |
| **Logging** | Loguru |
| **Deployment** | Docker, Google Cloud Run |
| **Package Manager** | uv |

---

<div align="center">

*University of Ottawa — CSI 5386 Affective Computing*

</div>
