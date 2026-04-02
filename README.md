<div align="center">

# Emotion Detection Lab

**Binary classification of facial expressions (happy vs neutral)**

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Cloud Run](https://img.shields.io/badge/Cloud%20Run-Live-4285F4?logo=googlecloud&logoColor=white)](https://emotion-detection-991380836709.us-central1.run.app/)
[![License](https://img.shields.io/badge/License-Academic-gray)]()

*An interactive web application for training, evaluating, and comparing 51 emotion classification models on the SMILE_PLUS dataset.*

**DTI 6402 -- Affective Computing | University of Ottawa**

---

**[Live Demo](https://emotion-detection-991380836709.us-central1.run.app/)** | **Best Model:** CLIP Linear Probe | **F1 = 0.9758** | **51 Models Trained**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Using the Web App](#using-the-web-app)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#1-clone-the-repository)
  - [Install Dependencies](#2-install-dependencies)
  - [Set Up the Dataset](#3-set-up-the-dataset)
  - [Run the App](#4-run-the-app)
  - [Train All Models (Optional)](#5-train-all-models-optional)
  - [Google OAuth Setup (Optional)](#6-google-oauth-setup-optional)
- [Pages](#pages)
- [Pipeline](#pipeline)
- [Results](#results)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

---

## Overview

Emotion Detection Lab is a full-stack ML application that lets you explore a facial expression dataset, engineer features, train classifiers, compare results on a leaderboard, and run batch predictions -- all through a clean Streamlit interface with Google OAuth.

The app supports **7 feature extraction methods** (from HOG to CLIP), **9 classifiers** (traditional + deep linear probes), and evaluates all **51 model combinations** with 10-fold stratified cross-validation.

---

## Using the Web App

The app is live at **[emotion-detection-991380836709.us-central1.run.app](https://emotion-detection-991380836709.us-central1.run.app/)**. Sign in with your Google account to get started.

### Exploring the Dataset

Navigate to **Dataset Explorer** to browse the SMILE_PLUS dataset. You can view the class distribution (250 happy, 250 neutral), browse sample images, and see image statistics like resolution and intensity histograms.

### Training a Model

1. Go to **Feature Engineering** and select which feature extraction methods to use (HOG, LBP, Gabor, Landmarks, ConvNeXt, CLIP, InsightFace). Click **Extract** to compute and cache features.
2. Go to **Model Training**, pick a classifier (SVM, Random Forest, XGBoost, etc.) and a feature set. Adjust hyperparameters with the sliders, then click **Train** to run 10-fold cross-validation.
3. View detailed per-fold results, confusion matrices, and ROC curves on the **Results** page.

### Comparing Models

The **Model Comparison** page shows a leaderboard of all trained models ranked by F1 score. Use the bar chart and radar chart to compare performance across metrics.

### Batch Prediction on Your Own Images

Go to the **Batch Prediction** page to classify your own face images:

1. **Select a model** from the dropdown (defaults to the best-performing model).
2. **Upload face images** (JPG, JPEG, PNG, or BMP). You can upload multiple images at once.
3. **(Optional) Upload a ground-truth label file** to evaluate the model on your data. The label file should be a CSV with two columns and no header: `filename,label`, where the label is either `happy` or `neutral`. For example:
   ```
   photo1.jpg,happy
   photo2.jpg,neutral
   photo3.jpg,happy
   ```
4. Click **Predict** to run inference.

**Results you get:**
- A summary with total image count and happy/neutral distribution.
- If you uploaded labels: **evaluation metrics** (Accuracy, F1, Precision, Recall, FPR, Specificity), a **confusion matrix** (color-coded: green for correct, red for errors), and an **ROC curve** with AUC.
- Per-image predictions with the predicted label, confidence score, and (if labels were provided) a checkmark or X indicating whether the prediction was correct.

Filename matching between your images and the label CSV is case-insensitive, so `Photo1.JPG` in the CSV will match `photo1.jpg` in your upload. If some images have no matching label (or vice versa), the app computes metrics on the matched subset and shows warnings for unmatched files.

---

## Getting Started

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.13+ | Required |
| **uv** | latest | Recommended package manager ([install](https://docs.astral.sh/uv/getting-started/installation/)) |
| **pip** | latest | Alternative to uv |
| **Git** | any | To clone the repository |

### 1. Clone the Repository

```bash
git clone https://github.com/pouriamrt/emotion-detection-lab.git
cd emotion-detection-lab
```

### 2. Install Dependencies

**Using uv (recommended):**

```bash
uv sync
```

**Using pip:**

```bash
pip install -r requirements.txt
```

> **Note:** The project uses CPU-only PyTorch by default. For GPU support, install PyTorch with CUDA separately before running `pip install -r requirements.txt`. See [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 3. Set Up the Dataset

Download the **SMILE_PLUS Training Set** and place it in the `data/` directory with this structure:

```
data/
  annotations.csv          # Format: filename,label (no header)
  SMILE PLUS Training Set/
    1a.jpg
    1b.jpg
    2a.jpg
    ...
```

The dataset contains **500 grayscale JPEG images** (162 x 193 px) -- 250 happy and 250 neutral.

### 4. Run the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**. If Google OAuth is not configured, it runs in **dev mode** with a default user -- no extra setup needed.

### 5. Train All Models (Optional)

Pre-trained models are included in `artifacts/`. To retrain everything from scratch:

```bash
python -m scripts.train_all
```

This trains all 51 model/feature combinations (~40 min on a modern CPU) and saves results to `artifacts/`.

### 6. Google OAuth Setup (Optional)

<details>
<summary>Click to expand</summary>

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

## Pages

| Page | Description |
|------|-------------|
| **Dataset Explorer** | Browse class distribution, sample images, and image statistics |
| **Feature Engineering** | Select, extract, and cache 7 feature methods with status tracking |
| **Model Training** | Train any model/feature combination with hyperparameter tuning |
| **Results** | Confusion matrices, ROC curves, and per-fold breakdowns |
| **Model Comparison** | Leaderboard, bar charts, and radar chart across all models |
| **Batch Prediction** | Upload face images to classify with any trained model; optionally upload ground-truth labels to get evaluation metrics (F1, accuracy, confusion matrix, ROC/AUC) |

---

## Pipeline

### Feature Extractors

| Method | Type | Dims | Description |
|--------|------|------|-------------|
| HOG | Traditional | -- | Histogram of Oriented Gradients -- edge & shape info |
| LBP | Traditional | -- | Local Binary Patterns -- multi-scale texture encoding |
| Gabor | Traditional | -- | Gabor filter bank -- multi-orientation frequency features |
| Landmarks | Traditional | -- | MediaPipe face landmarks -- geometric distances |
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

**3 Deep Linear Probes** -- 2-layer MLP on frozen embeddings

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

## Deployment

### Deploy to Google Cloud Run

<details>
<summary><strong>Prerequisites</strong></summary>

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and configured
- A GCP project with billing enabled
- Google OAuth credentials (see [OAuth setup](#6-google-oauth-setup-optional))

</details>

```bash
# One-time setup: enable APIs and upload secrets
make setup

# Build and deploy
make deploy
```

After deploying, add the Cloud Run URL to your OAuth authorized redirect URIs, update `.streamlit/secrets.toml`, then run `make update-secrets`.

### Docker (Local)

```bash
# Build and run the container locally
make docker-run
```

The app will be available at **http://localhost:8080**.

### Make Commands

| Command | Description |
|---------|-------------|
| `make setup` | One-time GCP setup (APIs + secrets) |
| `make deploy` | Build and deploy to Cloud Run |
| `make deploy-only` | Deploy without rebuilding (uses last image) |
| `make update-secrets` | Push updated secrets to GCP |
| `make run` | Run locally with Streamlit |
| `make docker-run` | Build and run locally in Docker |
| `make logs` | View Cloud Run logs |
| `make status` | Show service status |
| `make url` | Print the deployed URL |
| `make delete` | Delete the Cloud Run service |

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
│   └── 6_Batch_Prediction.py     # Inference + optional label evaluation
├── src/
│   ├── auth.py                   # Google OAuth 2.0 with PKCE
│   ├── config.py                 # Paths, constants, registries
│   ├── data_loader.py            # SMILE_PLUS dataset loading
│   ├── features.py               # 7 feature extractors with caching
│   ├── models.py                 # Model factories (traditional + deep)
│   ├── evaluation.py             # 10-fold CV + compute_metrics() helper
│   └── logger.py                 # Loguru logging setup
├── scripts/
│   └── train_all.py              # Offline batch training of all 51 models
├── artifacts/
│   ├── metadata.json             # Leaderboard & best model reference
│   ├── models/                   # Trained .joblib files (51 models)
│   ├── features/                 # Cached .npy feature matrices
│   └── results/                  # Per-model CV result .json files
├── tests/
│   └── test_evaluation.py        # Unit tests for compute_metrics()
├── data/                         # SMILE_PLUS dataset (500 images)
├── Dockerfile                    # Production container (CPU-only PyTorch)
├── Makefile                      # Build, deploy, and manage commands
├── requirements.txt              # Python dependencies
└── pyproject.toml                # Project metadata
```

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

*University of Ottawa -- DTI 6402 Affective Computing*

</div>
