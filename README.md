# Emotion Detection Lab

Binary classification of facial expressions (**happy** vs **neutral**) on the SMILE_PLUS dataset. Built as an interactive Streamlit web app with Google OAuth, 51 trained model combinations, and batch prediction.

> **Course:** CSI 5386 — Affective Computing, University of Ottawa
> **Best Model:** CLIP Linear Probe — F1 = 0.9758

---

## Features

- **Dataset Explorer** — Browse class distribution, sample images, and image statistics
- **Feature Engineering** — Select, extract, and cache 7 feature methods with status tracking
- **Model Training** — Train any model/feature combination interactively with hyperparameter tuning
- **Results Viewer** — Confusion matrices, ROC curves, per-fold breakdowns for every model run
- **Model Comparison** — Leaderboard, F1 bar chart, and radar chart across all 51 models
- **Batch Prediction** — Upload face images and classify them with any trained model

## Architecture

```
app.py                          # Home page + navigation
pages/
  1_Dataset_Explorer.py         # Data visualization
  2_Feature_Engineering.py      # Feature extraction pipeline
  3_Model_Training.py           # Interactive training with 10-fold CV
  4_Results.py                  # Per-model metrics & charts
  5_Model_Comparison.py         # Cross-model leaderboard
  6_Batch_Prediction.py         # Inference on uploaded images
src/
  auth.py                       # Google OAuth with PKCE
  config.py                     # Paths, constants, model/feature registries
  data_loader.py                # SMILE_PLUS dataset loading
  features.py                   # 7 feature extractors with caching
  models.py                     # Model factories (traditional + deep)
  evaluation.py                 # 10-fold stratified cross-validation
  logger.py                     # Loguru logging setup
scripts/
  train_all.py                  # Offline batch training of all 51 combos
artifacts/
  metadata.json                 # Leaderboard & best model reference
  models/                       # Trained .joblib files
  features/                     # Cached .npy feature matrices
  results/                      # Per-model CV result .json files
```

## Dataset

**SMILE_PLUS Training Set** — 500 grayscale JPEG images (162 x 193 px), balanced 250/250 happy/neutral.

| Class   | Count |
|---------|-------|
| Happy   | 250   |
| Neutral | 250   |

## Feature Extractors

| Method | Type | Dimensionality | Description |
|--------|------|----------------|-------------|
| HOG | Traditional | Variable | Histogram of Oriented Gradients — edge and shape information |
| LBP | Traditional | Variable | Local Binary Patterns — multi-scale texture encoding |
| Gabor | Traditional | Variable | Gabor filter bank — multi-scale, multi-orientation frequency features |
| Landmarks | Traditional | Variable | MediaPipe facial landmarks — geometric distances between key points |
| ConvNeXt-Tiny | Deep | 768 | ImageNet-pretrained general-purpose visual features |
| CLIP-ViT-B/32 | Deep | 512 | CLIP Vision Transformer — semantically rich embeddings |
| InsightFace | Deep | 512 | ArcFace — face-specific embeddings |

## Models

### Traditional ML (6 classifiers x 8 feature combinations = 48 models)

| Classifier | Key Hyperparameters |
|------------|-------------------|
| SVM | C, kernel, gamma |
| Random Forest | n_estimators, max_depth |
| XGBoost | n_estimators, learning_rate, max_depth |
| LightGBM | n_estimators, num_leaves, learning_rate |
| Logistic Regression | C, penalty |
| KNN | n_neighbors, weights, metric |

**Feature combinations trained:**

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

### Deep Linear Probes (3 models)

| Model | Feature Source | Architecture |
|-------|---------------|--------------|
| CLIP Linear Probe | CLIP-ViT-B/32 | 2-layer MLP on frozen CLIP embeddings |
| ConvNeXt-Tiny (Fine-tune) | ConvNeXt-Tiny | 2-layer MLP on frozen ConvNeXt embeddings |
| InsightFace + FC | InsightFace | 2-layer MLP on frozen ArcFace embeddings |

## Evaluation

All models are evaluated with **10-fold stratified cross-validation**:

- **Metrics:** F1, Precision, Recall, FPR, Confusion Matrix, ROC/AUC
- **Seed:** 42 (deterministic splits)
- **Scaling:** StandardScaler for traditional models (fit on train, transform on val per fold)

### Top 5 Models

| Rank | Model | F1 |
|------|-------|----|
| 1 | CLIP Linear Probe | 0.9758 |
| 2 | SVM (InsightFace + CLIP) | 0.9720 |
| 3 | Logistic Regression (InsightFace + CLIP) | 0.9700 |
| 4 | LightGBM (InsightFace + CLIP + ConvNeXt) | 0.9680 |
| 5 | XGBoost (InsightFace + CLIP) | 0.9660 |

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd "Project 1"

# Install dependencies
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

### Google OAuth Setup (Optional)

1. Create a project in [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the Google People API
3. Create OAuth 2.0 credentials (Web application)
4. Download as `client_secret.json` in the project root
5. Create `.streamlit/secrets.toml`:

```toml
[google_auth]
cookie_secret = "<random-hex-string>"
redirect_uri = "http://localhost:8501"
```

If OAuth is not configured, the app runs in **dev mode** with a default user.

### Running Locally

```bash
# Run the Streamlit app
streamlit run app.py

# Or with make
make run
```

### Training All Models (Offline)

```bash
python -m scripts.train_all
```

This trains all 51 model combinations (~40 minutes) and saves artifacts to `artifacts/`.

## Deployment (Google Cloud Run)

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

After deploying, add the Cloud Run URL to your Google OAuth authorized redirect URIs and update `.streamlit/secrets.toml`, then run:

```bash
make update-secrets
```

### Available Make Commands

| Command | Description |
|---------|-------------|
| `make setup` | One-time GCP setup (APIs + secrets) |
| `make deploy` | Build and deploy to Cloud Run |
| `make update-secrets` | Update secrets after config changes |
| `make logs` | View Cloud Run logs |
| `make status` | Show service status |
| `make url` | Print the deployed URL |
| `make docker-run` | Build and test locally with Docker |
| `make delete` | Delete the Cloud Run service |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| Auth | Google OAuth 2.0 with PKCE |
| ML | scikit-learn, XGBoost, LightGBM, PyTorch |
| Features | OpenCV, MediaPipe, CLIP, ConvNeXt, InsightFace |
| Charts | Plotly |
| Logging | Loguru |
| Deployment | Docker, Google Cloud Run |
| Package Manager | uv |

## License

University coursework — not licensed for redistribution.
