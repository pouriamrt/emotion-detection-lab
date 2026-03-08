# Emotion Detection Streamlit App — Design

## Problem
Binary classification: happy vs neutral facial expressions on SMILE_PLUS dataset (500 images, 250/class, grayscale 162x193).

Performance target: F1 >= 0.98 for full marks. Formula: min(F1*35/0.98, 35).

## Architecture

Multi-page Streamlit app with Google OAuth, loguru logging, pre-trained models.

```
app.py                              # Home + Google OAuth login
pages/
  1_Dataset_Explorer.py             # View images, class distribution
  2_Feature_Engineering.py          # Select & extract features
  3_Model_Training.py               # Select model, tune hyperparams, run 10-fold CV
  4_Results.py                      # Metrics, confusion matrix, ROC curves
  5_Model_Comparison.py             # Leaderboard, recommended model
src/
  config.py                         # Constants
  logger.py                         # Loguru (auth.log, training.log, app.log)
  auth.py                           # Google OAuth via streamlit-google-auth
  data_loader.py                    # Load SMILE_PLUS
  features.py                       # All feature extraction
  models.py                         # Model registry
  evaluation.py                     # 10-fold stratified CV
scripts/
  train_all.py                      # Offline: train all models, save artifacts
artifacts/
  features/                         # Cached feature matrices
  models/                           # Saved models (.joblib, .pt)
  results/                          # Pre-computed metrics
  metadata.json                     # Model catalog with scores
```

## Feature Engineering

### Traditional
- HOG (~1764 dim): edge/shape gradients
- LBP (~256 dim): texture patterns
- Gabor (~1600 dim): multi-scale frequency features
- Facial landmarks (~136 dim): geometric via MediaPipe

### Deep Embeddings (frozen backbone)
- ConvNeXt-Tiny (768 dim): general visual features
- CLIP ViT-B/32 (512 dim): semantic vision-language features
- InsightFace/ArcFace (512 dim): face-specific features

Feature fusion: concatenation of selected features, StandardScaler, optional PCA.

## Models

### Traditional (trained on extracted features)
- SVM (C, gamma, kernel)
- Random Forest (n_estimators, max_depth, min_samples_split)
- XGBoost (learning_rate, max_depth, n_estimators, subsample)
- LightGBM (learning_rate, num_leaves, n_estimators)
- Logistic Regression (C, penalty)
- KNN (n_neighbors, weights, metric)

### Deep Learning (last-layer fine-tuning only)
- ConvNeXt-Tiny: freeze all, train classifier head (768->2)
- CLIP Linear Probe: freeze CLIP, train linear (512->2)
- InsightFace + FC: freeze ArcFace, train FC (512->2)
- Hyperparams: learning_rate, epochs, batch_size, weight_decay

### Ensemble
- Soft Voting (top-N models)
- Stacking (meta-learner on base predictions)

## Evaluation
- 10-fold Stratified CV
- Positive class: happy
- Metrics: FPR, Precision, Recall, F1, Confusion Matrix, ROC

## Pre-trained Models
- `scripts/train_all.py` trains all models offline with optimal hyperparams
- Artifacts saved to `artifacts/`
- App loads pre-trained results on startup
- Users can optionally re-train with custom hyperparams

## Auth & Logging
- Google OAuth via streamlit-google-auth (dev bypass when unconfigured)
- Loguru: auth.log, training.log, app.log with rotation
