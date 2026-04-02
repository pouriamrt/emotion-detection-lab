import json

import cv2
import joblib
import numpy as np
import plotly.express as px
import streamlit as st

from src.auth import require_auth
from src.config import METADATA_PATH, MODELS_DIR, POSITIVE_LABEL, NEGATIVE_LABEL
from src.features import extract_features
from src.logger import get_logger

user = require_auth()
log = get_logger(user=user["name"], context="prediction")

with st.sidebar:
    st.markdown(f"**Logged in as:** {user['name']}")
    st.caption(user["email"])

st.title("Batch Prediction")
st.markdown("Upload face images and classify them as **happy** or **neutral** using a trained model.")

# ── Load metadata ───────────────────────────────────────────────────────
if not METADATA_PATH.exists():
    st.info("No trained models found. Run `python -m scripts.train_all` first.")
    st.stop()

with open(METADATA_PATH) as f:
    metadata = json.load(f)

best_key = metadata["best_model"]
model_keys = list(metadata["models"].keys())

# ── Model selector ──────────────────────────────────────────────────────
selected_model = st.selectbox(
    "Select model",
    model_keys,
    index=model_keys.index(best_key) if best_key in model_keys else 0,
    help="Defaults to the best performing model.",
)
model_info = metadata["models"][selected_model]
st.caption(
    f"**Type:** {model_info['type']} | "
    f"**Features:** {', '.join(model_info['features'])} | "
    f"**F1:** {model_info['f1']:.4f}"
)

# ── File uploader ───────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload face images",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
)

# ── Optional label upload ──────────────────────────────────────────────
label_file = st.file_uploader(
    "Upload ground-truth labels (optional)",
    type=["csv"],
    help=(
        "CSV with two columns, no header: filename,label. "
        "Labels must be 'happy' or 'neutral' (case-insensitive)."
    ),
)

ground_truth: dict[str, int] | None = None
if label_file is not None:
    import pandas as pd

    try:
        label_df = pd.read_csv(label_file, header=None, names=["filename", "label"])
    except Exception as exc:
        st.error(f"Could not parse label CSV: {exc}")
        label_df = None

    if label_df is not None:
        if label_df.shape[1] < 2:
            st.error("Label CSV must have two columns: filename, label.")
            label_df = None

    if label_df is not None:
        label_df["label_lower"] = label_df["label"].str.strip().str.lower()
        invalid = label_df[~label_df["label_lower"].isin(["happy", "neutral"])]
        if len(invalid) > 0:
            bad_vals = invalid["label"].unique().tolist()
            st.error(
                f"Invalid labels found: {bad_vals}. "
                "Only 'happy' and 'neutral' are accepted."
            )
            label_df = None

    if label_df is not None:
        ground_truth = {
            row["filename"].lower(): int(row["label_lower"] == "happy")
            for _, row in label_df.iterrows()
        }

if not uploaded_files:
    st.info("Upload one or more face images to get started.")
    st.stop()

# ── Read uploaded images ────────────────────────────────────────────────
images_color = []
images_gray = []
filenames = []
skipped = []

for f in uploaded_files:
    raw = np.frombuffer(f.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img_bgr is None:
        skipped.append(f.name)
        continue
    images_color.append(cv2.resize(img_bgr, (224, 224)))
    images_gray.append(cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (128, 128)))
    filenames.append(f.name)

if skipped:
    st.warning(f"Skipped {len(skipped)} unreadable file(s): {', '.join(skipped)}")

if not filenames:
    st.error("No valid images found in the uploaded files.")
    st.stop()

images_color_arr = np.array(images_color)
images_gray_arr = np.array(images_gray)

st.markdown(f"**{len(filenames)} image(s) loaded.** Click **Predict** to run inference.")

# ── Predict button ──────────────────────────────────────────────────────
if not st.button("Predict", type="primary"):
    st.stop()

from pathlib import Path

model_path = Path(model_info.get("model_path", ""))
if not model_path.exists():
    # Fallback: try standard naming in models dir
    safe_key = selected_model.replace("/", "_")
    model_path = MODELS_DIR / f"{safe_key}.joblib"

if not model_path.exists():
    st.error(f"Model file not found: `{model_path}`")
    st.stop()

with st.spinner("Extracting features and running predictions..."):
    # 1. Extract features
    feature_methods = model_info["features"]
    log.info(f"Extracting features {feature_methods} for {len(filenames)} images")
    X = extract_features(images_gray_arr, images_color_arr, feature_methods, cache=False)

    # 2. Load model and predict
    artifact = joblib.load(str(model_path))
    if model_info["type"] == "traditional":
        model = artifact["model"]
        scaler = artifact["scaler"]
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probas = model.predict_proba(X_scaled) if hasattr(model, "predict_proba") else None
    else:
        model = artifact
        predictions = model.predict(X)
        probas = model.predict_proba(X) if hasattr(model, "predict_proba") else None

labels = [POSITIVE_LABEL if p == 1 else NEGATIVE_LABEL for p in predictions]
confidences = None
if probas is not None:
    confidences = [probas[i, predictions[i]] for i in range(len(predictions))]

log.info(
    f"Predictions complete: {sum(predictions)} happy, "
    f"{len(predictions) - sum(predictions)} neutral"
)

# ── Summary ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Summary")

n_happy = int(sum(predictions))
n_neutral = len(predictions) - n_happy

col1, col2, col3 = st.columns(3)
col1.metric("Total Images", len(predictions))
col2.metric("Happy", n_happy)
col3.metric("Neutral", n_neutral)

fig_pie = px.pie(
    names=[POSITIVE_LABEL, NEGATIVE_LABEL],
    values=[n_happy, n_neutral],
    color_discrete_sequence=["#2ecc71", "#3498db"],
    title="Prediction Distribution",
)
st.plotly_chart(fig_pie, width="stretch")

# ── Per-image results ───────────────────────────────────────────────────
st.subheader("Per-Image Results")

n_cols = 4
for row_start in range(0, len(filenames), n_cols):
    cols = st.columns(n_cols)
    for i, col in enumerate(cols):
        idx = row_start + i
        if idx >= len(filenames):
            break
        with col:
            # Show thumbnail (convert BGR→RGB for display)
            img_rgb = cv2.cvtColor(images_color[idx], cv2.COLOR_BGR2RGB)
            st.image(img_rgb, width=150)
            label = labels[idx]
            color = "green" if label == POSITIVE_LABEL else "blue"
            conf_str = f" ({confidences[idx]:.1%})" if confidences else ""
            st.markdown(
                f"**:{color}[{label.upper()}]**{conf_str}",
            )
            st.caption(filenames[idx])

log.info(f"Batch prediction page completed for model: {selected_model}")
