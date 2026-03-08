import streamlit as st
import numpy as np
from src.auth import require_auth
from src.config import FEATURE_METHODS, FEATURES_DIR
from src.data_loader import load_images_for_traditional, load_images_for_deep, load_annotations
from src.features import extract_features, _cache_path
from src.logger import get_logger

user = require_auth()
log = get_logger(user=user["name"], context="features")

with st.sidebar:
    st.markdown(f"**Logged in as:** {user['name']}")
    st.caption(user["email"])

st.title("Feature Engineering")
st.markdown("Select feature extraction methods, check cache status, and extract features for model training.")

DESCRIPTIONS = {
    "HOG": "Histogram of Oriented Gradients — captures edge and shape information.",
    "LBP": "Local Binary Patterns — encodes texture patterns at multiple scales.",
    "Gabor": "Gabor filter bank — multi-scale, multi-orientation frequency features.",
    "Landmarks": "MediaPipe facial landmarks — geometric features between key facial points.",
    "ConvNeXt-Tiny": "ConvNeXt-Tiny (ImageNet pretrained) — 768-dim general-purpose visual features.",
    "CLIP-ViT-B/32": "CLIP Vision Transformer — 512-dim semantically rich embeddings.",
    "InsightFace": "ArcFace/InsightFace — 512-dim face-specific embeddings.",
}

# ── Feature method selection ─────────────────────────────────────────────
st.subheader("Select Feature Methods")

# Default: InsightFace + CLIP checked
default_methods = ["InsightFace", "CLIP-ViT-B/32"]
selected_methods = []

cols = st.columns(2)
for i, method in enumerate(FEATURE_METHODS):
    col = cols[i % 2]
    cache_file = _cache_path(method)
    cached = cache_file.exists()
    cache_icon = "✅" if cached else "⬜"

    checked = col.checkbox(
        f"{cache_icon} {method}",
        value=(method in default_methods),
        key=f"feat_{method}",
    )
    col.caption(DESCRIPTIONS.get(method, ""))

    if checked:
        selected_methods.append(method)

# ── Cache status summary ─────────────────────────────────────────────────
st.subheader("Cache Status")
cache_data = []
for method in FEATURE_METHODS:
    cp = _cache_path(method)
    cached = cp.exists()
    size_mb = cp.stat().st_size / (1024 * 1024) if cached else 0.0
    cache_data.append({
        "Method": method,
        "Cached": "Yes" if cached else "No",
        "Size (MB)": f"{size_mb:.2f}" if cached else "—",
    })

import pandas as pd
cache_df = pd.DataFrame(cache_data)
st.dataframe(cache_df, width="stretch", hide_index=True)

# ── Extract / Load button ────────────────────────────────────────────────
st.subheader("Extract Features")

if not selected_methods:
    st.warning("Please select at least one feature method.")
else:
    st.markdown(f"**Selected methods:** {', '.join(selected_methods)}")

    if st.button("Extract / Load Features", type="primary"):
        log.info(f"Feature extraction started: {selected_methods}")

        with st.spinner("Loading images and extracting features..."):
            # Determine which image types we need
            needs_gray = any(m in ["HOG", "LBP", "Gabor"] for m in selected_methods)
            needs_color = any(
                m in ["Landmarks", "ConvNeXt-Tiny", "CLIP-ViT-B/32", "InsightFace"]
                for m in selected_methods
            )

            df = load_annotations()
            labels = df["label_int"].values

            # Load images as needed
            if needs_gray:
                images_gray, _, _ = load_images_for_traditional()
            else:
                images_gray = np.empty((0,))

            if needs_color:
                images_color, _, _ = load_images_for_deep()
            else:
                images_color = np.empty((0,))

            features = extract_features(
                images_gray=images_gray,
                images_color=images_color,
                methods=selected_methods,
                cache=True,
            )

        # Store in session state
        st.session_state["features"] = features
        st.session_state["labels"] = labels
        st.session_state["feature_methods"] = selected_methods

        log.info(f"Features extracted: shape={features.shape}, methods={selected_methods}")
        st.success(f"Features extracted successfully! Shape: {features.shape}")

# ── Show feature statistics if available ─────────────────────────────────
if "features" in st.session_state:
    st.subheader("Feature Statistics")
    feat = st.session_state["features"]
    methods_used = st.session_state.get("feature_methods", [])

    st.markdown(f"**Methods:** {', '.join(methods_used)}")

    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
    stat_col1.metric("Shape", f"{feat.shape[0]} x {feat.shape[1]}")
    stat_col2.metric("Mean", f"{feat.mean():.4f}")
    stat_col3.metric("Std", f"{feat.std():.4f}")
    stat_col4.metric("Min", f"{feat.min():.4f}")
    stat_col5.metric("Max", f"{feat.max():.4f}")

log.info("Feature engineering page viewed")
