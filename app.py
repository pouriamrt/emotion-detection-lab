import streamlit as st
import json
from src.auth import check_auth, logout
from src.config import METADATA_PATH
from src.logger import get_logger

st.set_page_config(
    page_title="Emotion Detection Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

is_auth, user = check_auth()

if not is_auth:
    st.title("Emotion Detection Lab")
    st.markdown("### Binary Classification: Happy vs Neutral")
    st.markdown("Please log in with Google to continue.")
    st.stop()

log = get_logger(user=user["name"], context="general")

with st.sidebar:
    st.markdown(f"**Logged in as:** {user['name']}")
    st.caption(user["email"])
    if st.button("Logout"):
        logout()

st.title("Emotion Detection Lab")

# Description of the app and how to use it
st.markdown("""
Welcome to the **Emotion Detection Lab** — a machine learning workbench for classifying
**happy** vs **neutral** facial expressions on the SMILE_PLUS dataset.

### How to Use

1. **Dataset Explorer** — Browse the dataset, view sample images and class distribution
2. **Feature Engineering** — Select and extract features (traditional + deep embeddings)
3. **Model Training** — Choose a classifier, tune hyperparameters, run 10-fold cross-validation
4. **Results** — View detailed metrics, confusion matrix, and ROC curves
5. **Model Comparison** — Compare all models on a leaderboard with the best model highlighted

### Pre-Trained Models

All models come **pre-trained with optimized hyperparameters**. You can view their results
immediately or re-train with your own custom settings.
""")

# Show pre-trained summary if available
if METADATA_PATH.exists():
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    best_key = metadata["best_model"]
    best_info = metadata["models"][best_key]

    st.success(f"**Best Model:** {best_key} — F1 = {metadata['best_f1']:.4f}")
    col1, col2 = st.columns(2)
    col1.metric("Total Models Trained", len(metadata["models"]))
    col2.metric("Best F1 Score", f"{metadata['best_f1']:.4f}")
else:
    st.info("No pre-trained models found. Run `python -m scripts.train_all` to train all models, or use the Model Training page.")

log.info("Home page viewed")
