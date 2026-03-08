import streamlit as st
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from src.auth import require_auth
from src.config import TRADITIONAL_MODELS, DEEP_MODELS, FEATURE_METHODS, FEATURES_DIR
from src.models import MODEL_REGISTRY, DEEP_MODEL_REGISTRY, create_model, create_deep_model
from src.evaluation import cross_validate
from src.data_loader import load_annotations, load_images_for_traditional, load_images_for_deep
from src.features import extract_features, _cache_path
from src.logger import get_logger

user = require_auth()
log = get_logger(user=user["name"], context="training")

with st.sidebar:
    st.markdown(f"**Logged in as:** {user['name']}")
    st.caption(user["email"])

# ── Helper: display results inline ───────────────────────────────────────

def _display_results(results: dict):
    """Display training results with metrics and confusion matrix."""
    st.markdown("---")
    st.subheader("Training Results")

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("F1 Score", f"{results['f1']:.4f}")
    m2.metric("Precision", f"{results['precision']:.4f}")
    m3.metric("Recall", f"{results['recall']:.4f}")
    m4.metric("FPR", f"{results['fpr']:.4f}")
    m5.metric("Mark", f"{results['performance_mark']:.1f}/35")

    # Additional info
    info1, info2, info3 = st.columns(3)
    info1.metric("Samples", results["n_samples"])
    info2.metric("Feature Dim", results["feature_dim"])
    info3.metric("Time", f"{results['elapsed_seconds']:.1f}s")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = results["confusion_matrix"]
    labels = ["Neutral", "Happy"]
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Blues",
        showscale=True,
    )
    fig_cm.update_layout(
        title="Aggregated Confusion Matrix (across all folds)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    # Fix: annotated heatmap reverses y-axis by default
    fig_cm.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_cm, use_container_width=True)

    # Per-fold results
    st.subheader("Per-Fold Results")
    fold_df = pd.DataFrame(results["fold_results"])
    st.dataframe(fold_df[["fold", "f1", "precision", "recall", "fpr"]], use_container_width=True, hide_index=True)

    # Fold F1 bar chart
    fig_fold = go.Figure()
    fig_fold.add_trace(go.Bar(
        x=[f"Fold {r['fold']}" for r in results["fold_results"]],
        y=[r["f1"] for r in results["fold_results"]],
        marker_color="#3498db",
    ))
    fig_fold.update_layout(
        title="F1 Score per Fold",
        xaxis_title="Fold",
        yaxis_title="F1 Score",
        yaxis_range=[0, 1],
    )
    st.plotly_chart(fig_fold, use_container_width=True)


# ── Main page ────────────────────────────────────────────────────────────

st.title("Model Training")
st.markdown("Select a model, tune hyperparameters, and run 10-fold cross-validation.")

# Initialize session state for training results
if "training_results" not in st.session_state:
    st.session_state["training_results"] = {}

# ── Model type selection ─────────────────────────────────────────────────
model_type = st.radio(
    "Model Type",
    ["Traditional ML", "Deep Learning (Linear Probe)"],
    horizontal=True,
)

if model_type == "Traditional ML":
    model_name = st.selectbox("Select Model", TRADITIONAL_MODELS)
    registry = MODEL_REGISTRY
    param_grid = registry[model_name]["param_grid"]

    # Feature selection for traditional models
    st.subheader("Feature Selection")
    if "features" in st.session_state and "feature_methods" in st.session_state:
        st.info(f"Using features from Feature Engineering page: {', '.join(st.session_state['feature_methods'])}")
        use_session_features = st.checkbox("Use session features", value=True)
    else:
        use_session_features = False

    if not use_session_features:
        available_methods = [m for m in FEATURE_METHODS if _cache_path(m).exists()]
        if not available_methods:
            st.warning(
                "No cached features found. Please go to the Feature Engineering page "
                "to extract features first, or run `python -m scripts.train_all`."
            )
            st.stop()
        selected_feature_methods = st.multiselect(
            "Select feature methods",
            available_methods,
            default=available_methods[:2] if len(available_methods) >= 2 else available_methods,
        )
        if not selected_feature_methods:
            st.warning("Please select at least one feature method.")
            st.stop()
    else:
        selected_feature_methods = st.session_state["feature_methods"]

else:
    model_name = st.selectbox("Select Model", DEEP_MODELS)
    registry = DEEP_MODEL_REGISTRY
    param_grid = registry[model_name]["param_grid"]
    feature_source = registry[model_name]["feature_source"]
    st.info(f"This model uses **{feature_source}** features.")

# ── Hyperparameter tuning ────────────────────────────────────────────────
st.subheader("Hyperparameters")
custom_params = {}

for param_name, param_info in param_grid.items():
    ptype = param_info["type"]
    desc = param_info.get("desc", param_name)

    if ptype == "float":
        is_log = param_info.get("log", False)
        fmt = "%.5f" if is_log else "%.4f"
        value = st.slider(
            f"{param_name} — {desc}",
            min_value=float(param_info["min"]),
            max_value=float(param_info["max"]),
            value=float(param_info["default"]),
            format=fmt,
            key=f"param_{model_name}_{param_name}",
        )
        custom_params[param_name] = value

    elif ptype == "int":
        step = param_info.get("step", 1)
        value = st.slider(
            f"{param_name} — {desc}",
            min_value=int(param_info["min"]),
            max_value=int(param_info["max"]),
            value=int(param_info["default"]),
            step=int(step),
            key=f"param_{model_name}_{param_name}",
        )
        custom_params[param_name] = value

    elif ptype == "select":
        options = param_info["options"]
        default_val = param_info["default"]
        default_idx = options.index(default_val) if default_val in options else 0
        value = st.selectbox(
            f"{param_name} — {desc}",
            options=options,
            index=default_idx,
            key=f"param_{model_name}_{param_name}",
        )
        custom_params[param_name] = value

# ── Train button ─────────────────────────────────────────────────────────
st.markdown("---")
if st.button("Train Model (10-Fold CV)", type="primary"):
    log.info(f"Training started: {model_name} with params {custom_params}")

    with st.spinner(f"Training {model_name} with 10-fold cross-validation..."):
        if model_type == "Traditional ML":
            model = create_model(model_name, custom_params)

            # Get features
            if use_session_features and "features" in st.session_state:
                X = st.session_state["features"]
                y = st.session_state["labels"]
            else:
                # Load from cache
                df = load_annotations()
                y = df["label_int"].values

                needs_gray = any(m in ["HOG", "LBP", "Gabor"] for m in selected_feature_methods)
                needs_color = any(
                    m in ["Landmarks", "ConvNeXt-Tiny", "CLIP-ViT-B/32", "InsightFace"]
                    for m in selected_feature_methods
                )

                images_gray = np.empty((0,))
                images_color = np.empty((0,))
                if needs_gray:
                    images_gray, _, _ = load_images_for_traditional()
                if needs_color:
                    images_color, _, _ = load_images_for_deep()

                X = extract_features(
                    images_gray=images_gray,
                    images_color=images_color,
                    methods=selected_feature_methods,
                    cache=True,
                )

            results = cross_validate(model, X, y, user=user["name"])
            features_used = selected_feature_methods

        else:
            # Deep learning
            model, feat_source = create_deep_model(model_name, custom_params)

            # Check if feature source is cached
            cache_file = _cache_path(feat_source)
            if cache_file.exists():
                X = np.load(str(cache_file))
                df = load_annotations()
                y = df["label_int"].values
            else:
                st.warning(f"Feature source '{feat_source}' not cached. Extracting features...")
                df = load_annotations()
                y = df["label_int"].values
                images_color, _, _ = load_images_for_deep()
                images_gray = np.empty((0,))
                X = extract_features(
                    images_gray=images_gray,
                    images_color=images_color,
                    methods=[feat_source],
                    cache=True,
                )

            results = cross_validate(model, X, y, scale=False, user=user["name"])
            features_used = [feat_source]

    # Store results
    run_key = f"{model_name} ({', '.join(features_used)})"
    results["features"] = features_used
    results["type"] = "traditional" if model_type == "Traditional ML" else "deep"
    results["model_display_name"] = model_name
    st.session_state["training_results"][run_key] = results

    log.info(f"Training complete: {run_key} — F1={results['f1']:.4f}")
    st.success(f"Training complete! F1 = {results['f1']:.4f}")

    _display_results(results)

log.info("Model training page viewed")
