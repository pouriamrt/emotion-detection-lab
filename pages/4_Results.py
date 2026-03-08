import streamlit as st
import json
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from src.auth import require_auth
from src.config import METADATA_PATH
from src.logger import get_logger

user = require_auth()
log = get_logger(user=user["name"], context="results")

with st.sidebar:
    st.markdown(f"**Logged in as:** {user['name']}")
    st.caption(user["email"])

st.title("Results")
st.markdown("View detailed metrics, confusion matrix, and ROC curves for a selected model run.")

# ── Gather all available results ─────────────────────────────────────────
all_results = {}

# Pre-trained results from metadata.json
if METADATA_PATH.exists():
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    for key, info in metadata["models"].items():
        all_results[f"[Pre-trained] {key}"] = info

# Session results from custom training
session_results = st.session_state.get("training_results", {})
for key, info in session_results.items():
    all_results[f"[Custom] {key}"] = info

if not all_results:
    st.info(
        "No results available. Train a model on the **Model Training** page "
        "or run `python -m scripts.train_all` for pre-trained models."
    )
    st.stop()

# ── Model selection ──────────────────────────────────────────────────────
selected_key = st.selectbox("Select Model Run", list(all_results.keys()))
results = all_results[selected_key]
log.info(f"Viewing results for: {selected_key}")

# ── Metrics row ──────────────────────────────────────────────────────────
st.subheader("Metrics")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("F1 Score", f"{results['f1']:.4f}")
m2.metric("Precision", f"{results['precision']:.4f}")
m3.metric("Recall", f"{results['recall']:.4f}")
m4.metric("FPR", f"{results['fpr']:.4f}")
m5.metric("Mark", f"{results['performance_mark']:.1f}/35")

# Extra info if available
extra_cols = st.columns(3)
if "n_samples" in results:
    extra_cols[0].metric("Samples", results["n_samples"])
if "feature_dim" in results:
    extra_cols[1].metric("Feature Dim", results["feature_dim"])
if "elapsed_seconds" in results:
    extra_cols[2].metric("Time", f"{results['elapsed_seconds']:.1f}s")

# ── Confusion matrix heatmap ─────────────────────────────────────────────
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
fig_cm.update_yaxes(autorange="reversed")
st.plotly_chart(fig_cm, use_container_width=True)

# ── Per-fold results ─────────────────────────────────────────────────────
if "fold_results" in results and results["fold_results"]:
    st.subheader("Per-Fold Results")
    fold_data = results["fold_results"]
    fold_df = pd.DataFrame(fold_data)

    display_cols = [c for c in ["fold", "f1", "precision", "recall", "fpr"] if c in fold_df.columns]
    st.dataframe(fold_df[display_cols], use_container_width=True, hide_index=True)

    # Fold F1 bar chart
    fig_fold = go.Figure()
    fig_fold.add_trace(go.Bar(
        x=[f"Fold {r['fold']}" for r in fold_data],
        y=[r["f1"] for r in fold_data],
        marker_color="#3498db",
        text=[f"{r['f1']:.4f}" for r in fold_data],
        textposition="auto",
    ))
    fig_fold.update_layout(
        title="F1 Score per Fold",
        xaxis_title="Fold",
        yaxis_title="F1 Score",
        yaxis_range=[0, 1],
    )
    st.plotly_chart(fig_fold, use_container_width=True)

    # Summary statistics
    f1_values = [r["f1"] for r in fold_data]
    import numpy as np
    st.markdown(
        f"**F1 Mean:** {np.mean(f1_values):.4f} | "
        f"**F1 Std:** {np.std(f1_values):.4f} | "
        f"**F1 Min:** {np.min(f1_values):.4f} | "
        f"**F1 Max:** {np.max(f1_values):.4f}"
    )

# ── ROC curve ────────────────────────────────────────────────────────────
if "roc" in results and results["roc"]:
    roc = results["roc"]
    if roc.get("fpr") and roc.get("tpr"):
        st.subheader("ROC Curve")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=roc["fpr"],
            y=roc["tpr"],
            mode="lines",
            name=f"ROC (AUC = {roc.get('auc', 0):.4f})",
            line=dict(color="#2ecc71", width=2),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(color="gray", dash="dash"),
        ))
        fig_roc.update_layout(
            title=f"ROC Curve (AUC = {roc.get('auc', 0):.4f})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1.05],
        )
        st.plotly_chart(fig_roc, use_container_width=True)

log.info(f"Results page viewed for: {selected_key}")
