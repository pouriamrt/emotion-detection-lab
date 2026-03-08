import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.auth import require_auth
from src.config import METADATA_PATH
from src.logger import get_logger

user = require_auth()
log = get_logger(user=user["name"], context="comparison")

with st.sidebar:
    st.markdown(f"**Logged in as:** {user['name']}")
    st.caption(user["email"])

st.title("Model Comparison")
st.markdown("Compare all trained models on a leaderboard with charts and rankings.")

# ── Gather all results ───────────────────────────────────────────────────
all_models = {}

# Pre-trained models from metadata.json
if METADATA_PATH.exists():
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    for key, info in metadata["models"].items():
        model_type = info.get("type", "traditional")
        all_models[key] = {
            "Model": key,
            "F1": info["f1"],
            "Precision": info["precision"],
            "Recall": info["recall"],
            "FPR": info["fpr"],
            "Mark": info["performance_mark"],
            "Type": model_type,
            "Source": "Pre-trained",
        }

# Session results from custom training
session_results = st.session_state.get("training_results", {})
for key, info in session_results.items():
    all_models[f"[Custom] {key}"] = {
        "Model": f"[Custom] {key}",
        "F1": info["f1"],
        "Precision": info["precision"],
        "Recall": info["recall"],
        "FPR": info["fpr"],
        "Mark": info["performance_mark"],
        "Type": info.get("type", "traditional"),
        "Source": "Custom",
    }

if not all_models:
    st.info(
        "No models available for comparison. Train models on the **Model Training** page "
        "or run `python -m scripts.train_all` for pre-trained models."
    )
    st.stop()

# Build leaderboard DataFrame
leaderboard = pd.DataFrame(list(all_models.values()))
leaderboard = leaderboard.sort_values("F1", ascending=False).reset_index(drop=True)
leaderboard.index = leaderboard.index + 1  # 1-based ranking
leaderboard.index.name = "Rank"

# ── Best model highlight ─────────────────────────────────────────────────
best_row = leaderboard.iloc[0]
st.success(f"**Best Model:** {best_row['Model']} — F1 = {best_row['F1']:.4f} | Mark = {best_row['Mark']:.1f}/35")

# ── Filter by model type ─────────────────────────────────────────────────
available_types = sorted(leaderboard["Type"].unique().tolist())
selected_types = st.multiselect(
    "Filter by model type",
    available_types,
    default=available_types,
)

if selected_types:
    filtered = leaderboard[leaderboard["Type"].isin(selected_types)]
else:
    filtered = leaderboard

# ── Leaderboard table ────────────────────────────────────────────────────
st.subheader("Leaderboard")


def _highlight_best_f1(row):
    """Highlight the row with the best F1 score in the filtered view."""
    if row["F1"] == filtered["F1"].max():
        return ["background-color: #d4edda"] * len(row)
    return [""] * len(row)


styled = filtered.style.apply(_highlight_best_f1, axis=1).format({
    "F1": "{:.4f}",
    "Precision": "{:.4f}",
    "Recall": "{:.4f}",
    "FPR": "{:.4f}",
    "Mark": "{:.1f}",
})
st.dataframe(styled, use_container_width=True)

# ── F1 comparison bar chart ──────────────────────────────────────────────
st.subheader("F1 Score Comparison")

color_map = {"traditional": "#3498db", "deep": "#e74c3c"}
fig_bar = px.bar(
    filtered,
    x="Model",
    y="F1",
    color="Type",
    color_discrete_map=color_map,
    title="F1 Score by Model",
    text=filtered["F1"].apply(lambda x: f"{x:.4f}"),
)
fig_bar.update_layout(
    xaxis_title="Model",
    yaxis_title="F1 Score",
    yaxis_range=[0, 1],
    xaxis_tickangle=-45,
)
fig_bar.update_traces(textposition="outside")
st.plotly_chart(fig_bar, use_container_width=True)

# ── Radar chart for top 5 models ─────────────────────────────────────────
st.subheader("Radar Chart — Top 5 Models")

top_n = min(5, len(filtered))
top_models = filtered.head(top_n)

categories = ["F1", "Precision", "Recall", "1 - FPR"]

fig_radar = go.Figure()

colors = px.colors.qualitative.Set2
for i, (_, row) in enumerate(top_models.iterrows()):
    values = [
        row["F1"],
        row["Precision"],
        row["Recall"],
        1.0 - row["FPR"],
    ]
    # Close the radar chart by repeating the first value
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig_radar.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        name=row["Model"],
        line=dict(color=colors[i % len(colors)]),
        opacity=0.7,
    ))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1]),
    ),
    title="Top Models — Multi-Metric Radar",
    showlegend=True,
)
st.plotly_chart(fig_radar, use_container_width=True)

# ── Summary statistics ───────────────────────────────────────────────────
st.subheader("Summary")
scol1, scol2, scol3, scol4 = st.columns(4)
scol1.metric("Total Models", len(filtered))
scol2.metric("Best F1", f"{filtered['F1'].max():.4f}")
scol3.metric("Avg F1", f"{filtered['F1'].mean():.4f}")
scol4.metric("F1 Std", f"{filtered['F1'].std():.4f}")

log.info(f"Model comparison page viewed: {len(all_models)} models")
