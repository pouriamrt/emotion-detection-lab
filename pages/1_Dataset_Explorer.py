import streamlit as st
import plotly.express as px
import cv2
from src.auth import require_auth
from src.data_loader import load_annotations, _find_image_file
from src.logger import get_logger

user = require_auth()
log = get_logger(user=user["name"], context="dataset")

with st.sidebar:
    st.markdown(f"**Logged in as:** {user['name']}")
    st.caption(user["email"])

st.title("Dataset Explorer")
st.markdown("Browse the SMILE_PLUS dataset: class distribution, sample images, and image statistics.")

# ── Load annotations ─────────────────────────────────────────────────────
df = load_annotations()
log.info(f"Dataset explorer loaded with {len(df)} annotations")

# ── Metrics row ──────────────────────────────────────────────────────────
total = len(df)
happy_count = int((df["label"] == "happy").sum())
neutral_count = int((df["label"] == "neutral").sum())

col1, col2, col3 = st.columns(3)
col1.metric("Total Images", total)
col2.metric("Happy", happy_count)
col3.metric("Neutral", neutral_count)

# ── Class distribution pie chart ─────────────────────────────────────────
st.subheader("Class Distribution")
dist = df["label"].value_counts().reset_index()
dist.columns = ["Label", "Count"]
fig_pie = px.pie(
    dist,
    names="Label",
    values="Count",
    color="Label",
    color_discrete_map={"happy": "#2ecc71", "neutral": "#3498db"},
    title="Happy vs Neutral Distribution",
)
fig_pie.update_traces(textinfo="percent+value")
st.plotly_chart(fig_pie, width="stretch")

# ── Sample images ────────────────────────────────────────────────────────
st.subheader("Sample Images")
n_samples = st.slider("Number of samples per class", min_value=1, max_value=10, value=4)

for label in ["happy", "neutral"]:
    st.markdown(f"**{label.capitalize()}**")
    subset = df[df["label"] == label].sample(n=min(n_samples, len(df[df["label"] == label])), random_state=42)
    cols = st.columns(n_samples)
    for i, (_, row) in enumerate(subset.iterrows()):
        if i >= n_samples:
            break
        img_path = _find_image_file(row["filename"])
        if img_path is not None:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                cols[i].image(img_rgb, caption=row["filename"], width="stretch")
            else:
                cols[i].warning(f"Cannot read: {row['filename']}")
        else:
            cols[i].warning(f"Not found: {row['filename']}")

# ── Image statistics ─────────────────────────────────────────────────────
st.subheader("Image Statistics")
st.markdown("Statistics from a sample of images in the dataset.")

sample_stats = []
stat_sample = df.sample(n=min(50, len(df)), random_state=42)
for _, row in stat_sample.iterrows():
    img_path = _find_image_file(row["filename"])
    if img_path is not None:
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            channels = img.shape[2] if len(img.shape) == 3 else 1
            mode = "Color (BGR)" if channels == 3 else "Grayscale"
            ext = img_path.suffix.lower()
            sample_stats.append({
                "filename": row["filename"],
                "width": w,
                "height": h,
                "channels": channels,
                "mode": mode,
                "format": ext,
            })

if sample_stats:
    import pandas as pd
    stats_df = pd.DataFrame(sample_stats)

    scol1, scol2, scol3, scol4 = st.columns(4)
    scol1.metric("Avg Width", f"{stats_df['width'].mean():.0f} px")
    scol2.metric("Avg Height", f"{stats_df['height'].mean():.0f} px")
    scol3.metric("Most Common Mode", stats_df["mode"].mode().iloc[0])
    scol4.metric("Most Common Format", stats_df["format"].mode().iloc[0])

    with st.expander("Detailed image statistics table"):
        st.dataframe(stats_df, width="stretch")
else:
    st.warning("Could not load any images for statistics.")

log.info(f"Dataset explorer viewed: {total} images, {happy_count} happy, {neutral_count} neutral")
