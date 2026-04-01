# Label Upload & Evaluation Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional ground-truth label CSV upload to the Batch Prediction page, computing evaluation metrics when labels are provided.

**Architecture:** New `compute_metrics()` pure function in `src/evaluation.py` handles metric computation. `pages/6_Batch_Prediction.py` gets a second file uploader, CSV parsing/matching logic, and a metrics display section appended below existing results.

**Tech Stack:** Python 3.13, Streamlit, sklearn.metrics, Plotly, numpy, pandas

**Spec:** `docs/superpowers/specs/2026-04-01-label-upload-evaluation-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/evaluation.py` | Add `compute_metrics()` function (lines appended after `cross_validate`) |
| Modify | `pages/6_Batch_Prediction.py` | Add label uploader, matching, evaluation display |
| Create | `tests/test_evaluation.py` | Unit tests for `compute_metrics()` |
| Create | `tests/__init__.py` | Make tests a package |

---

### Task 1: Add `compute_metrics()` to `src/evaluation.py`

**Files:**
- Modify: `src/evaluation.py` (append after line 167)
- Create: `tests/__init__.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Create test directory and write failing tests**

Create `tests/__init__.py` (empty file) and `tests/test_evaluation.py`:

```python
"""Tests for compute_metrics() in src/evaluation.py."""

import numpy as np
import pytest

from src.evaluation import compute_metrics


class TestComputeMetrics:
    """Unit tests for compute_metrics."""

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        result = compute_metrics(y_true, y_pred)

        assert result["accuracy"] == 1.0
        assert result["f1"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["fpr"] == 0.0
        assert result["specificity"] == 1.0
        assert result["confusion_matrix"] == [[2, 0], [0, 2]]
        assert result["roc"] is None
        assert result["n_samples"] == 4

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        result = compute_metrics(y_true, y_pred)

        assert result["accuracy"] == 0.0
        assert result["f1"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["fpr"] == 1.0
        assert result["specificity"] == 0.0
        assert result["confusion_matrix"] == [[0, 2], [2, 0]]

    def test_mixed_predictions(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 0])
        result = compute_metrics(y_true, y_pred)

        # TP=2, FP=1, FN=1, TN=2
        assert result["accuracy"] == pytest.approx(4 / 6)
        assert result["recall"] == pytest.approx(2 / 3)
        assert result["precision"] == pytest.approx(2 / 3)
        assert result["fpr"] == pytest.approx(1 / 3)
        assert result["specificity"] == pytest.approx(2 / 3)
        assert result["confusion_matrix"] == [[2, 1], [1, 2]]
        assert result["n_samples"] == 6

    def test_with_probabilities(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.8, 0.9])
        result = compute_metrics(y_true, y_pred, y_prob=y_prob)

        assert result["roc"] is not None
        assert "fpr" in result["roc"]
        assert "tpr" in result["roc"]
        assert "auc" in result["roc"]
        assert result["roc"]["auc"] == pytest.approx(1.0)

    def test_without_probabilities_roc_is_none(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        result = compute_metrics(y_true, y_pred)

        assert result["roc"] is None

    def test_single_class_ground_truth(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 1, 0])
        result = compute_metrics(y_true, y_pred)

        assert result["n_samples"] == 3
        assert result["recall"] == pytest.approx(2 / 3)
        # FPR undefined (no negatives), should be 0.0
        assert result["fpr"] == 0.0

    def test_single_class_with_prob_roc_none(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 1, 1])
        y_prob = np.array([0.8, 0.9, 0.7])
        result = compute_metrics(y_true, y_pred, y_prob=y_prob)

        # ROC is undefined with single class -- should be None
        assert result["roc"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_metrics' from 'src.evaluation'`

- [ ] **Step 3: Implement `compute_metrics()` in `src/evaluation.py`**

Add `accuracy_score` to the existing sklearn imports at line 12, then append this function after line 167:

```python
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """Compute evaluation metrics comparing predictions to ground truth.

    Args:
        y_true: 1D array of 0/1 ground-truth labels.
        y_pred: 1D array of 0/1 predictions.
        y_prob: Optional 1D array of positive-class probabilities.

    Returns:
        Dict with accuracy, f1, precision, recall, fpr, specificity,
        confusion_matrix, roc (or None), and n_samples.
    """
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr_val = float(fp / max(fp + tn, 1))
    spec = 1.0 - fpr_val

    roc_data = None
    if y_prob is not None and len(set(y_true)) > 1:
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob, pos_label=1)
        roc_data = {
            "fpr": fpr_arr.tolist(),
            "tpr": tpr_arr.tolist(),
            "auc": float(auc(fpr_arr, tpr_arr)),
        }

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "fpr": fpr_val,
        "specificity": spec,
        "confusion_matrix": cm.tolist(),
        "roc": roc_data,
        "n_samples": len(y_true),
    }
```

The import line at the top of the file (line 12-18) should become:

```python
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py tests/test_evaluation.py src/evaluation.py
git commit -m "feat: add compute_metrics() helper to evaluation.py"
```

---

### Task 2: Add label CSV uploader to Batch Prediction page

**Files:**
- Modify: `pages/6_Batch_Prediction.py` (after line 54, below image uploader)

- [ ] **Step 1: Add the label file uploader and CSV parsing logic**

After the existing image uploader (line 54), add:

```python
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
```

- [ ] **Step 2: Run the app to verify the uploader appears and existing flow is intact**

Run: `uv run streamlit run app.py`
Verify: The new "Upload ground-truth labels (optional)" uploader appears below the image uploader. Uploading images without a CSV works exactly as before.

- [ ] **Step 3: Commit**

```bash
git add pages/6_Batch_Prediction.py
git commit -m "feat: add optional label CSV uploader to batch prediction page"
```

---

### Task 3: Add filename matching logic

**Files:**
- Modify: `pages/6_Batch_Prediction.py` (after predictions are computed, around line 127)

- [ ] **Step 1: Add matching logic after predictions**

After the existing `confidences` computation (after current line 126) and before the Summary section, add:

```python
# ── Match ground-truth labels to predictions ───────────────────────────
matched_gt: dict[int, int] | None = None  # idx -> ground-truth label_int
if ground_truth is not None:
    uploaded_lower = {fn.lower(): i for i, fn in enumerate(filenames)}
    matched_gt = {}
    unmatched_images = []
    unmatched_labels = []

    for fn_lower, gt_label in ground_truth.items():
        if fn_lower in uploaded_lower:
            matched_gt[uploaded_lower[fn_lower]] = gt_label
        else:
            # Find original-case filename from CSV for display
            unmatched_labels.append(fn_lower)

    for i, fn in enumerate(filenames):
        if i not in matched_gt:
            unmatched_images.append(fn)

    if len(matched_gt) == 0:
        st.warning("No filenames matched between uploaded images and label file.")
        matched_gt = None
    else:
        if unmatched_images:
            st.warning(
                f"{len(unmatched_images)} image(s) had no matching label: "
                f"{', '.join(unmatched_images[:10])}"
                + (f" and {len(unmatched_images) - 10} more" if len(unmatched_images) > 10 else "")
            )
        if unmatched_labels:
            st.warning(
                f"{len(unmatched_labels)} label(s) had no matching image: "
                f"{', '.join(unmatched_labels[:10])}"
                + (f" and {len(unmatched_labels) - 10} more" if len(unmatched_labels) > 10 else "")
            )
```

- [ ] **Step 2: Verify matching works by running the app**

Run: `uv run streamlit run app.py`
Test: Upload images + a CSV with some matching and some non-matching filenames. Verify warnings appear correctly.

- [ ] **Step 3: Commit**

```bash
git add pages/6_Batch_Prediction.py
git commit -m "feat: add case-insensitive filename matching for label upload"
```

---

### Task 4: Add per-image correctness indicators to grid

**Files:**
- Modify: `pages/6_Batch_Prediction.py` (in the per-image results grid section, around line 154-173)

- [ ] **Step 1: Update the per-image grid to show correct/incorrect indicators**

Replace the per-image results section (current lines 153-173) with:

```python
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
            img_rgb = cv2.cvtColor(images_color[idx], cv2.COLOR_BGR2RGB)
            st.image(img_rgb, width=150)
            label = labels[idx]
            color = "green" if label == POSITIVE_LABEL else "blue"
            conf_str = f" ({confidences[idx]:.1%})" if confidences else ""

            # Correctness indicator for matched images
            indicator = ""
            if matched_gt is not None and idx in matched_gt:
                gt_label_int = matched_gt[idx]
                pred_correct = predictions[idx] == gt_label_int
                indicator = " :white_check_mark:" if pred_correct else " :x:"

            st.markdown(
                f"**:{color}[{label.upper()}]**{conf_str}{indicator}",
            )
            st.caption(filenames[idx])
```

- [ ] **Step 2: Verify indicators appear in app**

Run: `uv run streamlit run app.py`
Test: Upload images with a matching label CSV. Verify checkmarks appear for correct predictions, X marks for incorrect ones, and no indicator for unmatched images.

- [ ] **Step 3: Commit**

```bash
git add pages/6_Batch_Prediction.py
git commit -m "feat: add correctness indicators to per-image prediction grid"
```

---

### Task 5: Add evaluation metrics display section

**Files:**
- Modify: `pages/6_Batch_Prediction.py` (append after per-image results, before the final log line)

- [ ] **Step 1: Add the evaluation display section**

Add the following after the per-image results grid and before the final `log.info(...)` line:

```python
# ── Evaluation metrics (only when ground-truth labels provided) ────────
if matched_gt is not None and len(matched_gt) > 0:
    from src.evaluation import compute_metrics

    st.markdown("---")
    st.subheader("Evaluation Metrics")

    n_matched = len(matched_gt)
    n_total = len(filenames)
    n_unmatched = n_total - n_matched
    st.info(
        f"Evaluated on **{n_matched}** of **{n_total}** uploaded images"
        + (f" ({n_unmatched} had no matching label)." if n_unmatched > 0 else ".")
    )

    # Build matched arrays
    matched_indices = sorted(matched_gt.keys())
    y_true = np.array([matched_gt[i] for i in matched_indices])
    y_pred = np.array([predictions[i] for i in matched_indices])
    y_prob_matched = None
    if probas is not None:
        y_prob_matched = np.array([probas[i, 1] for i in matched_indices])

    metrics = compute_metrics(y_true, y_pred, y_prob=y_prob_matched)

    # Metric cards
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    mc2.metric("F1 Score", f"{metrics['f1']:.4f}")
    mc3.metric("Precision", f"{metrics['precision']:.4f}")
    mc4.metric("Recall", f"{metrics['recall']:.4f}")
    mc5.metric("FPR", f"{metrics['fpr']:.4f}")
    mc6.metric("Specificity", f"{metrics['specificity']:.4f}")

    # Confusion matrix heatmap
    cm = np.array(metrics["confusion_matrix"])
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=[NEGATIVE_LABEL, POSITIVE_LABEL],
        y=[NEGATIVE_LABEL, POSITIVE_LABEL],
        text_auto=True,
        color_continuous_scale=[[0, "#2ecc71"], [0.5, "#f5f5f5"], [1, "#e74c3c"]],
        title="Confusion Matrix",
    )
    fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig_cm, use_container_width=True)

    # ROC curve (conditional)
    if metrics["roc"] is not None:
        import plotly.graph_objects as go

        roc = metrics["roc"]
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=roc["fpr"],
                y=roc["tpr"],
                mode="lines",
                name=f"ROC (AUC = {roc['auc']:.4f})",
                line=dict(color="#3498db", width=2),
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random",
                line=dict(color="gray", width=1, dash="dash"),
            )
        )
        fig_roc.update_layout(
            title=f"ROC Curve (AUC = {roc['auc']:.4f})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            showlegend=True,
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    elif y_prob_matched is not None:
        st.info(
            "ROC curve is not available: ground-truth labels contain only one class."
        )

    log.info(
        f"Evaluation metrics computed: F1={metrics['f1']:.4f} "
        f"Acc={metrics['accuracy']:.4f} on {n_matched} matched images"
    )
```

- [ ] **Step 2: Run the app end-to-end**

Run: `uv run streamlit run app.py`
Test with three scenarios:
1. Upload images only (no CSV) -- page behaves exactly as before
2. Upload images + CSV with all matching filenames -- full metrics display appears
3. Upload images + CSV with partial matches -- warnings + metrics on intersection

- [ ] **Step 3: Commit**

```bash
git add pages/6_Batch_Prediction.py
git commit -m "feat: add evaluation metrics display for uploaded ground-truth labels"
```

---

### Task 6: Final integration test and cleanup

**Files:**
- Verify: `pages/6_Batch_Prediction.py`, `src/evaluation.py`, `tests/test_evaluation.py`

- [ ] **Step 1: Run all unit tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run ruff lint and format**

Run: `uv run ruff check src/evaluation.py pages/6_Batch_Prediction.py tests/test_evaluation.py --fix && uv run ruff format src/evaluation.py pages/6_Batch_Prediction.py tests/test_evaluation.py`
Expected: No errors (or auto-fixed)

- [ ] **Step 3: Manual smoke test of error scenarios**

Run: `uv run streamlit run app.py`

Test each error scenario:
- Upload a malformed CSV (e.g., three columns) -- expect `st.error()`, predictions still display
- Upload CSV with invalid label "sad" -- expect `st.error()` listing "sad", predictions still display
- Upload CSV where no filenames match -- expect `st.warning()`, predictions still display
- Upload CSV with all-same-class labels (all "happy") -- metrics display, ROC shows info message

- [ ] **Step 4: Commit any lint fixes**

```bash
git add -u
git commit -m "chore: lint and format label upload feature"
```
