# Label Upload & Evaluation Metrics -- Design Spec

**Date:** 2026-04-01
**Status:** Approved
**Scope:** Add optional ground-truth label upload to Batch Prediction page with full evaluation metrics.

## Summary

Users can optionally upload a CSV of ground-truth labels alongside their images on the Batch Prediction page. When provided, the app computes evaluation metrics (accuracy, F1, precision, recall, FPR, specificity, confusion matrix, ROC/AUC) comparing model predictions against the ground truth. The feature is purely additive -- no labels uploaded means the page behaves exactly as before.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Filename matching | Case-insensitive | Consistent with existing `_find_image_file()` pattern |
| Unmatched files | Evaluate intersection + show all predictions | Most user-friendly; no data lost |
| Metric set | Extended (Accuracy, F1, Precision, Recall, FPR, Specificity, CM, ROC/AUC) | Matches existing Results page (page 4) |
| Results placement | Below existing predictions | Preserves current flow; additive only |
| Implementation | `compute_metrics()` helper in evaluation.py | Reusable; keeps UI file focused on rendering |

## Files Modified

1. **`src/evaluation.py`** -- Add `compute_metrics()` function
2. **`pages/6_Batch_Prediction.py`** -- Add label uploader, matching logic, evaluation display

## Design

### 1. Label Upload UI

- Optional `st.file_uploader` placed below the existing image uploader
- Label: "Upload ground-truth labels (optional)"
- Help text explains expected format
- Accepts `.csv` only
- Expected CSV format: two columns, no header -- `filename,label`
- Valid labels: `happy` or `neutral` (case-insensitive, e.g. `Happy`, `NEUTRAL` accepted)
- If no CSV uploaded, zero change to existing behavior

### 2. Filename Matching

- Normalize both sides with `.lower()`
- Build dict from CSV: `{filename.lower(): label_int}` where happy=1, neutral=0
- For each uploaded image, look up ground-truth by `image_filename.lower()`
- Three categories tracked:
  - **Matched**: image has both prediction and ground-truth
  - **Unmatched images**: uploaded images with no CSV row
  - **Unmatched labels**: CSV rows with no uploaded image
- Warning shown if either unmatched set is non-empty, listing filenames
- Invalid labels (not happy/neutral) produce `st.error()` and skip evaluation

### 3. `compute_metrics()` Helper

Location: `src/evaluation.py`

```
compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict
```

**Inputs:**
- `y_true`: 1D array of 0/1 ground-truth labels
- `y_pred`: 1D array of 0/1 predictions
- `y_prob`: optional 1D array of positive-class probabilities

**Returns dict with:**
- `accuracy`: float
- `f1`: float
- `precision`: float
- `recall`: float
- `fpr`: float
- `specificity`: float (1 - FPR)
- `confusion_matrix`: 2x2 list, labels [0, 1]
- `roc`: dict with `fpr`, `tpr`, `auc` keys (None if y_prob not provided)
- `n_samples`: int

**Characteristics:**
- Pure computation, no side effects, no logging
- Uses existing sklearn.metrics imports plus `accuracy_score`
- `cross_validate()` remains untouched

### 4. Evaluation Display

Rendered below the per-image results grid, only when labels uploaded and >= 1 match.

**4a. Header & match summary**
- `st.subheader("Evaluation Metrics")`
- Info box: "Evaluated on X of Y uploaded images (Z had no matching label)"

**4b. Metric cards**
- `st.columns(6)` with `st.metric()`: Accuracy, F1, Precision, Recall, FPR, Specificity

**4c. Confusion matrix**
- Plotly heatmap, axes: "Predicted" (x) vs "Actual" (y)
- Cells labeled with counts
- Green tones for correct, red tones for errors

**4d. ROC curve (conditional)**
- Only shown when `y_prob` is available (model supports predict_proba)
- Plotly line chart, AUC in title, diagonal reference line
- Skipped with `st.info()` if single-class ground truth

**4e. Per-image grid update**
- Matched images: checkmark or X next to label indicating correct/incorrect
- Unmatched images: display as today (prediction only)

### 5. Error Handling

| Scenario | Behavior |
|----------|----------|
| CSV parse error (wrong columns, encoding) | `st.error()`, skip evaluation, predictions display normally |
| Invalid labels in CSV | `st.error()` listing invalid values, skip evaluation |
| Zero filename matches | `st.warning()`, skip evaluation |
| All images matched | No warning, show metrics |
| Single-class ground truth | Metrics compute, ROC skipped with `st.info()` explanation |

**Principle:** Label upload errors never disrupt prediction display.
