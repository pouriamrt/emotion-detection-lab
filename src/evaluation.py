"""Cross-validation evaluation for emotion detection models.

Provides stratified k-fold evaluation with per-fold and aggregate metrics
including F1, precision, recall, FPR, confusion matrix, and ROC curve data.
"""

import time
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
)

from src.config import N_FOLDS, RANDOM_SEED
from src.logger import get_logger


def cross_validate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = N_FOLDS,
    scale: bool = True,
    user: str = "system",
) -> dict:
    """Run stratified k-fold cross-validation and compute aggregate metrics.

    Args:
        model: An sklearn-compatible estimator (must support fit/predict, and
               optionally predict_proba).
        X: Feature matrix (N, D).
        y: Label vector (N,) with 0/1 values (1 = happy / positive).
        n_folds: Number of CV folds.
        scale: If True, fit StandardScaler on each training fold and transform
               both train and val. Set False for deep models that scale
               internally.
        user: Username for logging context.

    Returns:
        Dict with aggregate and per-fold results, ROC data, and metadata.
    """
    log = get_logger(user=user, context="training")

    model_name = type(model).__name__
    log.info(
        f"Starting {n_folds}-fold CV for {model_name} | "
        f"X shape: {X.shape}, positive rate: {y.mean():.2f}"
    )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []
    all_y_true, all_y_prob = [], []
    start_time = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Optional per-fold scaling
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        # Clone ensures a fresh model each fold
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_val)

        # Probabilities (if available) for ROC
        y_prob = None
        if hasattr(fold_model, "predict_proba"):
            try:
                y_prob = fold_model.predict_proba(X_val)[:, 1]
            except Exception:
                y_prob = None

        # Per-fold metrics
        f1 = f1_score(y_val, y_pred, pos_label=1)
        prec = precision_score(y_val, y_pred, pos_label=1)
        rec = recall_score(y_val, y_pred, pos_label=1)
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr_val = fp / max(fp + tn, 1)

        fold_dict = {
            "fold": fold_idx,
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "fpr": float(fpr_val),
            "confusion_matrix": cm.tolist(),
            "n_val": len(y_val),
        }
        fold_results.append(fold_dict)

        # Accumulate for global ROC
        all_y_true.extend(y_val.tolist())
        if y_prob is not None:
            all_y_prob.extend(y_prob.tolist())

        log.info(
            f"  Fold {fold_idx}/{n_folds}: F1={f1:.4f}  Prec={prec:.4f}  "
            f"Rec={rec:.4f}  FPR={fpr_val:.4f}"
        )

    elapsed = time.time() - start_time

    # Aggregate metrics (mean across folds)
    avg_f1 = float(np.mean([r["f1"] for r in fold_results]))
    avg_prec = float(np.mean([r["precision"] for r in fold_results]))
    avg_rec = float(np.mean([r["recall"] for r in fold_results]))
    avg_fpr = float(np.mean([r["fpr"] for r in fold_results]))

    # Aggregate confusion matrix (sum across folds)
    agg_cm = np.sum([np.array(r["confusion_matrix"]) for r in fold_results], axis=0)

    # Global ROC curve from pooled predictions
    roc_data: dict = {"fpr": [], "tpr": [], "auc": 0.0}
    if len(all_y_prob) == len(all_y_true):
        fpr_arr, tpr_arr, _ = roc_curve(all_y_true, all_y_prob, pos_label=1)
        roc_auc = float(auc(fpr_arr, tpr_arr))
        roc_data = {
            "fpr": fpr_arr.tolist(),
            "tpr": tpr_arr.tolist(),
            "auc": roc_auc,
        }

    # Performance mark: min(f1 * 35 / 0.98, 35)
    performance_mark = float(min(avg_f1 * 35 / 0.98, 35))

    results = {
        "model_name": model_name,
        "n_folds": n_folds,
        "f1": avg_f1,
        "precision": avg_prec,
        "recall": avg_rec,
        "fpr": avg_fpr,
        "confusion_matrix": agg_cm.tolist(),
        "fold_results": fold_results,
        "roc": roc_data,
        "performance_mark": performance_mark,
        "elapsed_seconds": round(elapsed, 2),
        "n_samples": len(y),
        "feature_dim": X.shape[1],
    }

    log.info(
        f"CV complete for {model_name}: "
        f"F1={avg_f1:.4f}  Prec={avg_prec:.4f}  Rec={avg_rec:.4f}  "
        f"FPR={avg_fpr:.4f}  AUC={roc_data['auc']:.4f}  "
        f"Mark={performance_mark:.1f}/35  Time={elapsed:.1f}s"
    )

    return results


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
