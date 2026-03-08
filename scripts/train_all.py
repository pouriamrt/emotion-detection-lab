"""Offline training script: trains all model + feature combinations.

Runs 10-fold cross-validation for every traditional model / feature combo and
every deep linear-probe model, saves trained artefacts (.joblib) and per-model
results (.json), and writes a consolidated metadata.json leaderboard.

Usage:
    python -m scripts.train_all
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Ensure the project root is importable regardless of working directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import (
    FEATURES_DIR,
    METADATA_PATH,
    MODELS_DIR,
    RESULTS_DIR,
)
from src.data_loader import load_images_for_deep, load_images_for_traditional
from src.evaluation import cross_validate
from src.features import extract_features
from src.logger import get_logger
from src.models import (
    DEEP_MODEL_REGISTRY,
    MODEL_REGISTRY,
    create_deep_model,
    create_model,
)

log = get_logger(user="train_all", context="training")

# ── Feature combos for traditional models ────────────────────────────────

FEATURE_COMBOS: list[list[str]] = [
    ["InsightFace"],
    ["CLIP-ViT-B/32"],
    ["ConvNeXt-Tiny"],
    ["InsightFace", "CLIP-ViT-B/32"],
    ["InsightFace", "CLIP-ViT-B/32", "ConvNeXt-Tiny"],
    ["HOG", "LBP", "Gabor"],
    ["HOG", "LBP", "Gabor", "Landmarks"],
    ["HOG", "LBP", "Gabor", "InsightFace"],
]


def _run_key(model_name: str, features: list[str]) -> str:
    """Build a unique run key like ``SVM__InsightFace+CLIP-ViT-B_32``."""
    tag = "+".join(features).replace("/", "_")
    return f"{model_name}__{tag}"


def _safe_json(obj):
    """Convert numpy types so ``json.dumps`` doesn't choke."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── Traditional pipeline ─────────────────────────────────────────────────


def _train_traditional(
    images_gray: np.ndarray,
    images_color: np.ndarray,
    y: np.ndarray,
    all_results: dict,
) -> None:
    """Cross-validate and train all traditional model + feature combos."""

    # Pre-extract every unique feature method so caching kicks in once
    unique_methods: set[str] = set()
    for combo in FEATURE_COMBOS:
        unique_methods.update(combo)
    log.info(f"Pre-extracting {len(unique_methods)} unique feature methods")
    extract_features(images_gray, images_color, sorted(unique_methods), cache=True)

    total = len(MODEL_REGISTRY) * len(FEATURE_COMBOS)
    done = 0

    for combo in FEATURE_COMBOS:
        features_tag = "+".join(combo)
        log.info(f"Feature combo: {combo}")
        X = extract_features(images_gray, images_color, combo, cache=True)

        for model_name in MODEL_REGISTRY:
            done += 1
            run_key = _run_key(model_name, combo)
            log.info(
                f"[{done}/{total}] Training {run_key} "
                f"| X shape {X.shape}"
            )
            try:
                model = create_model(model_name)

                # --- 10-fold CV ---
                cv_results = cross_validate(
                    model, X, y, scale=True, user="train_all"
                )

                # --- Train final model on ALL data ---
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                final_model = create_model(model_name)
                final_model.fit(X_scaled, y)

                # --- Persist artefacts ---
                model_path = MODELS_DIR / f"{run_key}.joblib"
                joblib.dump(
                    {"model": final_model, "scaler": scaler}, str(model_path)
                )
                log.info(f"Saved model to {model_path}")

                result_path = RESULTS_DIR / f"{run_key}.json"
                with open(result_path, "w") as fh:
                    json.dump(cv_results, fh, indent=2, default=_safe_json)
                log.info(f"Saved CV results to {result_path}")

                # --- Collect for metadata ---
                all_results[run_key] = {
                    "model_name": model_name,
                    "features": combo,
                    "type": "traditional",
                    "f1": cv_results["f1"],
                    "precision": cv_results["precision"],
                    "recall": cv_results["recall"],
                    "fpr": cv_results["fpr"],
                    "confusion_matrix": cv_results["confusion_matrix"],
                    "performance_mark": cv_results["performance_mark"],
                    "fold_results": cv_results["fold_results"],
                    "roc": cv_results["roc"],
                    "model_path": str(model_path),
                }

            except Exception:
                log.exception(f"FAILED: {run_key}")


# ── Deep pipeline ────────────────────────────────────────────────────────


def _train_deep(
    images_gray: np.ndarray,
    images_color: np.ndarray,
    y: np.ndarray,
    all_results: dict,
) -> None:
    """Cross-validate and train all deep linear-probe models."""

    total = len(DEEP_MODEL_REGISTRY)
    done = 0

    for model_name in DEEP_MODEL_REGISTRY:
        done += 1
        run_key = model_name  # deep models use plain name as key
        log.info(f"[{done}/{total}] Training deep model: {run_key}")

        try:
            model, feature_source = create_deep_model(model_name)

            # Extract the corresponding feature source
            X = extract_features(
                images_gray, images_color, [feature_source], cache=True
            )

            # --- 10-fold CV (scale=False; DeepLinearProbe scales internally) ---
            cv_results = cross_validate(
                model, X, y, scale=False, user="train_all"
            )

            # --- Train final model on ALL data ---
            final_model, _ = create_deep_model(model_name)
            final_model.fit(X, y)

            # --- Persist artefacts ---
            model_path = MODELS_DIR / f"{run_key}.joblib"
            joblib.dump(final_model, str(model_path))
            log.info(f"Saved deep model to {model_path}")

            result_path = RESULTS_DIR / f"{run_key}.json"
            with open(result_path, "w") as fh:
                json.dump(cv_results, fh, indent=2, default=_safe_json)
            log.info(f"Saved CV results to {result_path}")

            # --- Collect for metadata ---
            all_results[run_key] = {
                "model_name": model_name,
                "features": [feature_source],
                "type": "deep",
                "f1": cv_results["f1"],
                "precision": cv_results["precision"],
                "recall": cv_results["recall"],
                "fpr": cv_results["fpr"],
                "confusion_matrix": cv_results["confusion_matrix"],
                "performance_mark": cv_results["performance_mark"],
                "fold_results": cv_results["fold_results"],
                "roc": cv_results["roc"],
                "model_path": str(model_path),
            }

        except Exception:
            log.exception(f"FAILED: {run_key}")


# ── Main entry point ─────────────────────────────────────────────────────


def main() -> None:
    """Run the full offline training pipeline."""
    wall_start = time.time()
    log.info("=" * 60)
    log.info("Starting offline training pipeline")
    log.info("=" * 60)

    # 1. Load images (grayscale for traditional features, colour for deep)
    log.info("Loading images (grayscale)...")
    images_gray, y_gray, fnames_gray = load_images_for_traditional()
    log.info("Loading images (colour / BGR)...")
    images_color, y_color, fnames_color = load_images_for_deep()

    # Sanity check: both loaders should yield the same sample order
    assert np.array_equal(y_gray, y_color), (
        "Label mismatch between grayscale and colour loaders"
    )
    y = y_gray

    log.info(
        f"Dataset: {len(y)} samples, "
        f"{int(y.sum())} positive, {int(len(y) - y.sum())} negative"
    )

    all_results: dict = {}

    # 2. Traditional models
    log.info("-" * 60)
    log.info("Phase 1: Traditional models")
    log.info("-" * 60)
    _train_traditional(images_gray, images_color, y, all_results)

    # 3. Deep linear probes
    log.info("-" * 60)
    log.info("Phase 2: Deep linear-probe models")
    log.info("-" * 60)
    _train_deep(images_gray, images_color, y, all_results)

    # 4. Build metadata.json
    wall_elapsed = round(time.time() - wall_start, 1)

    best_key = max(all_results, key=lambda k: all_results[k]["f1"]) if all_results else ""
    best_f1 = all_results[best_key]["f1"] if best_key else 0.0

    metadata = {
        "best_model": best_key,
        "best_f1": best_f1,
        "models": all_results,
        "training_time": wall_elapsed,
    }

    with open(METADATA_PATH, "w") as fh:
        json.dump(metadata, fh, indent=2, default=_safe_json)

    log.info("=" * 60)
    log.info(f"Training complete in {wall_elapsed}s")
    log.info(f"Best model: {best_key} (F1={best_f1:.4f})")
    log.info(f"Metadata saved to {METADATA_PATH}")
    log.info(f"Total runs: {len(all_results)}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
