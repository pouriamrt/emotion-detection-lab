import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import IMAGES_DIR, ANNOTATIONS_PATH, IMAGE_SIZE, POSITIVE_LABEL
from src.logger import get_logger


def load_annotations() -> pd.DataFrame:
    df = pd.read_csv(ANNOTATIONS_PATH, header=None, names=["filename", "label"])
    df["label_int"] = (df["label"] == POSITIVE_LABEL).astype(int)
    return df


def _find_image_file(filename: str) -> Path | None:
    """Find image file with case-insensitive matching."""
    target = filename.lower()
    for f in IMAGES_DIR.iterdir():
        if f.name.lower() == target:
            return f
    return None


def load_images(
    df: pd.DataFrame | None = None,
    grayscale: bool = False,
    resize: tuple[int, int] = IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all images. Returns (images_array, labels_array, filenames_list)."""
    log = get_logger(context="data")
    if df is None:
        df = load_annotations()

    images, labels, filenames = [], [], []
    skipped = 0

    for _, row in df.iterrows():
        img_path = _find_image_file(row["filename"])
        if img_path is None:
            skipped += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            continue

        if not grayscale and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if resize:
            img = cv2.resize(img, resize)

        images.append(img)
        labels.append(row["label_int"])
        filenames.append(row["filename"])

    if skipped > 0:
        log.warning(f"Skipped {skipped}/{len(df)} images (not found or unreadable)")

    log.info(f"Loaded {len(images)} images ({sum(labels)} happy, {len(labels) - sum(labels)} neutral)")
    return np.array(images), np.array(labels), filenames


def load_images_for_deep(resize: tuple[int, int] = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load images as BGR uint8 arrays for deep models. Shape: (N, H, W, 3)."""
    return load_images(grayscale=False, resize=resize)


def load_images_for_traditional(resize: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load images as grayscale for traditional feature extraction."""
    return load_images(grayscale=True, resize=resize or (128, 128))
