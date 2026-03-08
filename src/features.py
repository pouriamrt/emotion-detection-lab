"""Feature extraction methods for emotion detection.

Provides traditional (HOG, LBP, Gabor, Landmarks) and deep (ConvNeXt, CLIP,
InsightFace) feature extractors, plus a caching orchestrator.
"""

import hashlib
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.config import FEATURES_DIR, RANDOM_SEED
from src.logger import get_logger

log = get_logger(context="features")

# ---------------------------------------------------------------------------
# Traditional features
# ---------------------------------------------------------------------------


def extract_hog(images: np.ndarray) -> np.ndarray:
    """Extract HOG features from grayscale images (N, H, W).

    Uses orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2).
    Returns (N, n_features) float array.
    """
    from skimage.feature import hog

    log.info(f"Extracting HOG features from {len(images)} images")
    features = []
    for img in tqdm(images, desc="HOG"):
        feat = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            feature_vector=True,
        )
        features.append(feat)
    result = np.array(features, dtype=np.float32)
    log.info(f"HOG features shape: {result.shape}")
    return result


def extract_lbp(images: np.ndarray) -> np.ndarray:
    """Extract multi-scale LBP histogram features from grayscale images (N, H, W).

    Computes uniform LBP at 3 scales: (R=1, P=8), (R=2, P=16), (R=3, P=24).
    Concatenates normalised histograms for each scale.
    Returns (N, n_features) float array.
    """
    from skimage.feature import local_binary_pattern

    log.info(f"Extracting LBP features from {len(images)} images")
    scales = [(1, 8), (2, 16), (3, 24)]
    features = []
    for img in tqdm(images, desc="LBP"):
        histograms = []
        for radius, n_points in scales:
            lbp = local_binary_pattern(img, P=n_points, R=radius, method="uniform")
            # uniform LBP has n_points + 2 bins
            n_bins = n_points + 2
            hist, _ = np.histogram(
                lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
            )
            histograms.append(hist)
        features.append(np.concatenate(histograms))
    result = np.array(features, dtype=np.float32)
    log.info(f"LBP features shape: {result.shape}")
    return result


def extract_gabor(images: np.ndarray) -> np.ndarray:
    """Extract Gabor filter bank features from grayscale images (N, H, W).

    Uses frequencies [0.1..0.5] and thetas [0, pi/4, pi/2, 3*pi/4].
    For each kernel: mean and variance of real and imaginary filtered responses.
    Returns (N, n_features) float array.
    """
    from skimage.filters import gabor_kernel
    from scipy.ndimage import convolve

    log.info(f"Extracting Gabor features from {len(images)} images")
    frequencies = [0.1, 0.2, 0.3, 0.4, 0.5]
    thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # Pre-compute kernels
    kernels = []
    for freq in frequencies:
        for theta in thetas:
            kernel = gabor_kernel(freq, theta=theta)
            kernels.append(kernel)

    features = []
    for img in tqdm(images, desc="Gabor"):
        img_f = img.astype(np.float64)
        feat = []
        for kernel in kernels:
            filtered_real = convolve(img_f, np.real(kernel))
            filtered_imag = convolve(img_f, np.imag(kernel))
            feat.extend([
                filtered_real.mean(),
                filtered_real.var(),
                filtered_imag.mean(),
                filtered_imag.var(),
            ])
        features.append(feat)
    result = np.array(features, dtype=np.float32)
    log.info(f"Gabor features shape: {result.shape}")
    return result


def extract_landmarks(images_color: np.ndarray) -> np.ndarray:
    """Extract facial landmark features via MediaPipe Face Mesh.

    Extracts key points for mouth, eyes, and eyebrows, flattens (x,y,z)
    coordinates, and adds a mouth width/height ratio feature.
    Input: BGR uint8 images (N, H, W, 3).
    Returns (N, n_features) float array. Pads zeros if face not detected.
    """
    import mediapipe as mp

    log.info(f"Extracting Landmark features from {len(images_color)} images")

    # Key landmark indices
    mouth_idx = [61, 291, 0, 17, 13, 14, 78, 308]
    eyes_idx = [33, 263, 133, 362, 159, 386, 145, 374]
    eyebrows_idx = [70, 300, 63, 293, 105, 334, 66, 296]
    all_idx = mouth_idx + eyes_idx + eyebrows_idx

    # Total features: 24 landmarks * 3 coords + 1 ratio = 73
    n_features = len(all_idx) * 3 + 1

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    )

    features = []
    n_missed = 0
    for img_bgr in tqdm(images_color, desc="Landmarks"):
        import cv2
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            coords = []
            for idx in all_idx:
                coords.extend([lm[idx].x, lm[idx].y, lm[idx].z])
            # Mouth width/height ratio
            mouth_width = np.sqrt(
                (lm[61].x - lm[291].x) ** 2 + (lm[61].y - lm[291].y) ** 2
            )
            mouth_height = np.sqrt(
                (lm[0].x - lm[17].x) ** 2 + (lm[0].y - lm[17].y) ** 2
            )
            ratio = mouth_width / max(mouth_height, 1e-6)
            coords.append(ratio)
            features.append(coords)
        else:
            n_missed += 1
            features.append([0.0] * n_features)

    face_mesh.close()
    if n_missed > 0:
        log.warning(f"Face not detected in {n_missed}/{len(images_color)} images")
    result = np.array(features, dtype=np.float32)
    log.info(f"Landmark features shape: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Deep embeddings (frozen backbones)
# ---------------------------------------------------------------------------


def extract_convnext(images_color: np.ndarray) -> np.ndarray:
    """Extract ConvNeXt-Tiny ImageNet features (768-dim) from BGR uint8 images.

    Removes classifier head and extracts global-pooled features.
    Input: BGR uint8 (N, H, W, 3). Returns (N, 768) float array.
    """
    import torch
    import torchvision.transforms as T
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

    log.info(f"Extracting ConvNeXt-Tiny features from {len(images_color)} images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()  # Remove classification head
    model = model.to(device).eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(232),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    with torch.no_grad():
        for img_bgr in tqdm(images_color, desc="ConvNeXt"):
            import cv2
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor = transform(img_rgb).unsqueeze(0).to(device)
            feat = model(tensor).squeeze().cpu().numpy()
            # ConvNeXt with Identity classifier returns (1, 768, 1, 1) or (768,)
            feat = feat.flatten()
            features.append(feat)

    result = np.array(features, dtype=np.float32)
    log.info(f"ConvNeXt features shape: {result.shape}")
    return result


def extract_clip(images_color: np.ndarray) -> np.ndarray:
    """Extract CLIP ViT-B/32 embeddings (512-dim) from BGR uint8 images.

    Uses open_clip with pretrained='laion2b_s34b_b79k'.
    Input: BGR uint8 (N, H, W, 3). Returns (N, 512) float array.
    """
    import torch
    import open_clip
    from PIL import Image
    import cv2

    log.info(f"Extracting CLIP ViT-B/32 features from {len(images_color)} images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()

    features = []
    with torch.no_grad():
        for img_bgr in tqdm(images_color, desc="CLIP"):
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tensor = preprocess(pil_img).unsqueeze(0).to(device)
            feat = model.encode_image(tensor).squeeze().cpu().numpy()
            features.append(feat.flatten())

    result = np.array(features, dtype=np.float32)
    log.info(f"CLIP features shape: {result.shape}")
    return result


def extract_insightface(images_color: np.ndarray) -> np.ndarray:
    """Extract InsightFace ArcFace (buffalo_l) embeddings (512-dim).

    Resizes to 640x640 for detection. Pads zeros if face not detected.
    Input: BGR uint8 (N, H, W, 3). Returns (N, 512) float array.
    """
    import cv2
    from insightface.app import FaceAnalysis

    log.info(f"Extracting InsightFace features from {len(images_color)} images")

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    features = []
    n_missed = 0
    for img_bgr in tqdm(images_color, desc="InsightFace"):
        # Resize for detection
        img_resized = cv2.resize(img_bgr, (640, 640))
        faces = app.get(img_resized)
        if faces:
            # Use the face with highest detection score
            best_face = max(faces, key=lambda f: f.det_score)
            features.append(best_face.embedding.flatten())
        else:
            n_missed += 1
            features.append(np.zeros(512, dtype=np.float32))

    if n_missed > 0:
        log.warning(
            f"Face not detected in {n_missed}/{len(images_color)} images (InsightFace)"
        )
    result = np.array(features, dtype=np.float32)
    log.info(f"InsightFace features shape: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Feature extractor registry & orchestrator
# ---------------------------------------------------------------------------

FEATURE_EXTRACTORS: dict[str, callable] = {
    "HOG": extract_hog,
    "LBP": extract_lbp,
    "Gabor": extract_gabor,
    "Landmarks": extract_landmarks,
    "ConvNeXt-Tiny": extract_convnext,
    "CLIP-ViT-B/32": extract_clip,
    "InsightFace": extract_insightface,
}

# Methods that need colour (BGR) input rather than grayscale
_COLOR_METHODS = {"Landmarks", "ConvNeXt-Tiny", "CLIP-ViT-B/32", "InsightFace"}


def _cache_path(method: str) -> Path:
    """Return the .npy cache file path for a given method."""
    safe_name = method.replace("/", "_").replace(" ", "_")
    return FEATURES_DIR / f"{safe_name}.npy"


def extract_features(
    images_gray: np.ndarray,
    images_color: np.ndarray,
    methods: list[str],
    cache: bool = True,
) -> np.ndarray:
    """Extract features for selected methods, optionally caching to disk.

    Args:
        images_gray: Grayscale images (N, H, W) for traditional methods.
        images_color: BGR uint8 images (N, H, W, 3) for deep / landmark methods.
        methods: List of method names (keys of FEATURE_EXTRACTORS).
        cache: If True, save/load features as .npy in FEATURES_DIR.

    Returns:
        Concatenated feature matrix (N, total_features).
    """
    all_features = []

    for method in methods:
        if method not in FEATURE_EXTRACTORS:
            raise ValueError(
                f"Unknown feature method '{method}'. "
                f"Available: {list(FEATURE_EXTRACTORS.keys())}"
            )

        cp = _cache_path(method)

        # Try loading from cache
        if cache and cp.exists():
            log.info(f"Loading cached features for '{method}' from {cp}")
            feat = np.load(str(cp))
        else:
            extractor = FEATURE_EXTRACTORS[method]
            if method in _COLOR_METHODS:
                feat = extractor(images_color)
            else:
                feat = extractor(images_gray)

            if cache:
                np.save(str(cp), feat)
                log.info(f"Cached features for '{method}' to {cp}")

        all_features.append(feat)

    concatenated = np.concatenate(all_features, axis=1)
    log.info(
        f"Combined features from {methods}: shape {concatenated.shape}"
    )
    return concatenated
