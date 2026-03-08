# Emotion Detection App — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Streamlit web app for binary emotion classification (happy/neutral) on the SMILE_PLUS dataset with Google OAuth, loguru logging, multiple ML models (traditional + deep), hyperparameter tuning UI, and pre-trained model artifacts.

**Architecture:** Multi-page Streamlit app. Backend modules in `src/` handle data loading, feature extraction (HOG, LBP, Gabor, landmarks, ConvNeXt, CLIP, InsightFace), model training (SVM, RF, XGBoost, LightGBM, LR, KNN, deep linear probes, ensembles), and 10-fold stratified CV evaluation. Pre-trained models stored in `artifacts/`. Google OAuth for login, loguru for monitoring.

**Tech Stack:** Python 3.13, Streamlit, loguru, scikit-learn, XGBoost, LightGBM, PyTorch, torchvision, transformers, open-clip-torch, insightface, onnxruntime, mediapipe, scikit-image, OpenCV, plotly, streamlit-google-auth

---

### Task 1: Project Setup — Dependencies & Directory Structure

**Files:**
- Modify: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.streamlit/config.toml`
- Create: `.streamlit/secrets.toml.example`
- Create: `src/__init__.py`
- Modify: `.gitignore`

**Step 1: Create directory structure**

```bash
mkdir -p src pages scripts artifacts/features artifacts/models artifacts/results .streamlit logs
```

**Step 2: Write requirements.txt**

```
streamlit>=1.40.0
streamlit-google-auth>=0.1.0
loguru>=0.7.0
numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.5.0
scikit-image>=0.24.0
opencv-python-headless>=4.10.0
Pillow>=10.0.0
torch>=2.4.0
torchvision>=0.19.0
transformers>=4.45.0
open-clip-torch>=2.26.0
insightface>=0.7.3
onnxruntime>=1.19.0
xgboost>=2.1.0
lightgbm>=4.5.0
plotly>=5.24.0
matplotlib>=3.9.0
seaborn>=0.13.0
mediapipe>=0.10.0
joblib>=1.4.0
tqdm>=4.66.0
```

**Step 3: Update pyproject.toml** with project metadata and dependencies from requirements.txt

**Step 4: Write .streamlit/config.toml**

```toml
[theme]
primaryColor = "#8B0000"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
maxUploadSize = 500
```

**Step 5: Write .streamlit/secrets.toml.example**

```toml
[google_auth]
client_id = "your-client-id.apps.googleusercontent.com"
client_secret = "your-client-secret"
redirect_uri = "http://localhost:8501"
cookie_secret = "a-random-secret-string"
```

**Step 6: Create src/__init__.py** (empty file)

**Step 7: Update .gitignore** to include artifacts/, logs/, .streamlit/secrets.toml, *.pt, *.joblib, client_secret.json

**Step 8: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 9: Commit**

```bash
git add -A && git commit -m "chore: project setup with dependencies and directory structure"
```

---

### Task 2: Config & Logging

**Files:**
- Create: `src/config.py`
- Create: `src/logger.py`

**Step 1: Write src/config.py**

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "SMILE PLUS Training Set"
ANNOTATIONS_PATH = DATA_DIR / "annotations.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FEATURES_DIR = ARTIFACTS_DIR / "features"
MODELS_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

for d in [ARTIFACTS_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
RANDOM_SEED = 42
N_FOLDS = 10
POSITIVE_LABEL = "happy"
NEGATIVE_LABEL = "neutral"

FEATURE_METHODS = [
    "HOG", "LBP", "Gabor", "Landmarks",
    "ConvNeXt-Tiny", "CLIP-ViT-B/32", "InsightFace"
]

TRADITIONAL_MODELS = [
    "SVM", "Random Forest", "XGBoost", "LightGBM",
    "Logistic Regression", "KNN"
]

DEEP_MODELS = [
    "ConvNeXt-Tiny (Fine-tune)", "CLIP Linear Probe", "InsightFace + FC"
]

ENSEMBLE_MODELS = ["Soft Voting", "Stacking"]
```

**Step 2: Write src/logger.py**

```python
import sys
from loguru import logger
from src.config import LOGS_DIR

logger.remove()

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[user]:^20}</cyan> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

logger.add(sys.stderr, level="INFO", format=LOG_FORMAT,
           filter=lambda record: record["extra"].setdefault("user", "system") or True)

logger.add(str(LOGS_DIR / "app.log"), rotation="10 MB", retention="30 days",
           level="DEBUG", format=LOG_FORMAT,
           filter=lambda record: record["extra"].setdefault("user", "system") or True)

logger.add(str(LOGS_DIR / "auth.log"), rotation="5 MB", retention="30 days",
           level="INFO", format=LOG_FORMAT,
           filter=lambda record: "auth" in record["extra"].get("context", ""))

logger.add(str(LOGS_DIR / "training.log"), rotation="10 MB", retention="30 days",
           level="INFO", format=LOG_FORMAT,
           filter=lambda record: "training" in record["extra"].get("context", ""))


def get_logger(user: str = "system", context: str = "general"):
    return logger.bind(user=user, context=context)
```

**Step 3: Verify** — `python -c "from src.logger import get_logger; log = get_logger('test'); log.info('hello')"`

**Step 4: Commit**

```bash
git add src/config.py src/logger.py && git commit -m "feat: add config constants and loguru logging setup"
```

---

### Task 3: Google OAuth Authentication

**Files:**
- Create: `src/auth.py`

**Step 1: Write src/auth.py**

Implements Google OAuth with streamlit-google-auth. Falls back to dev mode (bypass) if credentials not configured.

```python
import streamlit as st
from src.logger import get_logger

def check_auth():
    """Returns (is_authenticated, user_info_dict_or_None). Handles Google OAuth or dev bypass."""
    log = get_logger(context="auth")

    # Dev mode bypass if no secrets configured
    if "google_auth" not in st.secrets:
        log.warning("Google OAuth not configured — running in dev mode")
        if "user" not in st.session_state:
            st.session_state["user"] = {
                "name": "Dev User",
                "email": "dev@localhost",
                "picture": "",
                "authenticated": True,
            }
            log.info("Dev user auto-logged in", user="Dev User")
        return True, st.session_state["user"]

    # Real Google OAuth
    try:
        from streamlit_google_auth import Authenticate
        authenticator = Authenticate(
            secret_credentials_path="client_secret.json",
            cookie_name="emotion_app_auth",
            cookie_key=st.secrets["google_auth"]["cookie_secret"],
            redirect_uri=st.secrets["google_auth"]["redirect_uri"],
        )
        authenticator.check_authentification()

        if st.session_state.get("connected"):
            user_info = st.session_state.get("user_info", {})
            if "user" not in st.session_state or st.session_state["user"].get("email") != user_info.get("email"):
                st.session_state["user"] = {
                    "name": user_info.get("name", "Unknown"),
                    "email": user_info.get("email", "Unknown"),
                    "picture": user_info.get("picture", ""),
                    "authenticated": True,
                }
                log.info(f"User logged in: {user_info.get('email')}", user=user_info.get("name", "Unknown"))
            return True, st.session_state["user"]
        else:
            authenticator.login()
            return False, None
    except Exception as e:
        log.error(f"Auth error: {e}")
        st.error(f"Authentication error: {e}. Running in dev mode.")
        st.session_state["user"] = {"name": "Dev User", "email": "dev@localhost", "picture": "", "authenticated": True}
        return True, st.session_state["user"]


def require_auth():
    """Call at top of every page. Returns user dict or stops execution."""
    is_auth, user = check_auth()
    if not is_auth:
        st.stop()
    return user


def logout():
    log = get_logger(context="auth")
    user = st.session_state.get("user", {})
    log.info(f"User logged out: {user.get('email')}", user=user.get("name", "system"))
    for key in ["user", "connected", "user_info"]:
        st.session_state.pop(key, None)
    st.rerun()
```

**Step 2: Verify** — import test: `python -c "from src.auth import check_auth"`

**Step 3: Commit**

```bash
git add src/auth.py && git commit -m "feat: add Google OAuth with dev mode fallback"
```

---

### Task 4: Data Loader

**Files:**
- Create: `src/data_loader.py`

**Step 1: Write src/data_loader.py**

Loads SMILE_PLUS images and annotations. Returns image arrays and labels. Handles case-insensitive filename matching.

```python
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
    """Load all images. Returns (images_array, labels_array, filenames_list).
    images_array shape: (N, H, W) if grayscale, (N, H, W, 3) if color.
    """
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
    """Load images as RGB uint8 arrays for deep models. Shape: (N, H, W, 3)."""
    return load_images(grayscale=False, resize=resize)


def load_images_for_traditional(resize: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load images as grayscale for traditional feature extraction."""
    return load_images(grayscale=True, resize=resize or (128, 128))
```

**Step 2: Verify** — `python -c "from src.data_loader import load_annotations; df = load_annotations(); print(df.shape, df['label'].value_counts().to_dict())"`

**Step 3: Commit**

```bash
git add src/data_loader.py && git commit -m "feat: add SMILE_PLUS data loader with case-insensitive file matching"
```

---

### Task 5: Traditional Feature Extraction

**Files:**
- Create: `src/features.py`

**Step 1: Write src/features.py (Part 1 — traditional features)**

Implements HOG, LBP, Gabor, facial landmarks. Each returns a 2D feature matrix (N_samples, N_features).

```python
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor_kernel
from scipy.ndimage import convolve
from tqdm import tqdm
from src.logger import get_logger
from src.config import FEATURES_DIR
import joblib


def extract_hog(images: np.ndarray) -> np.ndarray:
    """HOG features. Input: (N, H, W) grayscale. Output: (N, D)."""
    log = get_logger(context="features")
    log.info(f"Extracting HOG features from {len(images)} images")
    features = []
    for img in tqdm(images, desc="HOG"):
        feat = hog(img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), feature_vector=True)
        features.append(feat)
    result = np.array(features)
    log.info(f"HOG features shape: {result.shape}")
    return result


def extract_lbp(images: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """LBP histogram features. Input: (N, H, W) grayscale. Output: (N, n_bins)."""
    log = get_logger(context="features")
    log.info(f"Extracting LBP features (radius={radius}, n_points={n_points})")
    features = []
    for img in tqdm(images, desc="LBP"):
        lbp = local_binary_pattern(img, n_points, radius, method="uniform")
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        features.append(hist)
    # Multi-scale LBP: also compute for radius=2,3
    for r, p in [(2, 16), (3, 24)]:
        for img in tqdm(images, desc=f"LBP r={r}"):
            lbp = local_binary_pattern(img, p, r, method="uniform")
            n_bins = p + 2
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            features[images.tolist().index(img.tolist()) if False else len(features)] = hist  # Will fix below
    # Simpler: concatenate multi-scale
    all_features = []
    for img in tqdm(images, desc="LBP multi-scale"):
        combined = []
        for r, p in [(1, 8), (2, 16), (3, 24)]:
            lbp = local_binary_pattern(img, p, r, method="uniform")
            n_bins = p + 2
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            combined.extend(hist)
        all_features.append(combined)
    result = np.array(all_features)
    log.info(f"LBP features shape: {result.shape}")
    return result


def extract_gabor(images: np.ndarray) -> np.ndarray:
    """Gabor filter bank features. Input: (N, H, W) grayscale. Output: (N, D)."""
    log = get_logger(context="features")
    log.info("Extracting Gabor features")
    frequencies = [0.1, 0.2, 0.3, 0.4, 0.5]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    kernels = []
    for freq in frequencies:
        for theta in thetas:
            kernel = gabor_kernel(freq, theta=theta)
            kernels.append(kernel)

    features = []
    for img in tqdm(images, desc="Gabor"):
        img_features = []
        img_float = img.astype(np.float64) / 255.0
        for kernel in kernels:
            filtered_real = convolve(img_float, kernel.real)
            filtered_imag = convolve(img_float, kernel.imag)
            img_features.extend([
                filtered_real.mean(), filtered_real.var(),
                filtered_imag.mean(), filtered_imag.var(),
            ])
        features.append(img_features)
    result = np.array(features)
    log.info(f"Gabor features shape: {result.shape}")
    return result


def extract_landmarks(images_color: np.ndarray) -> np.ndarray:
    """Facial landmark geometric features via MediaPipe. Input: (N, H, W, 3) RGB. Output: (N, D)."""
    log = get_logger(context="features")
    log.info("Extracting facial landmark features")
    import mediapipe as mp

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
    )

    # Key landmark indices for expression: mouth, eyes, eyebrows
    MOUTH = [61, 291, 0, 17, 13, 14, 78, 308]
    EYES = [33, 263, 133, 362, 159, 386, 145, 374]
    EYEBROWS = [70, 300, 63, 293, 105, 334, 66, 296]
    KEY_POINTS = MOUTH + EYES + EYEBROWS

    features = []
    for img in tqdm(images_color, desc="Landmarks"):
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.shape[-1] == 3 else img)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            coords = [(lm[i].x, lm[i].y, lm[i].z) for i in KEY_POINTS if i < len(lm)]
            # Flatten coords + pairwise distances between mouth corners
            flat = [c for pt in coords for c in pt]
            # Add key distance ratios
            if len(coords) >= 8:
                # Mouth width / height ratio
                mouth_w = np.sqrt((coords[0][0]-coords[1][0])**2 + (coords[0][1]-coords[1][1])**2)
                mouth_h = np.sqrt((coords[2][0]-coords[3][0])**2 + (coords[2][1]-coords[3][1])**2)
                flat.append(mouth_w / (mouth_h + 1e-6))
                # Eye aspect ratios
                eye_w = np.sqrt((coords[8][0]-coords[9][0])**2 + (coords[8][1]-coords[9][1])**2)
                flat.append(eye_w)
            features.append(flat)
        else:
            features.append([0.0] * (len(KEY_POINTS) * 3 + 2))

    # Pad to same length
    max_len = max(len(f) for f in features)
    features = [f + [0.0] * (max_len - len(f)) for f in features]

    face_mesh.close()
    result = np.array(features)
    log.info(f"Landmarks features shape: {result.shape}")
    return result
```

**Step 2: Verify** — Quick test with dummy images:
```bash
python -c "
from src.data_loader import load_images_for_traditional
images, labels, _ = load_images_for_traditional()
from src.features import extract_hog
hog_feats = extract_hog(images[:5])
print('HOG shape:', hog_feats.shape)
"
```

**Step 3: Commit**

```bash
git add src/features.py && git commit -m "feat: add traditional feature extraction (HOG, LBP, Gabor, landmarks)"
```

---

### Task 6: Deep Feature Extraction

**Files:**
- Modify: `src/features.py` (append deep extraction functions)

**Step 1: Add deep embedding extractors to src/features.py**

ConvNeXt-Tiny, CLIP, and InsightFace as frozen feature extractors.

```python
import torch
from torchvision import transforms


def _get_deep_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_convnext(images_color: np.ndarray) -> np.ndarray:
    """ConvNeXt-Tiny embeddings. Input: (N, H, W, 3) BGR uint8. Output: (N, 768)."""
    log = get_logger(context="features")
    log.info("Extracting ConvNeXt-Tiny features")
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(device)
    model.classifier = torch.nn.Identity()  # Remove classification head
    model.eval()

    transform = _get_deep_transform()
    features = []

    with torch.no_grad():
        for img in tqdm(images_color, desc="ConvNeXt-Tiny"):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = transform(img_rgb).unsqueeze(0).to(device)
            feat = model(tensor).squeeze().cpu().numpy()
            features.append(feat)

    result = np.array(features)
    log.info(f"ConvNeXt features shape: {result.shape}")
    return result


def extract_clip(images_color: np.ndarray) -> np.ndarray:
    """CLIP ViT-B/32 embeddings. Input: (N, H, W, 3) BGR uint8. Output: (N, 512)."""
    log = get_logger(context="features")
    log.info("Extracting CLIP features")
    import open_clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()

    features = []
    with torch.no_grad():
        for img in tqdm(images_color, desc="CLIP"):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_img = Image.fromarray(img_rgb)
            tensor = preprocess(pil_img).unsqueeze(0).to(device)
            feat = model.encode_image(tensor).squeeze().cpu().numpy()
            features.append(feat)

    result = np.array(features).astype(np.float32)
    log.info(f"CLIP features shape: {result.shape}")
    return result


def extract_insightface(images_color: np.ndarray) -> np.ndarray:
    """InsightFace ArcFace embeddings. Input: (N, H, W, 3) BGR uint8. Output: (N, 512)."""
    log = get_logger(context="features")
    log.info("Extracting InsightFace features")
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    features = []
    failed = 0
    for img in tqdm(images_color, desc="InsightFace"):
        # InsightFace expects BGR
        img_resized = cv2.resize(img, (640, 640)) if img.shape[:2] != (640, 640) else img
        faces = app.get(img_resized)
        if faces:
            features.append(faces[0].embedding)
        else:
            # Fallback: pad with zeros
            features.append(np.zeros(512, dtype=np.float32))
            failed += 1

    if failed > 0:
        log.warning(f"InsightFace failed to detect face in {failed}/{len(images_color)} images")

    result = np.array(features)
    log.info(f"InsightFace features shape: {result.shape}")
    return result


# ── Feature pipeline orchestrator ──

FEATURE_EXTRACTORS = {
    "HOG": lambda imgs, imgs_color: extract_hog(imgs),
    "LBP": lambda imgs, imgs_color: extract_lbp(imgs),
    "Gabor": lambda imgs, imgs_color: extract_gabor(imgs),
    "Landmarks": lambda imgs, imgs_color: extract_landmarks(imgs_color),
    "ConvNeXt-Tiny": lambda imgs, imgs_color: extract_convnext(imgs_color),
    "CLIP-ViT-B/32": lambda imgs, imgs_color: extract_clip(imgs_color),
    "InsightFace": lambda imgs, imgs_color: extract_insightface(imgs_color),
}


def extract_features(
    images_gray: np.ndarray,
    images_color: np.ndarray,
    methods: list[str],
    cache: bool = True,
) -> np.ndarray:
    """Extract and concatenate features from selected methods.
    Returns (N, total_features) matrix.
    """
    log = get_logger(context="features")
    all_features = []

    for method in methods:
        cache_path = FEATURES_DIR / f"{method.replace('/', '_')}.npy"

        if cache and cache_path.exists():
            log.info(f"Loading cached {method} features")
            feat = np.load(str(cache_path))
        else:
            extractor = FEATURE_EXTRACTORS.get(method)
            if extractor is None:
                log.error(f"Unknown feature method: {method}")
                continue
            feat = extractor(images_gray, images_color)
            if cache:
                np.save(str(cache_path), feat)
                log.info(f"Cached {method} features to {cache_path}")

        all_features.append(feat)

    combined = np.hstack(all_features)
    log.info(f"Combined features shape: {combined.shape} from {methods}")
    return combined
```

**Step 2: Verify** — test ConvNeXt extraction on 2 images:
```bash
python -c "
from src.data_loader import load_images_for_deep
imgs, labels, _ = load_images_for_deep()
from src.features import extract_convnext
feats = extract_convnext(imgs[:2])
print('ConvNeXt shape:', feats.shape)
"
```

**Step 3: Commit**

```bash
git add src/features.py && git commit -m "feat: add deep feature extraction (ConvNeXt, CLIP, InsightFace) and feature pipeline"
```

---

### Task 7: Model Registry — Traditional Classifiers

**Files:**
- Create: `src/models.py`

**Step 1: Write src/models.py**

Registry of all classifiers with default hyperparameters and parameter ranges for UI sliders.

```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np
from src.config import RANDOM_SEED


MODEL_REGISTRY = {
    "SVM": {
        "class": SVC,
        "default_params": {"C": 10.0, "kernel": "rbf", "gamma": "scale", "probability": True, "random_state": RANDOM_SEED},
        "param_grid": {
            "C": {"type": "float", "min": 0.01, "max": 1000.0, "default": 10.0, "log": True,
                   "desc": "Regularization parameter. Higher C = less regularization, tighter fit."},
            "kernel": {"type": "select", "options": ["rbf", "linear", "poly", "sigmoid"], "default": "rbf",
                       "desc": "Kernel function. RBF is most flexible; linear for linearly separable data."},
            "gamma": {"type": "select", "options": ["scale", "auto"], "default": "scale",
                      "desc": "Kernel coefficient. 'scale' = 1/(n_features*X.var()), 'auto' = 1/n_features."},
        },
    },
    "Random Forest": {
        "class": RandomForestClassifier,
        "default_params": {"n_estimators": 500, "max_depth": 20, "min_samples_split": 2, "random_state": RANDOM_SEED, "n_jobs": -1},
        "param_grid": {
            "n_estimators": {"type": "int", "min": 50, "max": 1000, "default": 500, "step": 50,
                             "desc": "Number of trees. More trees = more stable but slower."},
            "max_depth": {"type": "int", "min": 3, "max": 50, "default": 20, "step": 1,
                          "desc": "Max tree depth. Deeper = more complex patterns but risk overfitting."},
            "min_samples_split": {"type": "int", "min": 2, "max": 20, "default": 2, "step": 1,
                                  "desc": "Min samples to split a node. Higher = more regularization."},
        },
    },
    "XGBoost": {
        "class": XGBClassifier,
        "default_params": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8,
                           "colsample_bytree": 0.8, "random_state": RANDOM_SEED, "use_label_encoder": False,
                           "eval_metric": "logloss"},
        "param_grid": {
            "n_estimators": {"type": "int", "min": 50, "max": 1000, "default": 300, "step": 50,
                             "desc": "Number of boosting rounds. More = better fit but risk overfitting."},
            "max_depth": {"type": "int", "min": 2, "max": 15, "default": 6, "step": 1,
                          "desc": "Max tree depth per round. Controls model complexity."},
            "learning_rate": {"type": "float", "min": 0.001, "max": 0.5, "default": 0.1, "log": True,
                              "desc": "Step size shrinkage. Lower = more conservative, needs more rounds."},
            "subsample": {"type": "float", "min": 0.5, "max": 1.0, "default": 0.8, "step": 0.05,
                          "desc": "Fraction of samples per tree. Lower = more regularization."},
        },
    },
    "LightGBM": {
        "class": LGBMClassifier,
        "default_params": {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.1,
                           "random_state": RANDOM_SEED, "verbose": -1, "n_jobs": -1},
        "param_grid": {
            "n_estimators": {"type": "int", "min": 50, "max": 1000, "default": 300, "step": 50,
                             "desc": "Number of boosting iterations."},
            "num_leaves": {"type": "int", "min": 10, "max": 100, "default": 31, "step": 1,
                           "desc": "Max leaves per tree. More = more complex. Key LightGBM param."},
            "learning_rate": {"type": "float", "min": 0.001, "max": 0.5, "default": 0.1, "log": True,
                              "desc": "Step size shrinkage per iteration."},
        },
    },
    "Logistic Regression": {
        "class": LogisticRegression,
        "default_params": {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 5000, "random_state": RANDOM_SEED},
        "param_grid": {
            "C": {"type": "float", "min": 0.001, "max": 100.0, "default": 1.0, "log": True,
                  "desc": "Inverse regularization strength. Smaller C = stronger regularization."},
            "penalty": {"type": "select", "options": ["l2", "l1", "none"], "default": "l2",
                        "desc": "Regularization type. L2 = Ridge, L1 = Lasso (sparse features)."},
        },
    },
    "KNN": {
        "class": KNeighborsClassifier,
        "default_params": {"n_neighbors": 5, "weights": "distance", "metric": "euclidean", "n_jobs": -1},
        "param_grid": {
            "n_neighbors": {"type": "int", "min": 1, "max": 50, "default": 5, "step": 1,
                            "desc": "Number of neighbors. Lower = more complex boundary."},
            "weights": {"type": "select", "options": ["uniform", "distance"], "default": "distance",
                        "desc": "Weight function. 'distance' = closer neighbors have more influence."},
            "metric": {"type": "select", "options": ["euclidean", "manhattan", "cosine"], "default": "euclidean",
                       "desc": "Distance metric for neighbor calculation."},
        },
    },
}


def create_model(name: str, params: dict | None = None):
    """Create a model instance with given or default params."""
    reg = MODEL_REGISTRY[name]
    final_params = {**reg["default_params"]}
    if params:
        final_params.update(params)
    # Handle special cases
    if name == "Logistic Regression" and final_params.get("penalty") == "l1":
        final_params["solver"] = "liblinear"
    elif name == "Logistic Regression" and final_params.get("penalty") == "none":
        final_params["penalty"] = None
    return reg["class"](**final_params)


def create_ensemble(base_models: dict, method: str = "voting"):
    """Create ensemble from trained base models.
    base_models: {name: model_instance}
    """
    estimators = [(name, model) for name, model in base_models.items()]

    if method == "voting":
        return VotingClassifier(estimators=estimators, voting="soft")
    elif method == "stacking":
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=5000, random_state=RANDOM_SEED),
            cv=5,
            n_jobs=-1,
        )
```

**Step 2: Verify** — `python -c "from src.models import create_model; m = create_model('SVM'); print(m)"`

**Step 3: Commit**

```bash
git add src/models.py && git commit -m "feat: add model registry with traditional classifiers and ensemble support"
```

---

### Task 8: Deep Learning Models (Last-Layer Fine-Tuning)

**Files:**
- Modify: `src/models.py` (append deep model classes)

**Step 1: Add deep model classes to src/models.py**

Each deep model freezes the backbone and only trains the final classification layer.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler


class DeepLinearProbe(BaseEstimator, ClassifierMixin):
    """Wraps a frozen backbone + trainable linear head as sklearn-compatible classifier.
    Works with pre-extracted embeddings (not raw images).
    """

    def __init__(self, input_dim=512, lr=0.001, epochs=50, batch_size=32, weight_decay=1e-4):
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X).astype(np.float32)
        y = y.astype(np.int64)

        self.model_ = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        ).to(self.device)

        dataset = TensorDataset(torch.from_numpy(X_scaled), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        self.model_.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model_(X_batch), y_batch)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        self.model_.eval()
        X_scaled = self.scaler_.transform(X).astype(np.float32)
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X_scaled).to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs


DEEP_MODEL_REGISTRY = {
    "ConvNeXt-Tiny (Fine-tune)": {
        "class": DeepLinearProbe,
        "default_params": {"input_dim": 768, "lr": 0.001, "epochs": 50, "batch_size": 32, "weight_decay": 1e-4},
        "feature_source": "ConvNeXt-Tiny",
        "param_grid": {
            "lr": {"type": "float", "min": 0.0001, "max": 0.01, "default": 0.001, "log": True,
                   "desc": "Learning rate for AdamW optimizer."},
            "epochs": {"type": "int", "min": 10, "max": 200, "default": 50, "step": 10,
                       "desc": "Number of training epochs."},
            "batch_size": {"type": "select", "options": [16, 32, 64], "default": 32,
                           "desc": "Mini-batch size. Smaller = noisier gradients, better generalization."},
            "weight_decay": {"type": "float", "min": 1e-6, "max": 1e-2, "default": 1e-4, "log": True,
                             "desc": "L2 regularization strength."},
        },
    },
    "CLIP Linear Probe": {
        "class": DeepLinearProbe,
        "default_params": {"input_dim": 512, "lr": 0.001, "epochs": 50, "batch_size": 32, "weight_decay": 1e-4},
        "feature_source": "CLIP-ViT-B/32",
        "param_grid": {
            "lr": {"type": "float", "min": 0.0001, "max": 0.01, "default": 0.001, "log": True,
                   "desc": "Learning rate for AdamW optimizer."},
            "epochs": {"type": "int", "min": 10, "max": 200, "default": 50, "step": 10,
                       "desc": "Number of training epochs."},
            "batch_size": {"type": "select", "options": [16, 32, 64], "default": 32,
                           "desc": "Mini-batch size."},
            "weight_decay": {"type": "float", "min": 1e-6, "max": 1e-2, "default": 1e-4, "log": True,
                             "desc": "L2 regularization strength."},
        },
    },
    "InsightFace + FC": {
        "class": DeepLinearProbe,
        "default_params": {"input_dim": 512, "lr": 0.001, "epochs": 50, "batch_size": 32, "weight_decay": 1e-4},
        "feature_source": "InsightFace",
        "param_grid": {
            "lr": {"type": "float", "min": 0.0001, "max": 0.01, "default": 0.001, "log": True,
                   "desc": "Learning rate for AdamW optimizer."},
            "epochs": {"type": "int", "min": 10, "max": 200, "default": 50, "step": 10,
                       "desc": "Number of training epochs."},
            "batch_size": {"type": "select", "options": [16, 32, 64], "default": 32,
                           "desc": "Mini-batch size."},
            "weight_decay": {"type": "float", "min": 1e-6, "max": 1e-2, "default": 1e-4, "log": True,
                             "desc": "L2 regularization strength."},
        },
    },
}


def create_deep_model(name: str, params: dict | None = None):
    """Create a deep linear probe model."""
    reg = DEEP_MODEL_REGISTRY[name]
    final_params = {**reg["default_params"]}
    if params:
        final_params.update(params)
    return reg["class"](**final_params), reg["feature_source"]
```

**Step 2: Verify** — `python -c "from src.models import create_deep_model; m, src = create_deep_model('CLIP Linear Probe'); print(m, src)"`

**Step 3: Commit**

```bash
git add src/models.py && git commit -m "feat: add deep learning linear probe models (ConvNeXt, CLIP, InsightFace)"
```

---

### Task 9: Evaluation Module

**Files:**
- Create: `src/evaluation.py`

**Step 1: Write src/evaluation.py**

10-fold stratified CV with all required metrics. Compatible with both sklearn and deep models.

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import time
from src.config import N_FOLDS, RANDOM_SEED
from src.logger import get_logger


def cross_validate(model, X, y, n_folds=N_FOLDS, scale=True, user="system"):
    """Run stratified k-fold CV. Returns dict with all metrics."""
    log = get_logger(user=user, context="training")
    log.info(f"Starting {n_folds}-fold CV with {type(model).__name__} on {X.shape}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []
    all_y_true, all_y_pred, all_y_prob = [], [], []

    start_time = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        # Clone model for each fold (handles both sklearn and custom models)
        try:
            fold_model = clone(model)
        except Exception:
            fold_model = model.__class__(**model.get_params())

        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_val)

        # Get probabilities if available
        y_prob = None
        if hasattr(fold_model, "predict_proba"):
            y_prob = fold_model.predict_proba(X_val)[:, 1]
        elif hasattr(fold_model, "decision_function"):
            y_prob = fold_model.decision_function(X_val)

        fold_f1 = f1_score(y_val, y_pred, pos_label=1)
        fold_precision = precision_score(y_val, y_pred, pos_label=1)
        fold_recall = recall_score(y_val, y_pred, pos_label=1)

        fold_results.append({
            "fold": fold_idx + 1,
            "f1": fold_f1,
            "precision": fold_precision,
            "recall": fold_recall,
        })

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        if y_prob is not None:
            all_y_prob.extend(y_prob)

        log.info(f"Fold {fold_idx+1}/{n_folds}: F1={fold_f1:.4f} P={fold_precision:.4f} R={fold_recall:.4f}")

    elapsed = time.time() - start_time

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    cm = confusion_matrix(all_y_true, all_y_pred)
    tn, fp, fn, tp = cm.ravel()

    overall_f1 = f1_score(all_y_true, all_y_pred, pos_label=1)
    overall_precision = precision_score(all_y_true, all_y_pred, pos_label=1)
    overall_recall = recall_score(all_y_true, all_y_pred, pos_label=1)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # ROC curve
    roc_data = None
    if all_y_prob:
        all_y_prob = np.array(all_y_prob)
        fpr_curve, tpr_curve, _ = roc_curve(all_y_true, all_y_prob, pos_label=1)
        roc_auc = auc(fpr_curve, tpr_curve)
        roc_data = {"fpr": fpr_curve.tolist(), "tpr": tpr_curve.tolist(), "auc": roc_auc}

    # Performance mark calculation: min(F1*35/0.98, 35)
    performance_mark = min(overall_f1 * 35 / 0.98, 35)

    results = {
        "model_name": type(model).__name__,
        "n_folds": n_folds,
        "f1": overall_f1,
        "precision": overall_precision,
        "recall": overall_recall,
        "fpr": fpr,
        "confusion_matrix": cm.tolist(),
        "fold_results": fold_results,
        "roc": roc_data,
        "performance_mark": performance_mark,
        "elapsed_seconds": elapsed,
        "n_samples": len(all_y_true),
        "feature_dim": X.shape[1],
    }

    log.info(f"CV complete: F1={overall_f1:.4f} P={overall_precision:.4f} R={overall_recall:.4f} "
             f"FPR={fpr:.4f} Mark={performance_mark:.1f}/35 Time={elapsed:.1f}s")

    return results
```

**Step 2: Verify** — Quick test with dummy data:
```bash
python -c "
from sklearn.datasets import make_classification
from src.evaluation import cross_validate
from src.models import create_model
X, y = make_classification(n_samples=100, n_features=20, random_state=42)
model = create_model('SVM')
res = cross_validate(model, X, y, n_folds=3)
print(f\"F1={res['f1']:.3f}, Mark={res['performance_mark']:.1f}/35\")
"
```

**Step 3: Commit**

```bash
git add src/evaluation.py && git commit -m "feat: add 10-fold stratified CV evaluation with all required metrics"
```

---

### Task 10: Offline Training Script

**Files:**
- Create: `scripts/train_all.py`

**Step 1: Write scripts/train_all.py**

Trains all model+feature combinations, saves artifacts. Run once before launching app.

```python
"""Train all models on SMILE_PLUS dataset and save artifacts.
Usage: python -m scripts.train_all
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    FEATURES_DIR, MODELS_DIR, RESULTS_DIR, METADATA_PATH, RANDOM_SEED,
)
from src.data_loader import load_images_for_traditional, load_images_for_deep
from src.features import extract_features, FEATURE_EXTRACTORS
from src.models import (
    MODEL_REGISTRY, DEEP_MODEL_REGISTRY,
    create_model, create_deep_model,
)
from src.evaluation import cross_validate
from src.logger import get_logger

log = get_logger(user="train_all", context="training")


def train_traditional_models(images_gray, images_color, labels):
    """Train all traditional models with best feature combinations."""
    results = {}

    # Best feature combos to try (ordered by expected performance)
    feature_combos = [
        ["InsightFace"],
        ["CLIP-ViT-B/32"],
        ["ConvNeXt-Tiny"],
        ["InsightFace", "CLIP-ViT-B/32"],
        ["InsightFace", "CLIP-ViT-B/32", "ConvNeXt-Tiny"],
        ["HOG", "LBP", "Gabor"],
        ["HOG", "LBP", "Gabor", "Landmarks"],
        ["HOG", "LBP", "Gabor", "InsightFace"],
    ]

    for features_list in feature_combos:
        log.info(f"Extracting features: {features_list}")
        X = extract_features(images_gray, images_color, features_list)
        features_key = "+".join(features_list)

        for model_name in MODEL_REGISTRY:
            run_key = f"{model_name}__{features_key}"
            log.info(f"Training {run_key}")

            try:
                model = create_model(model_name)
                cv_results = cross_validate(model, X, labels, user="train_all")

                # Save model (train on full data)
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                final_model = create_model(model_name)
                final_model.fit(X_scaled, labels)

                model_path = MODELS_DIR / f"{run_key}.joblib"
                joblib.dump({"model": final_model, "scaler": scaler}, str(model_path))

                results[run_key] = {
                    "model_name": model_name,
                    "features": features_list,
                    "features_key": features_key,
                    "cv_results": cv_results,
                    "model_path": str(model_path),
                    "type": "traditional",
                }
                log.info(f"{run_key}: F1={cv_results['f1']:.4f}")
            except Exception as e:
                log.error(f"Failed {run_key}: {e}")

    return results


def train_deep_models(images_gray, images_color, labels):
    """Train deep linear probe models."""
    results = {}

    for model_name, reg in DEEP_MODEL_REGISTRY.items():
        feature_source = reg["feature_source"]
        log.info(f"Training {model_name} with {feature_source} features")

        try:
            X = extract_features(images_gray, images_color, [feature_source])
            model, _ = create_deep_model(model_name)
            cv_results = cross_validate(model, X, labels, scale=False, user="train_all")

            # Save model trained on full data
            final_model, _ = create_deep_model(model_name)
            final_model.fit(X, labels)
            model_path = MODELS_DIR / f"{model_name.replace(' ', '_')}.joblib"
            joblib.dump({"model": final_model}, str(model_path))

            results[model_name] = {
                "model_name": model_name,
                "features": [feature_source],
                "features_key": feature_source,
                "cv_results": cv_results,
                "model_path": str(model_path),
                "type": "deep",
            }
            log.info(f"{model_name}: F1={cv_results['f1']:.4f}")
        except Exception as e:
            log.error(f"Failed {model_name}: {e}")

    return results


def main():
    log.info("=== Starting full training pipeline ===")
    start = time.time()

    # Load data
    log.info("Loading images...")
    images_gray, labels_gray, _ = load_images_for_traditional()
    images_color, labels_color, _ = load_images_for_deep()
    labels = labels_gray  # Same order

    all_results = {}

    # Train traditional models
    log.info("=== Training Traditional Models ===")
    trad_results = train_traditional_models(images_gray, images_color, labels)
    all_results.update(trad_results)

    # Train deep models
    log.info("=== Training Deep Models ===")
    deep_results = train_deep_models(images_gray, images_color, labels)
    all_results.update(deep_results)

    # Find best model
    best_key = max(all_results, key=lambda k: all_results[k]["cv_results"]["f1"])
    best = all_results[best_key]
    log.info(f"Best model: {best_key} with F1={best['cv_results']['f1']:.4f}")

    # Save metadata
    metadata = {
        "best_model": best_key,
        "best_f1": best["cv_results"]["f1"],
        "models": {},
        "training_time": time.time() - start,
    }
    for key, res in all_results.items():
        metadata["models"][key] = {
            "model_name": res["model_name"],
            "features": res["features"],
            "type": res["type"],
            "f1": res["cv_results"]["f1"],
            "precision": res["cv_results"]["precision"],
            "recall": res["cv_results"]["recall"],
            "fpr": res["cv_results"]["fpr"],
            "confusion_matrix": res["cv_results"]["confusion_matrix"],
            "performance_mark": res["cv_results"]["performance_mark"],
            "fold_results": res["cv_results"]["fold_results"],
            "roc": res["cv_results"]["roc"],
            "model_path": res["model_path"],
        }

    # Save results per model
    for key, res in all_results.items():
        result_path = RESULTS_DIR / f"{key.replace('/', '_')}.json"
        with open(result_path, "w") as f:
            json.dump(res["cv_results"], f, indent=2, default=str)

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    elapsed = time.time() - start
    log.info(f"=== Training complete in {elapsed:.0f}s. {len(all_results)} models trained. ===")
    log.info(f"Best: {best_key} F1={best['cv_results']['f1']:.4f} Mark={best['cv_results']['performance_mark']:.1f}/35")


if __name__ == "__main__":
    main()
```

**Step 2: Create scripts/__init__.py** (empty)

**Step 3: Commit** (don't run yet — will run after all code is in place)

```bash
git add scripts/ && git commit -m "feat: add offline training script for all model+feature combinations"
```

---

### Task 11: Streamlit Main App (Home + Auth)

**Files:**
- Create: `app.py`

**Step 1: Write app.py**

```python
import streamlit as st
from src.auth import check_auth, logout
from src.logger import get_logger

st.set_page_config(
    page_title="Emotion Detection Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Authentication ---
is_auth, user = check_auth()

if not is_auth:
    st.title("Emotion Detection Lab")
    st.markdown("### Binary Classification: Happy vs Neutral")
    st.markdown("Please log in with Google to continue.")
    st.stop()

# --- Logged-in view ---
log = get_logger(user=user["name"], context="general")

# Sidebar
with st.sidebar:
    st.markdown(f"**Logged in as:** {user['name']}")
    st.caption(user["email"])
    if st.button("Logout"):
        logout()

# Home page
st.title("Emotion Detection Lab")
st.markdown("""
Welcome to the **Emotion Detection Lab** — a machine learning workbench for classifying
**happy** vs **neutral** facial expressions on the SMILE_PLUS dataset.

### How to Use

1. **Dataset Explorer** — Browse the dataset, view sample images and class distribution
2. **Feature Engineering** — Select and extract features (traditional + deep embeddings)
3. **Model Training** — Choose a classifier, tune hyperparameters, run 10-fold cross-validation
4. **Results** — View detailed metrics, confusion matrix, and ROC curves
5. **Model Comparison** — Compare all models on a leaderboard with the best model highlighted

### Pre-Trained Models

All models come **pre-trained with optimized hyperparameters**. You can view their results
immediately or re-train with your own custom settings.
""")

# Load and display pre-trained summary if available
import json
from src.config import METADATA_PATH

if METADATA_PATH.exists():
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    st.success(f"**Best Model:** {metadata['best_model']} — F1 = {metadata['best_f1']:.4f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Models Trained", len(metadata["models"]))
    col2.metric("Best F1 Score", f"{metadata['best_f1']:.4f}")
    col3.metric("Performance Mark", f"{metadata['models'][metadata['best_model']]['performance_mark']:.1f}/35")
else:
    st.info("No pre-trained models found. Run `python -m scripts.train_all` to train all models, or use the Model Training page.")

log.info("Home page viewed")
```

**Step 2: Verify** — `streamlit run app.py` (manual check in browser)

**Step 3: Commit**

```bash
git add app.py && git commit -m "feat: add Streamlit home page with auth and pre-trained model summary"
```

---

### Task 12: Page — Dataset Explorer

**Files:**
- Create: `pages/1_Dataset_Explorer.py`

**Step 1: Write pages/1_Dataset_Explorer.py**

Shows sample images, class distribution, image statistics.

```python
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from src.auth import require_auth
from src.data_loader import load_annotations, _find_image_file
from src.config import IMAGES_DIR
from src.logger import get_logger

user = require_auth()
log = get_logger(user=user["name"], context="general")
log.info("Dataset Explorer page viewed")

st.title("Dataset Explorer")

df = load_annotations()

# Class distribution
st.subheader("Class Distribution")
col1, col2, col3 = st.columns(3)
col1.metric("Total Images", len(df))
col2.metric("Happy", len(df[df["label"] == "happy"]))
col3.metric("Neutral", len(df[df["label"] == "neutral"]))

fig = px.pie(df, names="label", title="Class Distribution",
             color="label", color_discrete_map={"happy": "#2ecc71", "neutral": "#3498db"})
st.plotly_chart(fig, use_container_width=True)

# Sample images
st.subheader("Sample Images")
n_samples = st.slider("Number of samples per class", 2, 10, 5)

for label in ["happy", "neutral"]:
    st.markdown(f"**{label.capitalize()}**")
    samples = df[df["label"] == label].sample(n=n_samples, random_state=42)
    cols = st.columns(n_samples)
    for i, (_, row) in enumerate(samples.iterrows()):
        img_path = _find_image_file(row["filename"])
        if img_path:
            cols[i].image(str(img_path), caption=row["filename"], width=120)

# Image statistics
st.subheader("Image Statistics")
sample_paths = [_find_image_file(f) for f in df["filename"].head(20)]
sample_paths = [p for p in sample_paths if p]
if sample_paths:
    widths, heights = [], []
    for p in sample_paths:
        img = Image.open(p)
        widths.append(img.width)
        heights.append(img.height)
    st.write(f"**Resolution range:** {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
    st.write(f"**Mode:** Grayscale (1 channel)")
    st.write(f"**Format:** JPEG")
```

**Step 2: Commit**

```bash
git add pages/1_Dataset_Explorer.py && git commit -m "feat: add dataset explorer page with class distribution and sample images"
```

---

### Task 13: Page — Feature Engineering

**Files:**
- Create: `pages/2_Feature_Engineering.py`

**Step 1: Write pages/2_Feature_Engineering.py**

UI for selecting feature methods, extracting features, and viewing feature dimensions.

```python
import streamlit as st
import numpy as np
import time
from src.auth import require_auth
from src.config import FEATURE_METHODS, FEATURES_DIR
from src.data_loader import load_images_for_traditional, load_images_for_deep
from src.features import extract_features
from src.logger import get_logger

user = require_auth()
log = get_logger(user=user["name"], context="features")

st.title("Feature Engineering")

st.markdown("""
Select feature extraction methods below. Traditional features work on grayscale images.
Deep embeddings use frozen pretrained models as feature extractors.
""")

# Feature descriptions
DESCRIPTIONS = {
    "HOG": "Histogram of Oriented Gradients — captures edge and shape information. Good for facial contours and smile detection.",
    "LBP": "Local Binary Patterns — encodes texture patterns at multiple scales. Detects micro-expressions and skin texture changes.",
    "Gabor": "Gabor filter bank — multi-scale, multi-orientation frequency features. Captures facial muscle patterns.",
    "Landmarks": "MediaPipe facial landmarks — geometric features (distances, angles) between key facial points like mouth corners.",
    "ConvNeXt-Tiny": "ConvNeXt-Tiny (ImageNet pretrained) — 768-dim general-purpose visual features from a modern CNN.",
    "CLIP-ViT-B/32": "CLIP Vision Transformer — 512-dim semantically rich embeddings from a vision-language model.",
    "InsightFace": "ArcFace/InsightFace — 512-dim face-specific embeddings trained on millions of face images.",
}

# Selection
st.subheader("Select Features")
selected = []
cols = st.columns(2)
for i, method in enumerate(FEATURE_METHODS):
    with cols[i % 2]:
        checked = st.checkbox(method, value=method in ["InsightFace", "CLIP-ViT-B/32"],
                              help=DESCRIPTIONS.get(method, ""))
        if checked:
            selected.append(method)

if not selected:
    st.warning("Select at least one feature method.")
    st.stop()

st.write(f"**Selected:** {', '.join(selected)}")

# Check for cached features
cached_status = {}
for method in selected:
    cache_path = FEATURES_DIR / f"{method.replace('/', '_')}.npy"
    cached_status[method] = cache_path.exists()

if all(cached_status.values()):
    st.success("All selected features are cached. Click 'Load Features' to use them.")
else:
    uncached = [m for m, c in cached_status.items() if not c]
    st.info(f"Need to extract: {', '.join(uncached)}. Cached: {', '.join(m for m, c in cached_status.items() if c) or 'none'}")

# Extract / Load
if st.button("Extract / Load Features", type="primary"):
    with st.spinner("Loading images..."):
        images_gray, labels, _ = load_images_for_traditional()
        images_color, _, _ = load_images_for_deep()
        log.info(f"User extracting features: {selected}")

    with st.spinner("Extracting features..."):
        start = time.time()
        X = extract_features(images_gray, images_color, selected)
        elapsed = time.time() - start

    st.session_state["features"] = X
    st.session_state["labels"] = labels
    st.session_state["feature_methods"] = selected

    st.success(f"Features extracted in {elapsed:.1f}s")
    st.write(f"**Shape:** {X.shape[0]} samples x {X.shape[1]} features")
    log.info(f"Features extracted: {selected}, shape={X.shape}, time={elapsed:.1f}s")

# Show current features if loaded
if "features" in st.session_state:
    st.subheader("Current Features")
    X = st.session_state["features"]
    st.write(f"**Methods:** {', '.join(st.session_state['feature_methods'])}")
    st.write(f"**Shape:** {X.shape}")
    st.write(f"**Stats:** mean={X.mean():.4f}, std={X.std():.4f}, min={X.min():.4f}, max={X.max():.4f}")
```

**Step 2: Commit**

```bash
git add pages/2_Feature_Engineering.py && git commit -m "feat: add feature engineering page with selection UI and caching"
```

---

### Task 14: Page — Model Training

**Files:**
- Create: `pages/3_Model_Training.py`

**Step 1: Write pages/3_Model_Training.py**

Model selection, hyperparameter tuning with sliders, training trigger, and results display. Loads pre-trained results by default.

```python
import streamlit as st
import json
import numpy as np
import time
from src.auth import require_auth
from src.config import METADATA_PATH, FEATURES_DIR
from src.models import MODEL_REGISTRY, DEEP_MODEL_REGISTRY, create_model, create_deep_model
from src.evaluation import cross_validate
from src.data_loader import load_images_for_traditional, load_images_for_deep
from src.features import extract_features
from src.logger import get_logger

user = require_auth()
log = get_logger(user=user["name"], context="training")

st.title("Model Training")

# Model type selection
model_type = st.radio("Model Type", ["Traditional ML", "Deep Learning (Linear Probe)"], horizontal=True)

if model_type == "Traditional ML":
    model_name = st.selectbox("Classifier", list(MODEL_REGISTRY.keys()))
    registry = MODEL_REGISTRY[model_name]

    # Feature selection for traditional models
    st.subheader("Features")
    if "features" in st.session_state:
        st.success(f"Using features from Feature Engineering page: {st.session_state['feature_methods']}")
        use_session_features = True
    else:
        st.info("No features in session. Select features to extract:")
        from src.config import FEATURE_METHODS
        trad_features = st.multiselect("Feature methods", FEATURE_METHODS,
                                        default=["InsightFace", "CLIP-ViT-B/32"])
        use_session_features = False

    # Hyperparameters
    st.subheader("Hyperparameters")
    custom_params = {}
    for param_name, param_config in registry["param_grid"].items():
        st.markdown(f"**{param_name}**: {param_config['desc']}")
        if param_config["type"] == "float":
            if param_config.get("log"):
                import math
                val = st.slider(param_name, float(param_config["min"]), float(param_config["max"]),
                                float(param_config["default"]), format="%.4f", key=f"{model_name}_{param_name}")
            else:
                val = st.slider(param_name, float(param_config["min"]), float(param_config["max"]),
                                float(param_config["default"]),
                                step=float(param_config.get("step", 0.01)), key=f"{model_name}_{param_name}")
            custom_params[param_name] = val
        elif param_config["type"] == "int":
            val = st.slider(param_name, param_config["min"], param_config["max"],
                            param_config["default"], step=param_config.get("step", 1),
                            key=f"{model_name}_{param_name}")
            custom_params[param_name] = val
        elif param_config["type"] == "select":
            val = st.selectbox(param_name, param_config["options"],
                               index=param_config["options"].index(param_config["default"]),
                               key=f"{model_name}_{param_name}")
            custom_params[param_name] = val

    # Train button
    if st.button("Train Model (10-Fold CV)", type="primary"):
        with st.spinner("Preparing data..."):
            if use_session_features:
                X = st.session_state["features"]
                y = st.session_state["labels"]
            else:
                images_gray, y, _ = load_images_for_traditional()
                images_color, _, _ = load_images_for_deep()
                X = extract_features(images_gray, images_color, trad_features)

        model = create_model(model_name, custom_params)
        log.info(f"Training {model_name} with params {custom_params}")

        with st.spinner(f"Running 10-fold CV with {model_name}..."):
            results = cross_validate(model, X, y, user=user["name"])

        # Store results
        if "training_results" not in st.session_state:
            st.session_state["training_results"] = {}
        run_key = f"{model_name} (custom)"
        st.session_state["training_results"][run_key] = {
            "model_name": model_name,
            "params": custom_params,
            "cv_results": results,
            "type": "traditional",
        }

        st.success(f"Training complete! F1 = {results['f1']:.4f}")
        _display_results(results)

else:  # Deep Learning
    model_name = st.selectbox("Model", list(DEEP_MODEL_REGISTRY.keys()))
    registry = DEEP_MODEL_REGISTRY[model_name]

    st.subheader("Hyperparameters")
    custom_params = {}
    for param_name, param_config in registry["param_grid"].items():
        st.markdown(f"**{param_name}**: {param_config['desc']}")
        if param_config["type"] == "float":
            val = st.slider(param_name, float(param_config["min"]), float(param_config["max"]),
                            float(param_config["default"]), format="%.5f", key=f"deep_{model_name}_{param_name}")
            custom_params[param_name] = val
        elif param_config["type"] == "int":
            val = st.slider(param_name, param_config["min"], param_config["max"],
                            param_config["default"], step=param_config.get("step", 1),
                            key=f"deep_{model_name}_{param_name}")
            custom_params[param_name] = val
        elif param_config["type"] == "select":
            options = [str(o) for o in param_config["options"]]
            val = st.selectbox(param_name, param_config["options"],
                               index=param_config["options"].index(param_config["default"]),
                               key=f"deep_{model_name}_{param_name}")
            custom_params[param_name] = val

    if st.button("Train Deep Model (10-Fold CV)", type="primary"):
        feature_source = registry["feature_source"]
        with st.spinner(f"Extracting {feature_source} embeddings..."):
            images_gray, y, _ = load_images_for_traditional()
            images_color, _, _ = load_images_for_deep()
            X = extract_features(images_gray, images_color, [feature_source])

        model, _ = create_deep_model(model_name, custom_params)
        log.info(f"Training {model_name} with params {custom_params}")

        with st.spinner(f"Running 10-fold CV with {model_name}..."):
            results = cross_validate(model, X, y, scale=False, user=user["name"])

        if "training_results" not in st.session_state:
            st.session_state["training_results"] = {}
        run_key = f"{model_name} (custom)"
        st.session_state["training_results"][run_key] = {
            "model_name": model_name,
            "params": custom_params,
            "cv_results": results,
            "type": "deep",
        }

        st.success(f"Training complete! F1 = {results['f1']:.4f}")
        _display_results(results)


def _display_results(results):
    """Display training results inline."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("F1 Score", f"{results['f1']:.4f}")
    col2.metric("Precision", f"{results['precision']:.4f}")
    col3.metric("Recall", f"{results['recall']:.4f}")
    col4.metric("FPR", f"{results['fpr']:.4f}")

    st.metric("Performance Mark", f"{results['performance_mark']:.1f} / 35")

    # Confusion matrix
    import plotly.figure_factory as ff
    cm = results["confusion_matrix"]
    fig = ff.create_annotated_heatmap(
        z=cm, x=["Neutral", "Happy"], y=["Neutral", "Happy"],
        colorscale="Blues", showscale=True,
    )
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig, use_container_width=True)
```

NOTE: The `_display_results` helper function should be defined BEFORE it's called. Move it to the top of the file or restructure as needed.

**Step 2: Commit**

```bash
git add pages/3_Model_Training.py && git commit -m "feat: add model training page with hyperparameter UI and 10-fold CV"
```

---

### Task 15: Page — Results

**Files:**
- Create: `pages/4_Results.py`

**Step 1: Write pages/4_Results.py**

Shows detailed results for a selected model run. Loads from pre-trained artifacts or session.

```python
import streamlit as st
import json
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from src.auth import require_auth
from src.config import METADATA_PATH
from src.logger import get_logger

user = require_auth()
log = get_logger(user=user["name"], context="general")

st.title("Results Dashboard")

# Collect available results
available = {}

# From pre-trained artifacts
if METADATA_PATH.exists():
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    for key, model_info in metadata["models"].items():
        available[f"[Pre-trained] {key}"] = model_info

# From session
if "training_results" in st.session_state:
    for key, run in st.session_state["training_results"].items():
        available[f"[Custom] {key}"] = run["cv_results"]

if not available:
    st.info("No results available. Train a model or run `python -m scripts.train_all`.")
    st.stop()

# Select model
selected_key = st.selectbox("Select model run", list(available.keys()))
results = available[selected_key]

log.info(f"Viewing results for {selected_key}")

# Metrics
st.subheader("Performance Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("F1 Score", f"{results['f1']:.4f}")
col2.metric("Precision", f"{results['precision']:.4f}")
col3.metric("Recall", f"{results['recall']:.4f}")
col4.metric("FPR", f"{results['fpr']:.4f}")
col5.metric("Mark", f"{results['performance_mark']:.1f}/35")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = results["confusion_matrix"]
fig = ff.create_annotated_heatmap(
    z=cm, x=["Neutral (Pred)", "Happy (Pred)"], y=["Neutral (True)", "Happy (True)"],
    colorscale="Blues", showscale=True,
)
fig.update_layout(width=500, height=400)
st.plotly_chart(fig, use_container_width=False)

# Per-fold results
if "fold_results" in results:
    st.subheader("Per-Fold Results")
    import pandas as pd
    fold_df = pd.DataFrame(results["fold_results"])
    st.dataframe(fold_df.style.format({"f1": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}"}),
                 use_container_width=True)

    # Fold F1 chart
    fig = go.Figure(data=go.Bar(x=[f"Fold {r['fold']}" for r in results["fold_results"]],
                                 y=[r["f1"] for r in results["fold_results"]],
                                 marker_color="#3498db"))
    fig.update_layout(title="F1 Score per Fold", yaxis_title="F1", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

# ROC Curve
if results.get("roc"):
    st.subheader("ROC Curve")
    roc = results["roc"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                             name=f"ROC (AUC={roc['auc']:.4f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"),
                             name="Random"))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)
```

**Step 2: Commit**

```bash
git add pages/4_Results.py && git commit -m "feat: add results dashboard with confusion matrix, per-fold results, ROC curve"
```

---

### Task 16: Page — Model Comparison

**Files:**
- Create: `pages/5_Model_Comparison.py`

**Step 1: Write pages/5_Model_Comparison.py**

Leaderboard comparing all models. Highlights recommended (highest F1) model.

```python
import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.auth import require_auth
from src.config import METADATA_PATH
from src.logger import get_logger

user = require_auth()
log = get_logger(user=user["name"], context="general")

st.title("Model Comparison")

# Collect all results
all_results = {}

if METADATA_PATH.exists():
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    for key, info in metadata["models"].items():
        all_results[key] = info

if "training_results" in st.session_state:
    for key, run in st.session_state["training_results"].items():
        all_results[f"[Custom] {key}"] = {
            **run["cv_results"],
            "model_name": run["model_name"],
            "type": run["type"],
            "features": run.get("features", []),
        }

if not all_results:
    st.info("No models to compare. Train models first.")
    st.stop()

# Build leaderboard dataframe
rows = []
for key, res in all_results.items():
    rows.append({
        "Run": key,
        "Model": res.get("model_name", key),
        "Type": res.get("type", "unknown"),
        "Features": ", ".join(res.get("features", [])),
        "F1": res["f1"],
        "Precision": res["precision"],
        "Recall": res["recall"],
        "FPR": res["fpr"],
        "Mark (/35)": res["performance_mark"],
    })

df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
df.index += 1  # 1-based ranking

# Highlight best
best_idx = df["F1"].idxmax()
best_model = df.loc[best_idx]

st.success(f"**Recommended Model:** {best_model['Run']} — F1 = {best_model['F1']:.4f} — Mark = {best_model['Mark (/35)']:.1f}/35")

# Leaderboard table
st.subheader("Leaderboard")
st.dataframe(
    df.style.format({
        "F1": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
        "FPR": "{:.4f}", "Mark (/35)": "{:.1f}",
    }).highlight_max(subset=["F1"], color="#2ecc71")
    .highlight_min(subset=["FPR"], color="#2ecc71"),
    use_container_width=True,
    height=min(len(df) * 40 + 40, 600),
)

# F1 comparison chart
st.subheader("F1 Score Comparison")
fig = px.bar(df, x="Run", y="F1", color="Type",
             color_discrete_map={"traditional": "#3498db", "deep": "#e74c3c"},
             title="F1 Score by Model")
fig.update_layout(xaxis_tickangle=-45, yaxis_range=[0, 1])
st.plotly_chart(fig, use_container_width=True)

# Metrics radar chart for top 5
st.subheader("Top 5 Models — Metric Comparison")
top5 = df.head(5)
fig = go.Figure()
for _, row in top5.iterrows():
    fig.add_trace(go.Scatterpolar(
        r=[row["F1"], row["Precision"], row["Recall"], 1 - row["FPR"]],
        theta=["F1", "Precision", "Recall", "1-FPR"],
        fill="toself", name=row["Run"][:30],
    ))
fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                  showlegend=True, height=500)
st.plotly_chart(fig, use_container_width=True)

# Filter by type
st.subheader("Filter by Model Type")
type_filter = st.multiselect("Show types", df["Type"].unique().tolist(),
                              default=df["Type"].unique().tolist())
filtered = df[df["Type"].isin(type_filter)]
st.dataframe(filtered.style.format({
    "F1": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
    "FPR": "{:.4f}", "Mark (/35)": "{:.1f}",
}), use_container_width=True)

log.info(f"Model comparison viewed, {len(df)} models")
```

**Step 2: Commit**

```bash
git add pages/5_Model_Comparison.py && git commit -m "feat: add model comparison page with leaderboard and visualizations"
```

---

### Task 17: Run Training Pipeline & Verify

**Step 1: Run the training script**

```bash
python -m scripts.train_all
```

This will take 10-30 minutes depending on hardware (deep feature extraction is the bottleneck).

**Step 2: Verify artifacts**

```bash
ls artifacts/features/ artifacts/models/ artifacts/results/
cat artifacts/metadata.json | python -m json.tool | head -20
```

**Step 3: Launch Streamlit and verify all pages**

```bash
streamlit run app.py
```

Walk through each page and verify functionality.

**Step 4: Commit artifacts metadata** (not the large model files)

```bash
git add artifacts/metadata.json && git commit -m "feat: add pre-trained model metadata"
```

---

### Task 18: Final Integration & Polish

**Step 1:** Ensure `.gitignore` excludes `artifacts/models/`, `artifacts/features/`, `logs/`, `.streamlit/secrets.toml`

**Step 2:** Verify all pages work with pre-trained data loaded

**Step 3:** Run the app end-to-end: login -> explore dataset -> view pre-trained results -> re-train a model -> compare

**Step 4: Final commit**

```bash
git add -A && git commit -m "feat: complete emotion detection web app with all models, auth, and logging"
```
