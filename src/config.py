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
    "ConvNeXt-Tiny", "CLIP-ViT-B/32", "InsightFace",
]

TRADITIONAL_MODELS = [
    "SVM", "Random Forest", "XGBoost", "LightGBM",
    "Logistic Regression", "KNN",
]

DEEP_MODELS = [
    "ConvNeXt-Tiny (Fine-tune)", "CLIP Linear Probe", "InsightFace + FC",
]

ENSEMBLE_MODELS = ["Soft Voting", "Stacking"]
