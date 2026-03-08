"""Model registry and factory for emotion detection classifiers.

Provides traditional ML models (SVM, RF, XGBoost, LightGBM, LR, KNN),
a deep linear-probe MLP, and ensemble constructors.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import RANDOM_SEED
from src.logger import get_logger

log = get_logger(context="models")

# ---------------------------------------------------------------------------
# Traditional model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    "SVM": {
        "class": SVC,
        "default_params": {
            "C": 10,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
            "random_state": RANDOM_SEED,
        },
        "param_grid": {
            "C": {
                "type": "float",
                "min": 0.01,
                "max": 1000.0,
                "default": 10,
                "log": True,
                "desc": "Regularisation parameter",
            },
            "kernel": {
                "type": "select",
                "options": ["rbf", "linear", "poly", "sigmoid"],
                "default": "rbf",
                "desc": "Kernel type",
            },
            "gamma": {
                "type": "select",
                "options": ["scale", "auto"],
                "default": "scale",
                "desc": "Kernel coefficient",
            },
        },
    },
    "Random Forest": {
        "class": RandomForestClassifier,
        "default_params": {
            "n_estimators": 500,
            "max_depth": 20,
            "min_samples_split": 2,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        },
        "param_grid": {
            "n_estimators": {
                "type": "int",
                "min": 50,
                "max": 2000,
                "default": 500,
                "step": 50,
                "desc": "Number of trees",
            },
            "max_depth": {
                "type": "int",
                "min": 5,
                "max": 100,
                "default": 20,
                "step": 5,
                "desc": "Maximum tree depth",
            },
            "min_samples_split": {
                "type": "int",
                "min": 2,
                "max": 20,
                "default": 2,
                "step": 1,
                "desc": "Min samples to split a node",
            },
        },
    },
    "XGBoost": {
        "class": XGBClassifier,
        "default_params": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "eval_metric": "logloss",
            "random_state": RANDOM_SEED,
            "use_label_encoder": False,
        },
        "param_grid": {
            "n_estimators": {
                "type": "int",
                "min": 50,
                "max": 1000,
                "default": 300,
                "step": 50,
                "desc": "Number of boosting rounds",
            },
            "max_depth": {
                "type": "int",
                "min": 3,
                "max": 15,
                "default": 6,
                "step": 1,
                "desc": "Maximum tree depth",
            },
            "learning_rate": {
                "type": "float",
                "min": 0.001,
                "max": 1.0,
                "default": 0.1,
                "log": True,
                "desc": "Learning rate",
            },
            "subsample": {
                "type": "float",
                "min": 0.5,
                "max": 1.0,
                "default": 0.8,
                "desc": "Row subsample ratio",
            },
        },
    },
    "LightGBM": {
        "class": LGBMClassifier,
        "default_params": {
            "n_estimators": 300,
            "num_leaves": 31,
            "learning_rate": 0.1,
            "verbose": -1,
            "random_state": RANDOM_SEED,
        },
        "param_grid": {
            "n_estimators": {
                "type": "int",
                "min": 50,
                "max": 1000,
                "default": 300,
                "step": 50,
                "desc": "Number of boosting rounds",
            },
            "num_leaves": {
                "type": "int",
                "min": 15,
                "max": 127,
                "default": 31,
                "step": 2,
                "desc": "Maximum number of leaves per tree",
            },
            "learning_rate": {
                "type": "float",
                "min": 0.001,
                "max": 1.0,
                "default": 0.1,
                "log": True,
                "desc": "Learning rate",
            },
        },
    },
    "Logistic Regression": {
        "class": LogisticRegression,
        "default_params": {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 5000,
            "random_state": RANDOM_SEED,
        },
        "param_grid": {
            "C": {
                "type": "float",
                "min": 0.001,
                "max": 100.0,
                "default": 1.0,
                "log": True,
                "desc": "Inverse regularisation strength",
            },
            "penalty": {
                "type": "select",
                "options": ["l1", "l2"],
                "default": "l2",
                "desc": "Regularisation penalty",
            },
        },
    },
    "KNN": {
        "class": KNeighborsClassifier,
        "default_params": {
            "n_neighbors": 5,
            "weights": "distance",
            "metric": "euclidean",
            "n_jobs": -1,
        },
        "param_grid": {
            "n_neighbors": {
                "type": "int",
                "min": 1,
                "max": 50,
                "default": 5,
                "step": 2,
                "desc": "Number of neighbours",
            },
            "weights": {
                "type": "select",
                "options": ["uniform", "distance"],
                "default": "distance",
                "desc": "Weight function for prediction",
            },
            "metric": {
                "type": "select",
                "options": ["euclidean", "manhattan", "minkowski"],
                "default": "euclidean",
                "desc": "Distance metric",
            },
        },
    },
}


def create_model(name: str, params: dict | None = None):
    """Instantiate a traditional model from the registry.

    Args:
        name: Key in MODEL_REGISTRY.
        params: Optional custom hyperparameters (merged over defaults).

    Returns:
        Configured sklearn estimator instance.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    entry = MODEL_REGISTRY[name]
    merged = {**entry["default_params"]}

    if params:
        merged.update(params)

    # Special case: LR with l1 penalty requires liblinear solver
    if name == "Logistic Regression" and merged.get("penalty") == "l1":
        merged["solver"] = "liblinear"

    model = entry["class"](**merged)
    log.info(f"Created model '{name}' with params: {merged}")
    return model


# ---------------------------------------------------------------------------
# Deep linear probe
# ---------------------------------------------------------------------------


class _ProbeNet(nn.Module):
    """Small 2-layer MLP for linear probing on frozen embeddings."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepLinearProbe(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper around a 2-layer MLP trained on frozen embeddings.

    Parameters:
        input_dim: Dimensionality of input features.
        lr: Learning rate for AdamW.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        weight_decay: L2 regularisation for AdamW.
    """

    def __init__(
        self,
        input_dim: int = 512,
        lr: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
    ):
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay

    def fit(self, X, y):
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.classes_ = np.unique(y)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_ = _ProbeNet(self.input_dim).to(device)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.model_.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(self.model_(xb), yb)
                loss.backward()
                optimizer.step()

        self.device_ = device
        return self

    def predict_proba(self, X):
        self.model_.eval()
        X_scaled = self.scaler_.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device_)
        with torch.no_grad():
            logits = self.model_(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


DEEP_MODEL_REGISTRY: dict[str, dict] = {
    "ConvNeXt-Tiny (Fine-tune)": {
        "class": DeepLinearProbe,
        "default_params": {"input_dim": 768, "lr": 0.001, "epochs": 50, "batch_size": 32, "weight_decay": 1e-4},
        "feature_source": "ConvNeXt-Tiny",
        "param_grid": {
            "lr": {"type": "float", "min": 1e-5, "max": 1e-2, "default": 0.001, "log": True, "desc": "Learning rate"},
            "epochs": {"type": "int", "min": 10, "max": 200, "default": 50, "step": 10, "desc": "Training epochs"},
            "batch_size": {"type": "int", "min": 8, "max": 128, "default": 32, "step": 8, "desc": "Batch size"},
            "weight_decay": {"type": "float", "min": 1e-6, "max": 1e-2, "default": 1e-4, "log": True, "desc": "Weight decay"},
        },
    },
    "CLIP Linear Probe": {
        "class": DeepLinearProbe,
        "default_params": {"input_dim": 512, "lr": 0.001, "epochs": 50, "batch_size": 32, "weight_decay": 1e-4},
        "feature_source": "CLIP-ViT-B/32",
        "param_grid": {
            "lr": {"type": "float", "min": 1e-5, "max": 1e-2, "default": 0.001, "log": True, "desc": "Learning rate"},
            "epochs": {"type": "int", "min": 10, "max": 200, "default": 50, "step": 10, "desc": "Training epochs"},
            "batch_size": {"type": "int", "min": 8, "max": 128, "default": 32, "step": 8, "desc": "Batch size"},
            "weight_decay": {"type": "float", "min": 1e-6, "max": 1e-2, "default": 1e-4, "log": True, "desc": "Weight decay"},
        },
    },
    "InsightFace + FC": {
        "class": DeepLinearProbe,
        "default_params": {"input_dim": 512, "lr": 0.001, "epochs": 50, "batch_size": 32, "weight_decay": 1e-4},
        "feature_source": "InsightFace",
        "param_grid": {
            "lr": {"type": "float", "min": 1e-5, "max": 1e-2, "default": 0.001, "log": True, "desc": "Learning rate"},
            "epochs": {"type": "int", "min": 10, "max": 200, "default": 50, "step": 10, "desc": "Training epochs"},
            "batch_size": {"type": "int", "min": 8, "max": 128, "default": 32, "step": 8, "desc": "Batch size"},
            "weight_decay": {"type": "float", "min": 1e-6, "max": 1e-2, "default": 1e-4, "log": True, "desc": "Weight decay"},
        },
    },
}


def create_deep_model(name: str, params: dict | None = None):
    """Instantiate a deep linear-probe model from the registry.

    Args:
        name: Key in DEEP_MODEL_REGISTRY.
        params: Optional custom hyperparameters (merged over defaults).

    Returns:
        Tuple of (model_instance, feature_source_name).
    """
    if name not in DEEP_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown deep model '{name}'. "
            f"Available: {list(DEEP_MODEL_REGISTRY.keys())}"
        )

    entry = DEEP_MODEL_REGISTRY[name]
    merged = {**entry["default_params"]}
    if params:
        merged.update(params)

    model = entry["class"](**merged)
    feature_source = entry["feature_source"]
    log.info(f"Created deep model '{name}' (source: {feature_source}) with params: {merged}")
    return model, feature_source


# ---------------------------------------------------------------------------
# Ensemble constructors
# ---------------------------------------------------------------------------


def create_ensemble(base_models: dict, method: str = "voting"):
    """Create an ensemble classifier from named base models.

    Args:
        base_models: Dict mapping model names to fitted or unfitted estimators.
        method: "voting" for VotingClassifier (soft), "stacking" for
                StackingClassifier with LogisticRegression meta-learner.

    Returns:
        Ensemble estimator.
    """
    estimators = [(name, model) for name, model in base_models.items()]

    if method == "voting":
        ensemble = VotingClassifier(estimators=estimators, voting="soft")
        log.info(f"Created Soft Voting ensemble with {len(estimators)} models")
    elif method == "stacking":
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                max_iter=5000, random_state=RANDOM_SEED
            ),
            cv=5,
        )
        log.info(f"Created Stacking ensemble with {len(estimators)} models")
    else:
        raise ValueError(f"Unknown ensemble method '{method}'. Use 'voting' or 'stacking'.")

    return ensemble
