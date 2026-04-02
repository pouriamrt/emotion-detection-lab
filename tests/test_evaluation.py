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
