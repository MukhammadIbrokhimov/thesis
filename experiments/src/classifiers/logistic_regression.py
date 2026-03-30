"""
Logistic Regression classifier with L2 regularization.
"""

from sklearn.linear_model import LogisticRegression
from .base import BaseClassifier, RANDOM_STATE


class LogisticRegressionClassifier(BaseClassifier):
    """Multi-class Logistic Regression with hyperparameter tuning."""

    @property
    def name(self):
        return "Logistic Regression"

    @property
    def needs_scaling(self):
        return True

    def get_model(self):
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    def get_param_grid(self):
        return {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "saga"],
        }
