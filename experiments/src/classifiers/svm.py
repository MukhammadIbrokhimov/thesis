"""
Support Vector Machine classifier with RBF kernel.
"""

from sklearn.svm import SVC
from .base import BaseClassifier, RANDOM_STATE


class SVMClassifier(BaseClassifier):
    """SVM with RBF kernel and hyperparameter tuning."""

    @property
    def name(self):
        return "SVM"

    @property
    def needs_scaling(self):
        return True

    def get_model(self):
        return SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    def get_param_grid(self):
        return {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.01, 0.1],
        }
