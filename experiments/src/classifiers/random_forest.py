"""
Random Forest classifier.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
from .base import BaseClassifier, RANDOM_STATE


class RandomForestClassifier(BaseClassifier):
    """Random Forest with hyperparameter tuning and feature importance."""

    @property
    def name(self):
        return "Random Forest"

    def get_model(self):
        return RF(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    def get_param_grid(self):
        return {
            "n_estimators": [100, 200, 500],
            "max_depth": [10, 20, 50, None],
            "min_samples_split": [2, 5, 10],
        }

    def get_feature_importance(self):
        """Return feature importances from trained model."""
        if self.model is not None:
            return self.model.feature_importances_
        return None
