"""
XGBoost classifier.
"""

import numpy as np
from xgboost import XGBClassifier
from .base import BaseClassifier, RANDOM_STATE


class XGBoostClassifier(BaseClassifier):
    """XGBoost with hyperparameter tuning and feature importance."""

    @property
    def name(self):
        return "XGBoost"

    def get_model(self):
        return XGBClassifier(
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    def get_param_grid(self):
        return {
            "n_estimators": [100, 200, 500],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1.0],
        }

    def get_feature_importance(self):
        """Return feature importances from trained model."""
        if self.model is not None:
            return self.model.feature_importances_
        return None
