"""
Naive Bayes (BernoulliNB) classifier for binary symptom features.
"""

from sklearn.naive_bayes import BernoulliNB
from .base import BaseClassifier


class NaiveBayesClassifier(BaseClassifier):
    """BernoulliNB with hyperparameter tuning."""

    @property
    def name(self):
        return "Naive Bayes"

    def get_model(self):
        return BernoulliNB()

    def get_param_grid(self):
        return {
            "alpha": [0.01, 0.1, 0.5, 1.0],
        }
