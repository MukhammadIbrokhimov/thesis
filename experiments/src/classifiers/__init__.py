"""
Classifier implementations for disease-symptom prediction.
"""

from .naive_bayes import NaiveBayesClassifier
from .logistic_regression import LogisticRegressionClassifier
from .svm import SVMClassifier
from .random_forest import RandomForestClassifier
from .xgboost_clf import XGBoostClassifier

ALL_CLASSIFIERS = [
    NaiveBayesClassifier,
    LogisticRegressionClassifier,
    SVMClassifier,
    RandomForestClassifier,
    XGBoostClassifier,
]

__all__ = [
    "NaiveBayesClassifier",
    "LogisticRegressionClassifier",
    "SVMClassifier",
    "RandomForestClassifier",
    "XGBoostClassifier",
    "ALL_CLASSIFIERS",
]
