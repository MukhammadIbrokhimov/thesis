"""
Base classifier interface and common utilities.
"""

from abc import ABC, abstractmethod
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


CV_FOLDS = 5
RANDOM_STATE = 42
SCORING = "f1_macro"


class BaseClassifier(ABC):
    """Base class for all classifiers."""

    def __init__(self):
        self.model = None
        self.best_params = None
        self.cv_results = None

    @abstractmethod
    def get_model(self):
        """Return the base model instance."""
        pass

    @abstractmethod
    def get_param_grid(self):
        """Return the hyperparameter search grid."""
        pass

    @property
    @abstractmethod
    def name(self):
        """Return classifier name."""
        pass

    @property
    def needs_scaling(self):
        """Whether this classifier needs scaled features."""
        return False

    def train(self, X_train, y_train, verbose=1):
        """Train with hyperparameter tuning via GridSearchCV."""
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        grid_search = GridSearchCV(
            self.get_model(),
            self.get_param_grid(),
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
            verbose=verbose,
            refit=True,
        )

        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.cv_results = grid_search.cv_results_

        return self

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate model and return metrics dict."""
        y_pred = self.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "best_params": self.best_params,
        }

    def get_feature_importance(self):
        """Return feature importance if available."""
        return None
