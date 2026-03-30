"""
Configuration file for ML experiments.
Contains hyperparameter search spaces and evaluation settings.
"""

# Hyperparameter search spaces (from Appendix A1)
HYPERPARAMETERS = {
    'naive_bayes': {
        'alpha': [0.01, 0.1, 0.5, 1.0]
    },
    'logistic_regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'saga'],
        'max_iter': [1000],
        'class_weight': ['balanced']
    },
    'svm': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'kernel': ['rbf'],
        'class_weight': ['balanced']
    },
    'random_forest': {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, 50, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced']
    },
    'xgboost': {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    }
}

# Cross-validation settings
CV_FOLDS = 5
SCORING_METRIC = 'f1_macro'
RANDOM_STATE = 42

# Train-test split
TEST_SIZE = 0.2
STRATIFY = True

# Evaluation metrics to compute
METRICS = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Dataset paths
DATASET1_PATH = 'data/raw/dataset1/'
DATASET2_PATH = 'data/raw/dataset2/'
PROCESSED_PATH = 'data/processed/'

# Output paths
RESULTS_PATH = 'results/'
FIGURES_PATH = 'figures/'
