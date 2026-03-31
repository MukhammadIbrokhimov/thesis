"""
Run all classifiers on Dataset 2 (Diseases and Symptoms Dataset).

Uses fixed hyperparameters (best params from Dataset 1) instead of GridSearchCV
to avoid prohibitive training time on 150K samples with 727 classes.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing import load_processed_data

RANDOM_STATE = 42
RESULTS_DIR = Path(__file__).parent.parent / "results"


CLASSIFIERS = [
    {
        "name": "Naive Bayes",
        "needs_scaling": False,
        "model": BernoulliNB(alpha=0.01),
        "best_params": {"alpha": 0.01},
        "get_importance": None,
    },
    {
        "name": "Logistic Regression",
        "needs_scaling": True,
        "model": LogisticRegression(
            C=0.01,
            solver="lbfgs",
            max_iter=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "best_params": {"C": 0.01, "solver": "lbfgs"},
        "get_importance": None,
    },
    {
        "name": "SVM",
        "needs_scaling": True,
        # SGDClassifier(hinge) approximates linear SVM in linear time — necessary for 150K samples
        "model": SGDClassifier(
            loss="hinge",
            alpha=1e-4,
            max_iter=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "best_params": {"loss": "hinge", "alpha": 1e-4, "note": "SGD approx of LinearSVM"},
        "get_importance": None,
    },
    {
        "name": "Random Forest",
        "needs_scaling": False,
        "model": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "best_params": {"max_depth": 10, "min_samples_split": 2, "n_estimators": 100},
        "get_importance": "feature_importances_",
    },
    {
        "name": "XGBoost",
        "needs_scaling": False,
        "model": XGBClassifier(
            n_estimators=20,
            max_depth=3,
            learning_rate=0.3,
            subsample=0.8,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "best_params": {"learning_rate": 0.3, "max_depth": 3, "n_estimators": 20, "subsample": 0.8},
        "get_importance": "feature_importances_",
    },
]


def evaluate(model, X_test, y_test, best_params):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "best_params": best_params,
    }


def run_experiments():
    print("=" * 60)
    print("Running experiments on Dataset 2 (fixed hyperparameters)")
    print("=" * 60)

    print("\nLoading preprocessed data...")
    data = load_processed_data("dataset2")

    X_train = data["X_train"]
    X_test = data["X_test"]
    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Classes: {len(data['label_encoder'].classes_)}")

    # Ensure float32 so numpy uses BLAS (integer arrays skip BLAS and are extremely slow)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    X_train_scaled = X_train_scaled.astype(np.float32)
    X_test_scaled = X_test_scaled.astype(np.float32)

    results = {}

    for clf_cfg in CLASSIFIERS:
        name = clf_cfg["name"]
        model = clf_cfg["model"]

        print(f"\n{'-' * 40}")
        print(f"Training {name}...")

        X_tr = X_train_scaled if clf_cfg["needs_scaling"] else X_train
        X_te = X_test_scaled if clf_cfg["needs_scaling"] else X_test

        start = time.time()
        model.fit(X_tr, y_train)
        train_time = time.time() - start

        metrics = evaluate(model, X_te, y_test, clf_cfg["best_params"])
        metrics["train_time_seconds"] = round(train_time, 2)

        # Feature importance
        importance_attr = clf_cfg.get("get_importance")
        if importance_attr and hasattr(model, importance_attr):
            importance = getattr(model, importance_attr)
            indices = importance.argsort()[::-1][:15]
            metrics["top_features"] = [
                {"feature": feature_names[i], "importance": float(importance[i])}
                for i in indices
            ]

        results[name] = metrics

        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Time:      {train_time:.2f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "dataset2_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_experiments()
