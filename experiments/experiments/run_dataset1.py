"""
Run all classifiers on Dataset 1 (Disease Symptom Prediction Dataset).

Trains 5 classifiers with hyperparameter tuning via GridSearchCV,
evaluates on held-out test set, and saves results to JSON.
"""

import json
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import load_processed_data
from src.classifiers import ALL_CLASSIFIERS


RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_experiments():
    """Run all classifiers on Dataset 1."""
    print("=" * 60)
    print("Running experiments on Dataset 1")
    print("=" * 60)

    # Load preprocessed data
    print("\nLoading preprocessed data...")
    data = load_processed_data("dataset1")

    X_train = data["X_train"]
    X_test = data["X_test"]
    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Classes: {len(data['label_encoder'].classes_)}")

    results = {}

    for clf_class in ALL_CLASSIFIERS:
        clf = clf_class()
        print(f"\n{'-' * 40}")
        print(f"Training {clf.name}...")

        # Select scaled or unscaled features
        if clf.needs_scaling:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test

        # Train with timing
        start_time = time.time()
        clf.train(X_tr, y_train, verbose=0)
        train_time = time.time() - start_time

        # Evaluate
        metrics = clf.evaluate(X_te, y_test)
        metrics["train_time_seconds"] = round(train_time, 2)

        # Get feature importance if available
        importance = clf.get_feature_importance()
        if importance is not None:
            # Get top 15 features
            feature_names = data["feature_names"]
            indices = importance.argsort()[::-1][:15]
            metrics["top_features"] = [
                {"feature": feature_names[i], "importance": float(importance[i])}
                for i in indices
            ]

        results[clf.name] = metrics

        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Time:      {train_time:.2f}s")
        print(f"  Best params: {metrics['best_params']}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "dataset1_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_experiments()
