"""
Generate feature importance chart for XGBoost on Dataset 1.

Shows top 15 symptoms ranked by gain-based importance.
Outputs PDF to thesis/figures/feature_importance.pdf
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import load_processed_data
from src.classifiers import XGBoostClassifier


FIGURES_DIR = Path(__file__).parent.parent.parent / "thesis" / "figures"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    # Try to load from saved results first
    results_path = RESULTS_DIR / "dataset1_results.json"

    if results_path.exists():
        print("Loading feature importance from saved results...")
        with open(results_path) as f:
            results = json.load(f)

        if "top_features" in results.get("XGBoost", {}):
            top_features = results["XGBoost"]["top_features"]
        else:
            top_features = None
    else:
        top_features = None

    if top_features is None:
        print("Training XGBoost to get feature importance...")
        data = load_processed_data("dataset1")
        clf = XGBoostClassifier()
        clf.train(data["X_train"], data["y_train"], verbose=0)

        importance = clf.get_feature_importance()
        feature_names = data["feature_names"]
        indices = importance.argsort()[::-1][:15]
        top_features = [
            {"feature": feature_names[i], "importance": float(importance[i])}
            for i in indices
        ]

    # Extract data for plotting
    features = [f["feature"].replace("_", " ").title() for f in top_features]
    importances = [f["importance"] for f in top_features]

    # Reverse for horizontal bar chart (top feature at top)
    features = features[::-1]
    importances = importances[::-1]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Horizontal bar chart
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))[::-1]
    bars = ax.barh(features, importances, color=colors, edgecolor="navy", linewidth=0.5)

    ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
    ax.set_ylabel("Symptom", fontsize=12)
    ax.set_title("Top 15 Most Important Symptoms - XGBoost on Dataset 1", fontsize=14)

    # Add value labels on bars
    for bar, imp in zip(bars, importances):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{imp:.3f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0, max(importances) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "feature_importance.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {output_path}")

    # Also save PNG for quick preview
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")

    plt.close()


if __name__ == "__main__":
    main()
