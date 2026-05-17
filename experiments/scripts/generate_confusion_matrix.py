"""
Generate confusion matrix visualization for XGBoost on Dataset 1.

Outputs PDF to thesis/figures/confusion_matrix.pdf
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import load_processed_data
from src.classifiers import XGBoostClassifier


FIGURES_DIR = Path(__file__).parent.parent.parent / "thesis" / "figures"


def main():
    print("Loading Dataset 1...")
    data = load_processed_data("dataset1")

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    label_encoder = data["label_encoder"]

    print(f"Training XGBoost with best params...")
    clf = XGBoostClassifier()
    clf.train(X_train, y_train, verbose=0)

    print("Generating predictions...")
    y_pred = clf.predict(X_test)

    # Compute confusion matrix (normalized)
    cm = confusion_matrix(y_test, y_pred, normalize="true")

    # Get class names
    class_names = label_encoder.classes_

    # Create figure - larger canvas for readability with 41 classes
    fig, ax = plt.subplots(figsize=(18, 16))

    # Plot heatmap with thin gridlines to separate cells
    sns.heatmap(
        cm,
        annot=False,  # Too many cells for full annotation
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion", "shrink": 0.7},
        linewidths=0.3,
        linecolor="lightgray",
        square=True,
    )

    # Annotate diagonal (correct-classification rate per class) only
    for i in range(len(class_names)):
        ax.text(
            i + 0.5, i + 0.5,
            f"{cm[i, i]:.2f}",
            ha="center", va="center",
            fontsize=7,
            color="white" if cm[i, i] > 0.5 else "black",
        )

    ax.set_xlabel("Predicted Disease", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Disease", fontsize=14, fontweight="bold")
    ax.set_title(
        "Normalized Confusion Matrix - XGBoost on Dataset 1",
        fontsize=16, pad=12,
    )

    # Larger tick labels for readability
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()

    # Save high-resolution PDF (vector) and PNG (preview)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "confusion_matrix.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {output_path}")

    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")

    plt.close()


if __name__ == "__main__":
    main()
