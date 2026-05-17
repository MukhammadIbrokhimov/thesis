"""
Preprocess only Dataset 1 (skip Dataset 2). Used when regenerating Figure 5.1.

Run after placing dataset.csv at experiments/data/raw/dataset1/dataset.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    load_dataset1,
    preprocess_dataset,
    save_processed_data,
    PROCESSED_DIR,
)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Dataset 1...")
    X, y, features = load_dataset1()

    print("Preprocessing...")
    data = preprocess_dataset(X, y, features, "dataset1")

    print("Saving processed data...")
    save_processed_data(data, "dataset1")

    print("\nDone. Processed data is in experiments/data/processed/dataset1/")


if __name__ == "__main__":
    main()
