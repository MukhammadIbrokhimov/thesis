"""
Preprocessing Pipeline for Disease-Symptom Classification

Implements Chapter 3.4 preprocessing steps:
1. Data loading and inspection
2. Missing value handling (impute 0)
3. Feature encoding (binary symptom vectors)
4. Label encoding (disease names to integers)
5. Train-test split (80/20 stratified)
6. Feature scaling (StandardScaler for LR/SVM)
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Paths
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset1():
    """Load and preprocess Dataset 1 (itachi9604/disease-symptom-description-dataset)."""
    path = RAW_DIR / "dataset1" / "dataset.csv"
    df = pd.read_csv(path)

    print(f"Dataset 1 loaded: {df.shape}")

    # Remove duplicates
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df)
    print(f"  Removed {n_removed} duplicate records")

    # Extract disease labels
    y = df["Disease"].values

    # Get all unique symptoms across symptom columns
    symptom_cols = [c for c in df.columns if c.startswith("Symptom_")]
    all_symptoms = set()
    for col in symptom_cols:
        symptoms = df[col].dropna().str.strip().str.lower().unique()
        all_symptoms.update(symptoms)
    all_symptoms.discard("")
    all_symptoms = sorted(all_symptoms)

    print(f"  Unique symptoms: {len(all_symptoms)}")

    # Create binary feature matrix
    X = np.zeros((len(df), len(all_symptoms)), dtype=np.int8)
    symptom_to_idx = {s: i for i, s in enumerate(all_symptoms)}

    for idx, row in df.iterrows():
        for col in symptom_cols:
            symptom = row[col]
            if pd.notna(symptom):
                symptom = symptom.strip().lower()
                if symptom in symptom_to_idx:
                    X[idx, symptom_to_idx[symptom]] = 1

    print(f"  Feature matrix shape: {X.shape}")

    return X, y, all_symptoms


def load_dataset2():
    """Load and preprocess Dataset 2 (dhivyeshrk/diseases-and-symptoms-dataset)."""
    path = RAW_DIR / "dataset2" / "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
    df = pd.read_csv(path)

    print(f"Dataset 2 loaded: {df.shape}")

    # Remove duplicates
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df)
    print(f"  Removed {n_removed} duplicate records")

    # Extract disease labels (first column)
    y = df["diseases"].values

    # Features are all other columns (already binary)
    feature_cols = [c for c in df.columns if c != "diseases"]
    X = df[feature_cols].fillna(0).values.astype(np.int8)

    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Unique diseases: {len(np.unique(y))}")

    return X, y, feature_cols


def preprocess_dataset(X, y, feature_names, dataset_name, min_samples_per_class=2):
    """
    Apply preprocessing pipeline to a dataset.

    Returns:
        dict with X_train, X_test, y_train, y_test,
             X_train_scaled, X_test_scaled, label_encoder, scaler, feature_names
    """
    print(f"\nPreprocessing {dataset_name}...")

    # Step 2: Missing value handling (already done during loading, but verify)
    X = np.nan_to_num(X, nan=0)

    # Filter out classes with too few samples (needed for stratified split)
    unique, counts = np.unique(y, return_counts=True)
    valid_classes = unique[counts >= min_samples_per_class]
    mask = np.isin(y, valid_classes)

    if mask.sum() < len(y):
        n_removed = len(y) - mask.sum()
        n_classes_removed = len(unique) - len(valid_classes)
        print(f"  Removed {n_removed} samples from {n_classes_removed} rare classes (< {min_samples_per_class} samples)")
        X = X[mask]
        y = y[mask]

    # Step 4: Label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"  Classes: {len(label_encoder.classes_)}")

    # Step 5: Train-test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        stratify=y_encoded,
        random_state=RANDOM_STATE
    )
    print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Step 6: Feature scaling (for LR/SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "feature_names": feature_names,
    }


def save_processed_data(data, dataset_name):
    """Save processed data to disk."""
    output_dir = PROCESSED_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save numpy arrays
    np.save(output_dir / "X_train.npy", data["X_train"])
    np.save(output_dir / "X_test.npy", data["X_test"])
    np.save(output_dir / "y_train.npy", data["y_train"])
    np.save(output_dir / "y_test.npy", data["y_test"])
    np.save(output_dir / "X_train_scaled.npy", data["X_train_scaled"])
    np.save(output_dir / "X_test_scaled.npy", data["X_test_scaled"])

    # Save label encoder and scaler
    with open(output_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(data["label_encoder"], f)
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(data["scaler"], f)

    # Save feature names
    with open(output_dir / "feature_names.pkl", "wb") as f:
        pickle.dump(data["feature_names"], f)

    print(f"  Saved to {output_dir}")


def load_processed_data(dataset_name):
    """Load processed data from disk."""
    input_dir = PROCESSED_DIR / dataset_name

    data = {
        "X_train": np.load(input_dir / "X_train.npy"),
        "X_test": np.load(input_dir / "X_test.npy"),
        "y_train": np.load(input_dir / "y_train.npy"),
        "y_test": np.load(input_dir / "y_test.npy"),
        "X_train_scaled": np.load(input_dir / "X_train_scaled.npy"),
        "X_test_scaled": np.load(input_dir / "X_test_scaled.npy"),
    }

    with open(input_dir / "label_encoder.pkl", "rb") as f:
        data["label_encoder"] = pickle.load(f)
    with open(input_dir / "scaler.pkl", "rb") as f:
        data["scaler"] = pickle.load(f)
    with open(input_dir / "feature_names.pkl", "rb") as f:
        data["feature_names"] = pickle.load(f)

    return data


def main():
    """Run preprocessing pipeline on both datasets."""
    print("=" * 60)
    print("Disease-Symptom Classification Preprocessing Pipeline")
    print("=" * 60)

    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Process Dataset 1
    print("\n" + "-" * 40)
    X1, y1, features1 = load_dataset1()
    data1 = preprocess_dataset(X1, y1, features1, "dataset1")
    save_processed_data(data1, "dataset1")

    # Process Dataset 2
    print("\n" + "-" * 40)
    X2, y2, features2 = load_dataset2()
    data2 = preprocess_dataset(X2, y2, features2, "dataset2")
    save_processed_data(data2, "dataset2")

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)

    # Summary
    print("\nSummary:")
    print(f"  Dataset 1: {data1['X_train'].shape[0]} train, {data1['X_test'].shape[0]} test, "
          f"{data1['X_train'].shape[1]} features, {len(data1['label_encoder'].classes_)} classes")
    print(f"  Dataset 2: {data2['X_train'].shape[0]} train, {data2['X_test'].shape[0]} test, "
          f"{data2['X_train'].shape[1]} features, {len(data2['label_encoder'].classes_)} classes")


if __name__ == "__main__":
    main()
