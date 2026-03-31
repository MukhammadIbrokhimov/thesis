# ML Experiments — Disease Prediction from Symptoms

Experimental pipeline for the thesis *"Comparing Machine Learning Methods for Predicting Diseases from Symptoms in Healthcare"*. Implements five classifiers evaluated on two public Kaggle datasets.

---

## Project Structure

```
experiments/
├── src/
│   ├── classifiers/
│   │   ├── base.py                  # BaseClassifier with GridSearchCV training
│   │   ├── naive_bayes.py
│   │   ├── logistic_regression.py
│   │   ├── svm.py
│   │   ├── random_forest.py
│   │   └── xgboost_clf.py
│   └── preprocessing.py             # Full preprocessing pipeline
├── experiments/
│   ├── run_dataset1.py              # Dataset 1 runner (with GridSearchCV)
│   └── run_dataset2.py              # Dataset 2 runner (fixed hyperparameters)
├── scripts/
│   ├── export_latex_tables.py       # JSON results → LaTeX booktabs tables
│   ├── generate_confusion_matrix.py # Confusion matrix figure
│   └── generate_feature_importance.py # Feature importance figure
├── data/
│   ├── raw/                         # Raw CSVs from Kaggle (not committed)
│   │   ├── dataset1/
│   │   └── dataset2/
│   └── processed/                   # Preprocessed .npy files (not committed)
├── results/
│   ├── dataset1_results.json
│   ├── dataset2_results.json
│   ├── table3.tex                   # LaTeX table for Dataset 1
│   └── table4.tex                   # LaTeX table for Dataset 2
├── config.py
└── requirements.txt
```

---

## Setup

### 1. Create virtual environment

```bash
cd experiments
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download datasets from Kaggle

Set up your Kaggle API token first (see `KAGGLE_SETUP.md` for detailed instructions).

```bash
# Dataset 1: Disease Symptom Prediction (~5,000 records, 132 symptoms, 41 classes)
kaggle datasets download -d itachi9604/disease-symptom-description-dataset \
  -p data/raw/dataset1/ --unzip

# Dataset 2: Diseases and Symptoms (~246,000 records, 377 symptoms, 727 classes)
kaggle datasets download -d dhivyeshrk/diseases-and-symptoms-dataset \
  -p data/raw/dataset2/ --unzip
```

### 4. Run preprocessing

```bash
python src/preprocessing.py
```

This saves train/test splits as `.npy` files under `data/processed/`.

---

## Running Experiments

### Dataset 1

Uses GridSearchCV with 5-fold stratified cross-validation to tune hyperparameters.

```bash
python experiments/run_dataset1.py
```

**Expected runtime:** ~5 minutes total on a modern laptop.

### Dataset 2

Uses **fixed hyperparameters** (best params from Dataset 1) instead of GridSearchCV.
GridSearchCV is not feasible on 151K samples x 727 classes — it would take many hours.

```bash
python experiments/run_dataset2.py
```

**Expected runtime:** ~10 minutes total.

| Classifier          | Approx. time |
|---------------------|-------------|
| Naive Bayes         | 2s          |
| Logistic Regression | 3 min       |
| SVM (SGD)           | 3 min       |
| Random Forest       | 3s          |
| XGBoost             | 3 min       |

> **Note on SVM for Dataset 2:** The standard `SVC(kernel='rbf')` and `LinearSVC` are
> computationally infeasible at this scale (727 one-vs-rest classifiers x 151K samples).
> Dataset 2 uses `SGDClassifier(loss='hinge')`, a stochastic approximation of a linear SVM.

---

## Generating Outputs

```bash
# LaTeX booktabs tables -> results/table3.tex and results/table4.tex
python scripts/export_latex_tables.py

# Confusion matrix figure -> ../thesis/figures/confusion_matrix.pdf
python scripts/generate_confusion_matrix.py

# Feature importance figure -> ../thesis/figures/feature_importance.pdf
python scripts/generate_feature_importance.py
```

---

## Results Summary

### Dataset 1 (41 classes)

| Classifier          | Accuracy | Precision | Recall  | F1      |
|---------------------|----------|-----------|---------|---------|
| Naive Bayes         | 98.36%   | 99.19%    | 98.78%  | 98.70%  |
| Logistic Regression | 100.00%  | 100.00%   | 100.00% | 100.00% |
| SVM (RBF)           | 100.00%  | 100.00%   | 100.00% | 100.00% |
| Random Forest       | 98.36%   | 99.19%    | 98.78%  | 98.70%  |
| XGBoost             | 95.08%   | 92.28%    | 93.90%  | 92.52%  |

### Dataset 2 (727 classes)

| Classifier          | Accuracy | Precision | Recall | F1     |
|---------------------|----------|-----------|--------|--------|
| Naive Bayes         | 84.11%   | 77.18%    | 77.00% | 75.97% |
| Logistic Regression | 81.82%   | 64.50%    | 80.43% | 67.33% |
| SVM (Linear, SGD)   | 69.56%   | 59.61%    | 64.58% | 54.86% |
| Random Forest       | 46.69%   | 55.92%    | 52.09% | 45.17% |
| XGBoost             | 79.84%   | 56.64%    | 54.07% | 54.63% |

---

## Troubleshooting

### Training appears frozen (near 0% CPU on macOS)

If you previously ran a script using `GridSearchCV(n_jobs=-1)` and killed it mid-run,
joblib worker processes may still be running in the background consuming all CPU silently.
Kill them with:

```bash
pkill -f "loky.backend.popen_loky_posix"
```

Then restart your experiment script.

### Very slow training on Dataset 2

The raw data arrays are stored as `int8`. Numpy's integer matmul bypasses BLAS and runs
element-by-element, making even simple classifiers take hours. The `run_dataset2.py`
script converts arrays to `float64` automatically before training — do not remove this step.

### `ModuleNotFoundError`

Make sure the virtual environment is activated:

```bash
source venv/bin/activate
```

### Kaggle `401 Unauthorized`

Your token may have expired. Create a new one: Kaggle → Settings → API → Create New Token,
then replace `~/.kaggle/kaggle.json`.

### Kaggle `403 Forbidden`

Visit the dataset page on Kaggle and click Download once to accept the terms of use,
then retry the API download.

---

## Classifier Implementations

All classifiers extend `BaseClassifier` in `src/classifiers/base.py`, which provides:

- `train(X, y)` — fits with GridSearchCV (used for Dataset 1)
- `evaluate(X_test, y_test)` — returns accuracy, precision, recall, F1 (macro-averaged)
- `get_feature_importance()` — returns feature importances where available (RF, XGBoost)

Each classifier exposes `get_param_grid()` for hyperparameter search and a `needs_scaling`
property (`True` for Logistic Regression and SVM).

---

## References

See `../thesis/references.bib` for full dataset and method citations.
