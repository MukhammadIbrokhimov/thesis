# ML Experiments for Thesis: Disease Prediction from Symptoms

This project implements the experimental pipeline for the thesis "Comparing Machine Learning Methods for Predicting Diseases from Symptoms in Healthcare".

## Project Structure

```
experiments/
├── src/                    # Source code
│   ├── classifiers/       # Classifier implementations
│   └── preprocessing.py   # Data preprocessing pipeline
├── experiments/           # Experiment runner scripts
├── scripts/               # Utility scripts (LaTeX export, visualization)
├── notebooks/             # Jupyter notebooks for exploration
├── data/                  # Data directory (not committed)
│   ├── raw/              # Raw datasets from Kaggle
│   └── processed/        # Preprocessed data
├── results/              # Experiment results (JSON format)
├── figures/              # Generated figures (PDF/PNG)
└── requirements.txt      # Python dependencies

## Setup Instructions

### 1. Create Virtual Environment

```bash
cd experiments
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Datasets

Download the following datasets from Kaggle:

**Dataset 1: Disease Symptom Prediction Dataset**
- Kaggle: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
- Save to: `data/raw/dataset1/`
- Expected: ~5,000 records, 132 symptoms, 41 diseases

**Dataset 2: Diseases and Symptoms Dataset**
- Kaggle: https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset
- Save to: `data/raw/dataset2/`
- Expected: ~246,000 records

You'll need a Kaggle account and API key. See: https://www.kaggle.com/docs/api

### 4. Run Preprocessing

```bash
python src/preprocessing.py
```

### 5. Run Experiments

```bash
# Run experiments on Dataset 1
python experiments/run_dataset1.py

# Run experiments on Dataset 2
python experiments/run_dataset2.py
```

### 6. Generate Visualizations

```bash
# Generate LaTeX tables
python scripts/export_latex_tables.py

# Generate confusion matrix
python scripts/generate_confusion_matrix.py

# Generate feature importance chart
python scripts/generate_feature_importance.py
```

## Experiment Details

### Classifiers
1. Naive Bayes (BernoulliNB)
2. Logistic Regression
3. Support Vector Machine (RBF kernel)
4. Random Forest
5. XGBoost

### Evaluation Metrics
- Accuracy
- Precision (macro-averaged)
- Recall (macro-averaged)
- F1-score (macro-averaged)

### Cross-Validation
- Stratified 5-fold cross-validation
- Hyperparameter tuning via GridSearchCV
- Scoring metric: macro-F1

### Class Imbalance Handling
- Class weighting (`class_weight='balanced'`)
- Stratified sampling for train-test splits

## Expected Runtime

- Dataset 1 preprocessing: < 1 minute
- Dataset 1 experiments: 5-10 minutes
- Dataset 2 preprocessing: < 5 minutes
- Dataset 2 experiments: 30-60 minutes (SVM is slowest)

## Output Files

- `results/dataset1_results.json` - Metrics for all classifiers on Dataset 1
- `results/dataset2_results.json` - Metrics for all classifiers on Dataset 2
- `results/table3.tex` - LaTeX booktabs table for Dataset 1
- `results/table4.tex` - LaTeX booktabs table for Dataset 2
- `figures/confusion_matrix.pdf` - Confusion matrix for XGBoost
- `figures/feature_importance.pdf` - Top 15 feature importance chart

## Troubleshooting

### Import errors
Make sure virtual environment is activated and all dependencies are installed.

### Memory errors on Dataset 2
SVM with large datasets can be memory-intensive. Reduce CV folds or use a smaller sample for initial testing.

### Kaggle API authentication
Set up `~/.kaggle/kaggle.json` with your API credentials.

## References

See thesis `references.bib` for full citations of datasets and methods.
