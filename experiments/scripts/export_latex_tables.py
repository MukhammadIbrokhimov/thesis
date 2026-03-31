"""
Export JSON results to LaTeX booktabs tables.

Generates:
  results/table3.tex  — Dataset 1 results
  results/table4.tex  — Dataset 2 results
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

CLASSIFIER_LABELS = {
    "Naive Bayes": "Naive Bayes",
    "Logistic Regression": "Logistic Regression",
    "SVM": "SVM",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
}

DATASET1_SVM_LABEL = "SVM (RBF kernel)"
DATASET2_SVM_LABEL = "SVM (Linear, SGD)"


def fmt(value):
    return f"{value * 100:.2f}"


def generate_table(results, caption, label, svm_label):
    rows = []
    for clf_name, label_str in CLASSIFIER_LABELS.items():
        if clf_name not in results:
            continue
        m = results[clf_name]
        display = svm_label if clf_name == "SVM" else label_str
        rows.append(
            f"{display:<28} & {fmt(m['accuracy'])} & {fmt(m['precision'])} "
            f"& {fmt(m['recall'])} & {fmt(m['f1'])} \\\\"
        )

    body = "\n".join(rows)
    return f"""\\begin{{table}}[H]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\\toprule
\\textbf{{Classifier}} & \\textbf{{Accuracy}} & \\textbf{{Precision}} & \\textbf{{Recall}} & \\textbf{{F1-score}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def main():
    with open(RESULTS_DIR / "dataset1_results.json") as f:
        ds1 = json.load(f)
    with open(RESULTS_DIR / "dataset2_results.json") as f:
        ds2 = json.load(f)

    table3 = generate_table(
        ds1,
        caption="Classification results on Dataset~1 (macro-averaged, \\%)",
        label="tab:results_dataset1",
        svm_label=DATASET1_SVM_LABEL,
    )
    table4 = generate_table(
        ds2,
        caption="Classification results on Dataset~2 (macro-averaged, \\%)",
        label="tab:results_dataset2",
        svm_label=DATASET2_SVM_LABEL,
    )

    out3 = RESULTS_DIR / "table3.tex"
    out4 = RESULTS_DIR / "table4.tex"

    out3.write_text(table3)
    out4.write_text(table4)

    print(f"Written: {out3}")
    print(f"Written: {out4}")
    print("\n--- table3.tex ---")
    print(table3)
    print("--- table4.tex ---")
    print(table4)


if __name__ == "__main__":
    main()
