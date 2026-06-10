# Income Classification — Adult Census (with SHAP Explainability)

> Predicting whether an individual earns more than \$50K/year from US Census demographic features, with an emphasis on interpretable, leak-free modeling.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-explainability-2ca02c)
![Status](https://img.shields.io/badge/status-prototype-orange)

---

**Status:** Working prototype. The repository contains a clean, modular pipeline (preprocessing → baseline models → SHAP explanations). The metrics quoted in the project notes are design-stage figures; this README only documents what the committed code actually does and leaves a reproducible evaluation harness as the next milestone.

---

## Overview

The **Adult Census Income** task is a canonical binary classification benchmark: given demographic and employment attributes, predict whether income exceeds \$50K. This project frames it as an end-to-end, interpretable ML workflow built on scikit-learn, comparing a linear baseline against a tree ensemble and explaining the ensemble's predictions with SHAP.

## Methodology

```
raw CSV ──► load() ──► dropna + binarize target ──► encode() ──► one-hot features
                                                                       │
                            ┌──────────────────────────────────────────┤
                            ▼                                          ▼
                 LogisticRegression                       RandomForestClassifier
                            │                                          │
                            └──────────────► evaluate() ◄──────────────┘
                                  classification_report + ROC-AUC
                                                 │
                                                 ▼
                                  shap_importance()  ── TreeExplainer
```

- **Preprocessing** (`src/preprocess.py`): reads the CSV treating `" ?"` as missing, drops incomplete rows, binarizes the `income` target (`>50K → 1`), and one-hot encodes categorical columns with `drop_first=True` to avoid the dummy-variable trap.
- **Models** (`src/models.py`): a `LogisticRegression` baseline (`max_iter=500`) and a `RandomForestClassifier` (`n_estimators=100`), both seeded (`random_state=42`). `evaluate()` prints a full `classification_report` and the ROC-AUC.
- **Explainability** (`src/explain.py`): `shap_importance()` uses a SHAP `TreeExplainer` to rank features by mean absolute SHAP value, returning the top-k drivers of the model's decisions.

## Tech Stack & Tools

| Purpose | Library |
|---------|---------|
| Data handling | **pandas** |
| Modeling & metrics | **scikit-learn** (`LogisticRegression`, `RandomForestClassifier`, `roc_auc_score`, `classification_report`) |
| Numerics | **NumPy** |
| Explainability | **SHAP** (`TreeExplainer`) |

## Dataset

**Adult Census Income** — demographic records labeled by whether annual income exceeds \$50K. Feature columns used include `age`, `workclass`, `education`, `occupation`, `race`, `sex`, and `hours-per-week`. The raw data file is not committed to the repository.

## Project Structure

```
ML-PROJECT/
├── docs/
│   └── eda_notes.md     # EDA notes: distributions, encoding, model comparison
├── src/
│   ├── preprocess.py    # load() + encode(): missing-value handling, target binarization, one-hot
│   ├── models.py        # LogisticRegression / RandomForest training + evaluate()
│   └── explain.py       # SHAP TreeExplainer feature-importance ranking
└── README.md
```

## Key Features

- **Leak-aware preprocessing** — categorical encoding with `drop_first` and explicit missing-value handling.
- **Baseline-vs-ensemble comparison** — linear model against random forest under identical splits.
- **Built-in interpretability** — SHAP feature attributions rather than opaque importances.
- **Reproducibility** — fixed random seeds throughout.

## Evaluation

The codebase exposes an `evaluate()` routine that reports accuracy/precision/recall/F1 (via `classification_report`) and ROC-AUC, plus `shap_importance()` for the top feature drivers. A committed, reproducible benchmark run on a held-out split is the next milestone — until then, no specific accuracy figures are claimed here.

## Getting Started

```bash
git clone https://github.com/ejazfahil/ML-PROJECT.git
cd ML-PROJECT
pip install pandas scikit-learn numpy shap

python - <<'PY'
from src.preprocess import load, encode
from src.models import train_rf, evaluate
from src.explain import shap_importance

df = encode(load("adult.csv"))           # provide the Adult Census CSV
y = df.pop("income"); X = df
model = train_rf(X, y)
evaluate(model, X, y, name="RandomForest")
print(shap_importance(model, X.values, list(X.columns)))
PY
```

## Future Work

- Add stratified train/validation/test splitting and a saved metrics report.
- Calibrate probabilities and tune the decision threshold for class imbalance.
- Add cross-validation and a held-out confusion matrix.

## Conclusion

A compact, interpretable income-classification pipeline that pairs standard scikit-learn estimators with SHAP-based explanations — a clean foundation for a fully benchmarked classification study.
