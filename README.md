# Early Detection of Alzheimer's Disease: From Logistic Regression to XGBoost

Course project for STA314 (Fall 2024), submitted to the course Kaggle competition.
We fit and compare logistic regression and XGBoost for binary classification of
Alzheimer's disease from demographic, lifestyle, clinical, and cognitive features.

**Team "Vegetable Dogs":** Haocheng Sun, Chi Zhang, Xile Chen

Full write-up: [`report/early_detection_of_alzheimers_disease.pdf`](report/early_detection_of_alzheimers_disease.pdf)

## Results

| Model | Accuracy | Precision | FPR | FNR | ROC AUC |
|---|---|---|---|---|---|
| Logistic regression (L1, C = 0.14)¹ | 0.7832 | 0.7655 | 0.0936 | 0.4417 | 0.8249 |
| **XGBoost**² | **0.9654** | **0.9521** | **0.0303** | **0.0414** | **0.9652** |

Positive class is Alzheimer's (`Diagnosis = 1`).

¹ C chosen by 5-fold CV (CV accuracy 0.7806); metrics computed on the training set at threshold 0.5.
The report's Table 4 lists precision 0.9064, FPR 0.2345 and FNR 0.2106: the notebook reads
sklearn's confusion matrix with `cm[0,0]` as TP, so those values are actually specificity,
false discovery rate and false omission rate. The values above are recomputed from the same
confusion matrix (TN = 881, FP = 91, FN = 235, TP = 297).
² Tuned by grid search over 32,000 configurations on a 25% validation split; final
parameters `n_estimators=36, max_depth=3, learning_rate=0.1, subsample=0.9, colsample_bytree=0.85`.
Metrics computed on that validation split.

**Kaggle (XGBoost submission):** public 0.95195 (rank 53), private 0.91025 (rank 108).

The most important XGBoost features are ADL, MMSE, and FunctionalAssessment, followed by
BehavioralProblems and MemoryComplaints.

## Repository structure

```
├── data/
│   ├── train.csv                  # 1,504 labelled patients (target: Diagnosis)
│   ├── test.csv                   # 645 unlabelled patients for Kaggle scoring
│   └── sample_submission.csv      # Kaggle submission format
├── notebooks/
│   ├── project_LR.ipynb           # logistic regression (L1 / L2, CV over C)
│   ├── project_XGBoost.ipynb      # XGBoost grid search (final model)
│   └── project_random_forest.ipynb  # random forest (exploratory, not in report)
├── submissions/                   # Kaggle prediction files written by the notebooks
└── report/
    ├── early_detection_of_alzheimers_disease.pdf
    ├── references.bib
    └── references/                # cited papers
```

## Running the notebooks

```bash
conda create -n alzheimer python=3.10
conda activate alzheimer
pip install -r requirements.txt
jupyter notebook notebooks/
```

Notebooks read data from `../data/` and write predictions to `../submissions/`, so run
them from the `notebooks/` directory. The XGBoost grid search takes roughly 45–60 minutes
on a 4-core laptop CPU.

## Data

Alzheimer's Disease Dataset (Rabie El Kharoua, 2023), distributed via the course Kaggle competition.
