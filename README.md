# Early Detection of Alzheimer's Disease: From Logistic Regression to XGBoost

Course project for STA314 (Fall 2024), submitted to the course Kaggle competition, revised in October 2026.
We compare logistic regression, random forest and XGBoost for classifying Alzheimer's disease (AD) from
demographic, lifestyle, clinical and cognitive features.

**Team "Vegetable Dogs":** Haocheng Sun, Chi Zhang, Xile Chen

- Revised report: [`report/revised_2026.pdf`](report/revised_2026.pdf) (LaTeX source in [`report/revised_src/`](report/revised_src/))
- Report as submitted in December 2024: [`report/submitted_2024.pdf`](report/submitted_2024.pdf) (student numbers redacted)

> **The dataset is synthetic.** Its author generated it for educational purposes, so the results describe
> the data generator, not real patients.

## Results

Nested 5-fold cross-validation on the 1,504 labelled records (hyperparameters tuned inside each outer fold),
mean ± standard deviation over outer folds, threshold 0.5. AD (`Diagnosis = 1`) is the positive class.

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC AUC |
|---|---|---|---|---|---|---|
| Logistic regression (L1) | 0.833 ± 0.023 | 0.796 ± 0.030 | 0.710 ± 0.063 | 0.900 ± 0.018 | 0.750 ± 0.042 | 0.897 ± 0.022 |
| Random forest | 0.953 ± 0.014 | 0.947 ± 0.030 | 0.921 ± 0.021 | 0.971 ± 0.017 | 0.933 ± 0.019 | 0.956 ± 0.010 |
| XGBoost | 0.957 ± 0.009 | 0.952 ± 0.020 | 0.927 ± 0.012 | 0.974 ± 0.011 | 0.939 ± 0.013 | 0.958 ± 0.010 |

- XGBoost and random forest are indistinguishable (paired per-fold accuracy difference 0.004 ± 0.006).
  XGBoost beats logistic regression in every fold, by 9.6 to 14.0 points of accuracy.
- **Screening threshold** (chosen inside the training folds to catch ≥ 90% of AD cases): logistic regression
  flags 28.5% of non-AD records as positive, XGBoost 2.3%, at similar recall (0.893 vs 0.912).
- **Why trees win:** AD risk changes in steps at MMSE = 24, FunctionalAssessment = 5 and ADL = 5
  (for example, 42.5% AD below MMSE 24 versus 3.6% above). Trees split at those cut-offs; a logistic model
  that is linear in the features cannot.
- The same five features drive both model families: FunctionalAssessment, ADL, MMSE, MemoryComplaints and
  BehavioralProblems.

**Kaggle:** the 2024 XGBoost submission ([`submissions/original_2024/xgboost.csv`](submissions/original_2024/xgboost.csv))
scored 0.95195 public (rank 53) and 0.91025 private (rank 108). The revised predictions in `submissions/`
have not been scored.

## Changes from the 2024 submission

Full list in Section 6 of the revised report. The main ones:

1. **Logistic regression omitted all 15 binary features**, including MemoryComplaints and BehavioralProblems.
   Under the same protocol, that feature set gives an accuracy of 0.772; with all features, 0.833.
2. **The logistic regression confusion matrix was mislabelled**: the reported precision of 0.906 was actually
   the specificity (true precision 0.766, false-negative rate 0.442).
3. **XGBoost's 96.5% was optimistically biased**: it was the best of 32,000 configurations, scored on the
   same validation split it was chosen on. The 2024 configuration scores 0.952 ± 0.006 under cross-validation.
4. Logistic regression was evaluated on its training data; the final XGBoost model was trained on only 75%
   of the labelled records; the threshold discussion confused precision with recall; and the synthetic
   nature of the data was not mentioned.

## Repository structure

```
├── data/
│   ├── train.csv                    # 1,504 labelled records (target: Diagnosis)
│   ├── test.csv                     # 645 unlabelled records scored on Kaggle
│   └── sample_submission.csv        # Kaggle submission format
├── notebooks/
│   ├── common.py                    # shared loading, preprocessing and nested-CV evaluation
│   ├── 01_eda.ipynb
│   ├── 02_logistic_regression.ipynb
│   ├── 03_xgboost.ipynb
│   ├── 04_random_forest.ipynb
│   └── 05_model_comparison.ipynb    # writes the report's tables and summary.json
├── results/                         # per-fold metrics, out-of-fold predictions, chosen hyperparameters
├── submissions/                     # revised Kaggle predictions; original_2024/ holds the submitted ones
└── report/
    ├── revised_2026.pdf
    ├── revised_src/                 # main.tex, references.bib, generated figures/ and tables/
    ├── submitted_2024.pdf
    └── references/                  # cited papers
```

## Reproducing

```bash
conda create -n alzheimer python=3.13
conda activate alzheimer
pip install -r requirements.txt

cd notebooks
for nb in 01_eda 02_logistic_regression 03_xgboost 04_random_forest 05_model_comparison; do
  jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb"
done

cd ../report/revised_src
latexmk -pdf main.tex && cp main.pdf ../revised_2026.pdf
```

The notebooks take about 4 minutes on an 8-core laptop, most of it the random forest search. All
randomness is seeded, so the results are identical across runs with the pinned package versions.

## Data

Rabie El Kharoua (2024), *Alzheimer's Disease Dataset*, Kaggle, [doi:10.34740/KAGGLE/DSV/8668279](https://doi.org/10.34740/KAGGLE/DSV/8668279).
Licensed under CC BY 4.0.
