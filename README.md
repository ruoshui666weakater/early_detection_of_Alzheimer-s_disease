# Project Overview

Early detection of Alzheimer’s disease is crucial for timely intervention and treatment planning. This project explores various machine learning techniques, aiming to accurately predict the presence or progression of Alzheimer’s. We use both classical and ensemble-based models.

The goal is to:

Compare model performance (accuracy, sensitivity, specificity) across Logistic Regression, XGBoost, and Random Forest.

Identify key features for Alzheimer’s disease detection.

Provide an organized framework for replicable analysis, including data preprocessing, model training, hyperparameter tuning, and evaluation.

# Repository Structure

Since the total number of files are small, for coding convinience, all files (data files, notebooks and paper files) are in one single folder.

# Data Files

train.csv: Training dataset for model development.

test.csv: Test dataset for hyperparameter tuning.

sample_submission.csv: predicted dataset to be submitted for Kaggle evaluation.

# Notebooks

## project_LR.ipynb

Implements Logistic Regression for binary classification of Alzheimer’s presence.

Includes data preprocessing, feature scaling, and model evaluation.

## project_XGBoost.ipynb

Trains an XGBoost model, typically well-suited for tabular data.

Contains hyperparameter tuning steps and comparison with baseline methods.

## project_random_forest.ipynb

Similar as previous two, but not included in the paper.

