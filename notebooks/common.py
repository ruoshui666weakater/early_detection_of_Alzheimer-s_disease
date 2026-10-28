"""Data loading, preprocessing and evaluation shared by the notebooks.

Every model is evaluated with the same nested cross-validation: hyperparameters
are tuned on the training part of each outer fold only, so the outer-fold
scores are an honest estimate of performance on unseen patients.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SUBMISSIONS_DIR = ROOT / "submissions"
FIGURES_DIR = ROOT / "report" / "revised_src" / "figures"
TABLES_DIR = ROOT / "report" / "revised_src" / "tables"

TARGET = "Diagnosis"
ID_COLUMNS = ["PatientID", "DoctorInCharge"]  # DoctorInCharge is the constant "XXXConfid"
CONTINUOUS = [
    "Age", "BMI", "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality",
    "SystolicBP", "DiastolicBP", "CholesterolTotal", "CholesterolLDL", "CholesterolHDL",
    "CholesterolTriglycerides", "MMSE", "FunctionalAssessment", "ADL",
]
NOMINAL = ["Ethnicity", "EducationLevel"]
BINARY = [
    "Gender", "Smoking", "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes",
    "Depression", "HeadInjury", "Hypertension", "MemoryComplaints", "BehavioralProblems",
    "Confusion", "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks",
    "Forgetfulness",
]
FEATURES = CONTINUOUS + NOMINAL + BINARY

SEED = 0
OUTER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
INNER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
TARGET_RECALL = 0.90  # screening operating point: catch at least 90% of AD cases

for d in (RESULTS_DIR, SUBMISSIONS_DIR, FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)


def load_train():
    df = pd.read_csv(DATA_DIR / "train.csv")
    assert set(df.columns) == set(FEATURES + ID_COLUMNS + [TARGET])
    return df[FEATURES], df[TARGET]


def load_test():
    df = pd.read_csv(DATA_DIR / "test.csv")
    return df["PatientID"], df[FEATURES]


def make_preprocessor(features=FEATURES):
    """Min-max scale continuous features, one-hot encode the two nominal ones, pass binaries through.

    Fitted inside each training fold, so no information leaks from validation rows.
    Tree models are invariant to the scaling; it is kept so all models see the same inputs.
    """
    return ColumnTransformer(
        [
            ("continuous", MinMaxScaler(), [c for c in CONTINUOUS if c in features]),
            ("nominal", OneHotEncoder(drop="first", sparse_output=False), [c for c in NOMINAL if c in features]),
            ("binary", "passthrough", [c for c in BINARY if c in features]),
        ],
        verbose_feature_names_out=False,
    )


def classification_metrics(y_true, y_pred, y_prob):
    """Metrics with Alzheimer's (Diagnosis = 1) as the positive class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "fpr": fp / (fp + tn),
        "fnr": fn / (fn + tp),
        "f1": 2 * tp / (2 * tp + fp + fn),
        "auc": roc_auc_score(y_true, y_prob),
    }


def recall_threshold(y_true, y_prob, target=TARGET_RECALL):
    """Largest probability threshold whose recall on (y_true, y_prob) is at least `target`."""
    _, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[np.argmax(tpr >= target)])


def nested_cv(search, X, y):
    """Evaluate a tuning procedure (e.g. GridSearchCV) with outer cross-validation.

    For each outer fold the search is refit on the training part, and a recall-targeted
    threshold is chosen from inner out-of-fold probabilities of the tuned model, so the
    outer test part is never used for any choice.

    Returns per-fold metrics (one row per fold and threshold rule), out-of-fold
    predictions for all rows, and the hyperparameters chosen in each fold.
    """
    fold_rows, params, oof = [], [], []
    for fold, (tr, te) in enumerate(OUTER_CV.split(X, y)):
        X_tr, y_tr, X_te, y_te = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
        fitted = clone(search).fit(X_tr, y_tr)
        prob = fitted.predict_proba(X_te)[:, 1]

        inner_prob = cross_val_predict(
            clone(fitted.best_estimator_), X_tr, y_tr, cv=INNER_CV, method="predict_proba"
        )[:, 1]
        threshold = recall_threshold(y_tr, inner_prob)

        for rule, thr in (("0.5", 0.5), ("recall>=0.90", threshold)):
            pred = (prob >= thr).astype(int)
            fold_rows.append({"fold": fold, "rule": rule, "threshold": thr,
                              **classification_metrics(y_te, pred, prob)})
        params.append({"fold": fold, **fitted.best_params_})
        oof.append(pd.DataFrame({"row": te, "fold": fold, "y": y_te.to_numpy(), "prob": prob,
                                 "pred_05": (prob >= 0.5).astype(int),
                                 "pred_recall": (prob >= threshold).astype(int)}))
    oof = pd.concat(oof).sort_values("row").reset_index(drop=True)
    return pd.DataFrame(fold_rows), oof, pd.DataFrame(params)


def summarize(folds):
    """Mean and standard deviation of each metric across outer folds, per threshold rule."""
    metrics = ["accuracy", "precision", "recall", "specificity", "fpr", "fnr", "f1", "auc"]
    return folds.groupby("rule")[metrics].agg(["mean", "std"]).round(4)


def save_results(name, folds, oof, params):
    folds.to_csv(RESULTS_DIR / f"{name}_folds.csv", index=False)
    oof.to_csv(RESULTS_DIR / f"{name}_oof.csv", index=False)
    params.to_csv(RESULTS_DIR / f"{name}_params.csv", index=False)


def write_submission(name, model):
    """Predict the Kaggle test set with threshold 0.5 (the competition scores accuracy)."""
    ids, X_test = load_test()
    out = pd.DataFrame({"PatientID": ids, "Diagnosis": model.predict(X_test)})
    path = SUBMISSIONS_DIR / f"{name}.csv"
    out.to_csv(path, index=False)
    return path.relative_to(ROOT)
