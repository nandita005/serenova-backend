from pathlib import Path
import time
import warnings

import numpy as np
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "fetal_health"

ROOT = Path("data/processed") / DATASET_NAME
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RESULTS_DIR / "fetal_health_xgboost_best.pkl"

RANDOM_STATE = 42
N_CLASSES = 3


# ============================================================
# DATA LOADING
# ============================================================

def load_data():

    X_train = np.load(ROOT / "X_train.npy")
    y_train = np.load(ROOT / "y_train.npy")

    X_val = np.load(ROOT / "X_val.npy")
    y_val = np.load(ROOT / "y_val.npy")

    X_test = np.load(ROOT / "X_test.npy")
    y_test = np.load(ROOT / "y_test.npy")

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


# ============================================================
# METRICS
# ============================================================

def evaluate_model(model, X, y, name):

    pred = model.predict(X)

    accuracy = accuracy_score(y, pred)

    precision = precision_score(
        y,
        pred,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y,
        pred,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y,
        pred,
        average="macro",
        zero_division=0,
    )

    print()
    print(f"{name}")
    print("-" * 70)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Macro F1 : {f1:.4f}")

    print()
    print("CONFUSION MATRIX")
    print(confusion_matrix(y, pred))

    print()
    print("CLASSIFICATION REPORT")
    print(
        classification_report(
            y,
            pred,
            digits=4,
            zero_division=0,
        )
    )

    return accuracy, precision, recall, f1


# ============================================================
# XGBOOST MODEL
# ============================================================

def create_model(params):

    return xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=N_CLASSES,

        eval_metric="mlogloss",

        tree_method="hist",
        device="cuda",

        random_state=RANDOM_STATE,

        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],

        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],

        gamma=params["gamma"],

        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],

        n_jobs=1,
    )


# ============================================================
# CROSS VALIDATION
# ============================================================

def cross_validate(params, X, y):

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X, y),
        start=1,
    ):

        model = create_model(params)

        model.fit(
            X[train_idx],
            y[train_idx],
            verbose=False,
        )

        pred = model.predict(X[val_idx])

        score = accuracy_score(
            y[val_idx],
            pred,
        )

        scores.append(score)

        print(
            f"    Fold {fold}: "
            f"{score:.4f}"
        )

    return float(np.mean(scores)), float(np.std(scores))


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SERENOVA — XGBOOST FINE TUNING")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = load_data()

    print()
    print("DATA")
    print("-" * 70)

    print(f"Train X : {X_train.shape}")
    print(f"Train y : {y_train.shape}")

    print(f"Val X   : {X_val.shape}")
    print(f"Val y   : {y_val.shape}")

    print(f"Test X  : {X_test.shape}")
    print(f"Test y  : {y_test.shape}")

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    assert np.isfinite(X_train).all()
    assert np.isfinite(X_val).all()
    assert np.isfinite(X_test).all()

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    print()
    print("TRAIN CLASS DISTRIBUTION")

    classes, counts = np.unique(
        y_train,
        return_counts=True,
    )

    for c, n in zip(classes, counts):

        print(
            f"  Class {c}: {n}"
        )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    print()
    print("XGBOOST DEVICE")
    print("-" * 70)

    print(
        "XGBoost version:",
        xgb.__version__,
    )

    print(
        "Training device: CUDA GPU"
    )

    # --------------------------------------------------------
    # HYPERPARAMETER SEARCH
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 1 — HYPERPARAMETER SEARCH")
    print("=" * 70)

    parameter_sets = [

        # Baseline / stronger depth
        {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.03,
            "min_child_weight": 1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
        },

        {
            "n_estimators": 700,
            "max_depth": 4,
            "learning_rate": 0.025,
            "min_child_weight": 1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
        },

        # More regularization
        {
            "n_estimators": 700,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 1,
            "subsample": 0.95,
            "colsample_bytree": 1.0,
            "gamma": 0,
            "reg_alpha": 0.05,
            "reg_lambda": 2,
        },

        # Deeper trees
        {
            "n_estimators": 600,
            "max_depth": 5,
            "learning_rate": 0.03,
            "min_child_weight": 1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
        },

        {
            "n_estimators": 800,
            "max_depth": 5,
            "learning_rate": 0.02,
            "min_child_weight": 1,
            "subsample": 0.9,
            "colsample_bytree": 0.95,
            "gamma": 0,
            "reg_alpha": 0.01,
            "reg_lambda": 2,
        },

        # Stronger split regularization
        {
            "n_estimators": 700,
            "max_depth": 4,
            "learning_rate": 0.025,
            "min_child_weight": 2,
            "subsample": 0.95,
            "colsample_bytree": 0.95,
            "gamma": 0.05,
            "reg_alpha": 0.01,
            "reg_lambda": 2,
        },

        # More conservative
        {
            "n_estimators": 900,
            "max_depth": 3,
            "learning_rate": 0.02,
            "min_child_weight": 2,
            "subsample": 0.95,
            "colsample_bytree": 0.95,
            "gamma": 0,
            "reg_alpha": 0.01,
            "reg_lambda": 3,
        },
    ]

    results = []

    for i, params in enumerate(
        parameter_sets,
        start=1,
    ):

        print()
        print("-" * 70)
        print(f"CONFIGURATION {i}/{len(parameter_sets)}")
        print("-" * 70)

        print(params)

        start = time.time()

        cv_mean, cv_std = cross_validate(
            params,
            X_train,
            y_train,
        )

        elapsed = time.time() - start

        # Also check validation set
        model = create_model(params)

        model.fit(
            X_train,
            y_train,
            verbose=False,
        )

        val_pred = model.predict(X_val)

        val_accuracy = accuracy_score(
            y_val,
            val_pred,
        )

        val_f1 = f1_score(
            y_val,
            val_pred,
            average="macro",
            zero_division=0,
        )

        print()
        print(
            f"CV Accuracy : {cv_mean:.4f}"
        )

        print(
            f"CV Std      : {cv_std:.4f}"
        )

        print(
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        print(
            f"Val F1      : {val_f1:.4f}"
        )

        print(
            f"Time        : {elapsed:.2f}s"
        )

        results.append(
            {
                "params": params,
                "cv_accuracy": cv_mean,
                "cv_std": cv_std,
                "val_accuracy": val_accuracy,
                "val_f1": val_f1,
            }
        )

    # --------------------------------------------------------
    # SELECT BEST CONFIG
    # --------------------------------------------------------

    # Primary criterion = validation accuracy.
    # Secondary = CV accuracy.
    #
    # Test set is NOT used here.

    results.sort(
        key=lambda r: (
            r["val_accuracy"],
            r["cv_accuracy"],
        ),
        reverse=True,
    )

    best = results[0]

    print()
    print("=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)

    print(
        f"Validation Accuracy : "
        f"{best['val_accuracy']:.4f}"
    )

    print(
        f"Validation Macro F1  : "
        f"{best['val_f1']:.4f}"
    )

    print(
        f"CV Accuracy          : "
        f"{best['cv_accuracy']:.4f}"
    )

    print(
        f"CV Std               : "
        f"{best['cv_std']:.4f}"
    )

    print()
    print("PARAMETERS")

    for key, value in best["params"].items():

        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # FINAL TRAINING
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 2 — FINAL MODEL")
    print("=" * 70)

    final_model = create_model(
        best["params"]
    )

    start = time.time()

    # IMPORTANT:
    # We now train on TRAIN + VALIDATION.
    #
    # The TEST set remains untouched.

    X_final = np.concatenate(
        [X_train, X_val],
        axis=0,
    )

    y_final = np.concatenate(
        [y_train, y_val],
        axis=0,
    )

    print(
        f"Final training X: "
        f"{X_final.shape}"
    )

    print(
        f"Final training y: "
        f"{y_final.shape}"
    )

    final_model.fit(
        X_final,
        y_final,
        verbose=False,
    )

    elapsed = time.time() - start

    print(
        f"Training time: "
        f"{elapsed:.2f}s"
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 3 — FINAL TEST")
    print("=" * 70)

    print(
        "IMPORTANT: "
        "This is the first use of the untouched test set."
    )

    test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(
        final_model,
        X_test,
        y_test,
        "FINAL TEST — XGBOOST",
    )

    # --------------------------------------------------------
    # SAVE ONLY FINAL MODEL
    # --------------------------------------------------------

    final_model.save_model(
        str(MODEL_PATH)
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"Best Validation Accuracy : "
        f"{best['val_accuracy']:.4f}"
    )

    print(
        f"Cross-Validation Accuracy: "
        f"{best['cv_accuracy']:.4f}"
    )

    print(
        f"Final Test Accuracy      : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Final Test Macro F1      : "
        f"{test_f1:.4f}"
    )

    print()

    if test_accuracy >= 0.95:

        print(
            "TARGET ACHIEVED: "
            "Test accuracy >= 95%"
        )

    else:

        print(
            "TARGET NOT YET ACHIEVED."
        )

        print(
            "Do NOT modify the test set "
            "or tune directly against it."
        )

    print()
    print(
        f"Model saved: {MODEL_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()