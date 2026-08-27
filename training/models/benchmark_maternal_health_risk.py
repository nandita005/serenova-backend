from pathlib import Path
import time
import warnings

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from training.datasets.pytorch_datasets import SerenovaNumpyDataset


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "maternal_health_risk"

RESULTS_DIR = Path("data/results")
MODEL_PATH = RESULTS_DIR / "maternal_health_risk_best.json"

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# DATA LOADING
# ============================================================

def load_split(split):
    dataset = SerenovaNumpyDataset(
        dataset_name=DATASET_NAME,
        split=split,
    )

    X = np.asarray(dataset.X, dtype=np.float32)

    if dataset.y is None:
        raise ValueError(
            f"{DATASET_NAME}: labels are missing for split={split}"
        )

    y = np.asarray(dataset.y, dtype=np.int64)

    if not np.isfinite(X).all():
        raise ValueError(
            f"{DATASET_NAME}: X contains NaN or Inf"
        )

    if not np.isfinite(y).all():
        raise ValueError(
            f"{DATASET_NAME}: y contains NaN or Inf"
        )

    return X, y


# ============================================================
# GPU INFORMATION
# ============================================================

def print_gpu_status():
    print("\n" + "=" * 70)
    print("DEVICE INFORMATION")
    print("=" * 70)

    try:
        import torch

        print(f"PyTorch version : {torch.__version__}")
        print(f"CUDA available  : {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(
                f"GPU             : "
                f"{torch.cuda.get_device_name(0)}"
            )
            print(
                f"CUDA version    : "
                f"{torch.version.cuda}"
            )
        else:
            print("GPU             : CPU ONLY")

    except Exception as e:
        print(f"GPU check unavailable: {e}")


# ============================================================
# MODEL FACTORY
# ============================================================

def create_models():
    models = {}

    models["LogisticRegression"] = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    C=1.0,
                    class_weight=None,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    models["XGBoost"] = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        device="cuda",
        random_state=RANDOM_STATE,
    )

    models["MLP"] = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return models


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(name, model, X_train, y_train, X_val, y_val):

    print("\n" + "-" * 70)
    print(f"TRAINING: {name}")
    print("-" * 70)

    start = time.time()

    model.fit(X_train, y_train)

    elapsed = time.time() - start

    val_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, val_pred)
    balanced = balanced_accuracy_score(y_val, val_pred)
    macro_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
    )

    print(f"\nTraining time        : {elapsed:.2f} sec")
    print(f"Validation accuracy  : {accuracy:.4f}")
    print(f"Balanced accuracy    : {balanced:.4f}")
    print(f"Macro F1             : {macro_f1:.4f}")

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(y_val, val_pred))

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_val,
            val_pred,
            digits=4,
        )
    )

    return {
        "name": name,
        "model": model,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
        "time": elapsed,
    }


# ============================================================
# CROSS VALIDATION
# ============================================================

def cross_validate_best_model(model, X, y):

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION")
    print("=" * 70)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("Running 5-fold stratified CV...")

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
    )

    print("\nFold accuracies:")

    for i, score in enumerate(scores, start=1):
        print(f"  Fold {i}: {score:.4f}")

    print(
        f"\nMean CV accuracy: {scores.mean():.4f}"
    )

    print(
        f"CV std          : {scores.std():.4f}"
    )

    return scores


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SERENOVA — MATERNAL HEALTH RISK MODEL BENCHMARK")
    print("=" * 70)

    print_gpu_status()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA")
    print("=" * 70)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    print(f"Train X : {X_train.shape}")
    print(f"Train y : {y_train.shape}")
    print(f"Val X   : {X_val.shape}")
    print(f"Val y   : {y_val.shape}")
    print(f"Test X  : {X_test.shape}")
    print(f"Test y  : {y_test.shape}")

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAIN CLASS DISTRIBUTION")
    print("=" * 70)

    classes, counts = np.unique(
        y_train,
        return_counts=True,
    )

    for cls, count in zip(classes, counts):
        print(f"Class {cls}: {count}")

    # --------------------------------------------------------
    # DATA INTEGRITY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA INTEGRITY")
    print("=" * 70)

    print(
        f"Train finite: "
        f"{np.isfinite(X_train).all()}"
    )

    print(
        f"Val finite  : "
        f"{np.isfinite(X_val).all()}"
    )

    print(
        f"Test finite : "
        f"{np.isfinite(X_test).all()}"
    )

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 1 — MODEL BENCHMARK")
    print("=" * 70)

    models = create_models()

    results = []

    for name, model in models.items():

        try:

            result = evaluate_model(
                name,
                model,
                X_train,
                y_train,
                X_val,
                y_val,
            )

            results.append(result)

        except Exception as e:

            print(
                f"\n{name} FAILED:"
                f" {type(e).__name__}: {e}"
            )

    if not results:
        raise RuntimeError(
            "No models completed successfully."
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 70)

    results.sort(
        key=lambda x: (
            x["macro_f1"],
            x["balanced_accuracy"],
            x["accuracy"],
        ),
        reverse=True,
    )

    for i, result in enumerate(results, start=1):

        print(
            f"{i}. {result['name']:<20}"
            f" Accuracy={result['accuracy']:.4f}"
            f" | Balanced={result['balanced_accuracy']:.4f}"
            f" | Macro F1={result['macro_f1']:.4f}"
        )

    best = results[0]

    print("\n" + "=" * 70)
    print("BEST VALIDATION MODEL")
    print("=" * 70)

    print(f"Model              : {best['name']}")
    print(f"Accuracy            : {best['accuracy']:.4f}")
    print(
        f"Balanced Accuracy  : "
        f"{best['balanced_accuracy']:.4f}"
    )
    print(f"Macro F1            : {best['macro_f1']:.4f}")

    # --------------------------------------------------------
    # CROSS VALIDATION
    # --------------------------------------------------------

    cv_X = np.concatenate(
        [X_train, X_val],
        axis=0,
    )

    cv_y = np.concatenate(
        [y_train, y_val],
        axis=0,
    )

    cv_scores = cross_validate_best_model(
        best["model"],
        cv_X,
        cv_y,
    )

    # --------------------------------------------------------
    # FINAL TRAINING
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 2 — FINAL MODEL FIT")
    print("=" * 70)

    final_X = np.concatenate(
        [X_train, X_val],
        axis=0,
    )

    final_y = np.concatenate(
        [y_train, y_val],
        axis=0,
    )

    print(f"Final training X: {final_X.shape}")
    print(f"Final training y: {final_y.shape}")

    final_model = best["model"]

    start = time.time()

    final_model.fit(
        final_X,
        final_y,
    )

    final_time = time.time() - start

    print(
        f"Training time: {final_time:.2f} sec"
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 3 — FINAL LOCKED TEST")
    print("=" * 70)

    print(
        "IMPORTANT: The test set is used ONLY here."
    )

    test_pred = final_model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        test_pred,
    )

    test_balanced = balanced_accuracy_score(
        y_test,
        test_pred,
    )

    test_macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
    )

    print("\nFINAL TEST")
    print("-" * 70)

    print(
        f"Accuracy          : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{test_balanced:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{test_macro_f1:.4f}"
    )

    print("\nCONFUSION MATRIX")
    print(
        confusion_matrix(
            y_test,
            test_pred,
        )
    )

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            test_pred,
            digits=4,
        )
    )

    # --------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 4 — MODEL SERIALIZATION")
    print("=" * 70)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove stale maternal model artifacts.
    stale_files = [
        RESULTS_DIR / "maternal_health_risk_best.pkl",
        RESULTS_DIR / "maternal_health_risk_xgboost_best.pkl",
        RESULTS_DIR / "maternal_health_risk_final.pkl",
    ]

    for stale in stale_files:

        if stale.exists():
            stale.unlink()

            print(
                f"Deleted stale model: {stale}"
            )

    # --------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------

    if best["name"] == "XGBoost":

        final_model.save_model(
            str(MODEL_PATH)
        )

        print(
            f"\nSaved final XGBoost model:"
        )
        print(
            f"{MODEL_PATH.resolve()}"
        )

        print(
            "Format: native XGBoost JSON"
        )

    else:

        fallback_path = (
            RESULTS_DIR /
            "maternal_health_risk_best.joblib"
        )

        joblib.dump(
            final_model,
            fallback_path,
        )

        print(
            "\nSaved final model:"
        )
        print(
            f"{fallback_path.resolve()}"
        )

        print(
            "Format: Joblib"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"Best Model          : "
        f"{best['name']}"
    )

    print(
        f"Validation Accuracy : "
        f"{best['accuracy']:.4f}"
    )

    print(
        f"CV Accuracy         : "
        f"{cv_scores.mean():.4f}"
    )

    print(
        f"CV Std              : "
        f"{cv_scores.std():.4f}"
    )

    print(
        f"Test Accuracy       : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test Balanced Acc   : "
        f"{test_balanced:.4f}"
    )

    print(
        f"Test Macro F1       : "
        f"{test_macro_f1:.4f}"
    )

    print("\nTEST SET STATUS:")
    print("LOCKED / USED ONLY FOR FINAL EVALUATION")

    print("=" * 70)


if __name__ == "__main__":
    main()