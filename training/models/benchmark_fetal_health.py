from pathlib import Path
import time
import pickle
import numpy as np
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from training.datasets.pytorch_datasets import SerenovaNumpyDataset


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "fetal_health"

CHECKPOINT_DIR = Path("data/checkpoints")
RESULT_DIR = Path("data/results")

MODEL_PATH = RESULT_DIR / "fetal_health_xgboost_best.json"

RANDOM_STATE = 42

N_SPLITS = 5

# Accuracy is important, but we also want to improve
# Class-1 sensitivity and macro F1.
MIN_ACCEPTABLE_ACCURACY = 0.95

# Class-1 weight multipliers to investigate.
#
# Normal class weights are:
#   Class 0 -> 0.4283
#   Class 1 -> 2.3961
#   Class 2 -> 4.0325
#
# We deliberately test moderate additional emphasis
# on Class 1.
CLASS1_MULTIPLIERS = [
    1.00,
    1.10,
    1.20,
    1.30,
    1.40,
]

# XGBoost configurations.
#
# These are deliberately compact rather than performing
# hundreds of experiments on a 1488-sample dataset.
PARAMETER_CONFIGS = [
    {
        "n_estimators": 600,
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 1,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "n_estimators": 800,
        "max_depth": 5,
        "learning_rate": 0.02,
        "min_child_weight": 1,
        "subsample": 0.90,
        "colsample_bytree": 0.95,
        "gamma": 0.0,
        "reg_alpha": 0.01,
        "reg_lambda": 2.0,
    },
    {
        "n_estimators": 700,
        "max_depth": 4,
        "learning_rate": 0.025,
        "min_child_weight": 1,
        "subsample": 0.95,
        "colsample_bytree": 0.95,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.5,
    },
]


# ============================================================
# CLEAN OLD TRAINED FILES
# ============================================================

def clean_old_models():

    print("=" * 70)
    print("CLEANING OLD FETAL HEALTH MODELS")
    print("=" * 70)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    patterns = [
        "fetal_health*",
        "*fetal_health*",
    ]

    deleted = set()

    for directory in [CHECKPOINT_DIR, RESULT_DIR]:

        for pattern in patterns:

            for path in directory.glob(pattern):

                if path.is_file() and path not in deleted:

                    try:
                        path.unlink()
                        deleted.add(path)

                        print(f"Deleted: {path}")

                    except Exception as e:

                        print(
                            f"WARNING: Could not delete "
                            f"{path}: {e}"
                        )

    if not deleted:
        print("No previous fetal-health model files found.")

    print()


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING FETAL HEALTH DATA")
    print("=" * 70)

    train_dataset = SerenovaNumpyDataset(
        dataset_name=DATASET_NAME,
        split="train",
    )

    val_dataset = SerenovaNumpyDataset(
        dataset_name=DATASET_NAME,
        split="val",
    )

    test_dataset = SerenovaNumpyDataset(
        dataset_name=DATASET_NAME,
        split="test",
    )

    X_train = np.asarray(train_dataset.X, dtype=np.float32)
    y_train = np.asarray(train_dataset.y, dtype=np.int64)

    X_val = np.asarray(val_dataset.X, dtype=np.float32)
    y_val = np.asarray(val_dataset.y, dtype=np.int64)

    X_test = np.asarray(test_dataset.X, dtype=np.float32)
    y_test = np.asarray(test_dataset.y, dtype=np.int64)

    print()
    print(f"Train X : {X_train.shape}")
    print(f"Train y : {y_train.shape}")

    print(f"Val X   : {X_val.shape}")
    print(f"Val y   : {y_val.shape}")

    print(f"Test X  : {X_test.shape}")
    print(f"Test y  : {y_test.shape}")

    print()

    print("CLASS DISTRIBUTION")

    for split_name, y in [
        ("TRAIN", y_train),
        ("VALIDATION", y_val),
        ("TEST", y_test),
    ]:

        unique, counts = np.unique(y, return_counts=True)

        print(f"\n{split_name}")

        for cls, count in zip(unique, counts):

            print(
                f"  Class {cls}: {count}"
            )

    print()

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


# ============================================================
# BASE CLASS WEIGHTS
# ============================================================

def get_class_weights(y):

    classes = np.unique(y)

    weights = compute_sample_weight(
        class_weight="balanced",
        y=y,
    )

    class_weight_dict = {}

    for cls in classes:

        mask = y == cls

        class_weight_dict[int(cls)] = float(
            np.mean(weights[mask])
        )

    return class_weight_dict


# ============================================================
# CREATE SAMPLE WEIGHTS
# ============================================================

def create_sample_weights(
    y,
    class_weight_dict,
    class1_multiplier,
):

    weights = np.ones(
        len(y),
        dtype=np.float32,
    )

    for cls, weight in class_weight_dict.items():

        weights[y == cls] = weight

    # Additional emphasis on Class 1.
    weights[y == 1] *= class1_multiplier

    return weights


# ============================================================
# CREATE XGBOOST MODEL
# ============================================================

def create_model(params):

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,

        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        min_child_weight=params["min_child_weight"],

        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],

        gamma=params["gamma"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],

        eval_metric="mlogloss",

        tree_method="hist",
        device="cuda",

        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    return model


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "class1_recall": recall_score(
            y_true,
            y_pred,
            labels=[1],
            average="macro",
            zero_division=0,
        ),
    }


# ============================================================
# CROSS VALIDATION
# ============================================================

def cross_validate_configuration(
    X,
    y,
    params,
    class_weight_dict,
    class1_multiplier,
):

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_accuracy = []
    fold_macro_f1 = []
    fold_class1_recall = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X, y),
        start=1,
    ):

        X_fold_train = X[train_idx]
        y_fold_train = y[train_idx]

        X_fold_val = X[val_idx]
        y_fold_val = y[val_idx]

        sample_weights = create_sample_weights(
            y_fold_train,
            class_weight_dict,
            class1_multiplier,
        )

        model = create_model(params)

        model.fit(
            X_fold_train,
            y_fold_train,
            sample_weight=sample_weights,
            verbose=False,
        )

        predictions = model.predict(
            X_fold_val
        )

        metrics = calculate_metrics(
            y_fold_val,
            predictions,
        )

        fold_accuracy.append(
            metrics["accuracy"]
        )

        fold_macro_f1.append(
            metrics["f1"]
        )

        fold_class1_recall.append(
            metrics["class1_recall"]
        )

        print(
            f"    Fold {fold}: "
            f"Acc={metrics['accuracy']:.4f} | "
            f"Macro-F1={metrics['f1']:.4f} | "
            f"Class-1 Recall={metrics['class1_recall']:.4f}"
        )

    return {
        "cv_accuracy": float(
            np.mean(fold_accuracy)
        ),

        "cv_f1": float(
            np.mean(fold_macro_f1)
        ),

        "cv_class1_recall": float(
            np.mean(fold_class1_recall)
        ),

        "cv_accuracy_std": float(
            np.std(fold_accuracy)
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SERENOVA — FETAL HEALTH CLASS-1 OPTIMIZATION")
    print("=" * 70)

    print()
    print("XGBoost version :", xgb.__version__)
    print("Training device : CUDA GPU")
    print()

    # --------------------------------------------------------
    # DELETE OLD MODELS FIRST
    # --------------------------------------------------------

    clean_old_models()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = load_data()

    # --------------------------------------------------------
    # DATA INTEGRITY
    # --------------------------------------------------------

    print("=" * 70)
    print("DATA INTEGRITY")
    print("=" * 70)

    if not np.isfinite(X_train).all():
        raise ValueError(
            "Training data contains NaN or Inf."
        )

    if not np.isfinite(X_val).all():
        raise ValueError(
            "Validation data contains NaN or Inf."
        )

    if not np.isfinite(X_test).all():
        raise ValueError(
            "Test data contains NaN or Inf."
        )

    print("Train finite : True")
    print("Val finite   : True")
    print("Test finite  : True")
    print()

    # --------------------------------------------------------
    # BASE CLASS WEIGHTS
    # --------------------------------------------------------

    class_weight_dict = get_class_weights(
        y_train
    )

    print("=" * 70)
    print("BASE CLASS WEIGHTS")
    print("=" * 70)

    for cls, weight in class_weight_dict.items():

        print(
            f"Class {cls}: "
            f"{weight:.4f}"
        )

    print()

    # --------------------------------------------------------
    # HYPERPARAMETER + CLASS-1 SEARCH
    # --------------------------------------------------------

    print("=" * 70)
    print("PHASE 1 — CLASS-1 OPTIMIZATION")
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "The test set is NOT used during optimization."
    )
    print()

    results = []

    total_configs = (
        len(PARAMETER_CONFIGS)
        * len(CLASS1_MULTIPLIERS)
    )

    config_number = 0

    for params in PARAMETER_CONFIGS:

        for class1_multiplier in CLASS1_MULTIPLIERS:

            config_number += 1

            print("-" * 70)
            print(
                f"CONFIGURATION "
                f"{config_number}/{total_configs}"
            )
            print("-" * 70)

            print("Parameters:")

            for key, value in params.items():

                print(
                    f"  {key}: {value}"
                )

            print(
                f"  class1_multiplier: "
                f"{class1_multiplier}"
            )

            start_time = time.time()

            # ------------------------------------------------
            # CV
            # ------------------------------------------------

            cv_metrics = cross_validate_configuration(
                X_train,
                y_train,
                params,
                class_weight_dict,
                class1_multiplier,
            )

            # ------------------------------------------------
            # VALIDATION MODEL
            # ------------------------------------------------

            sample_weights = create_sample_weights(
                y_train,
                class_weight_dict,
                class1_multiplier,
            )

            model = create_model(params)

            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
                verbose=False,
            )

            val_predictions = model.predict(
                X_val
            )

            val_metrics = calculate_metrics(
                y_val,
                val_predictions,
            )

            elapsed = time.time() - start_time

            print()
            print(
                f"CV Accuracy      : "
                f"{cv_metrics['cv_accuracy']:.4f}"
            )

            print(
                f"CV Macro F1      : "
                f"{cv_metrics['cv_f1']:.4f}"
            )

            print(
                f"CV Class-1 Recall: "
                f"{cv_metrics['cv_class1_recall']:.4f}"
            )

            print(
                f"CV Accuracy Std  : "
                f"{cv_metrics['cv_accuracy_std']:.4f}"
            )

            print(
                f"Validation Acc   : "
                f"{val_metrics['accuracy']:.4f}"
            )

            print(
                f"Validation F1    : "
                f"{val_metrics['f1']:.4f}"
            )

            print(
                f"Validation Recall: "
                f"{val_metrics['recall']:.4f}"
            )

            print(
                f"Class-1 Recall   : "
                f"{val_metrics['class1_recall']:.4f}"
            )

            print(
                f"Time             : "
                f"{elapsed:.2f}s"
            )

            results.append(
                {
                    "params": params.copy(),
                    "class1_multiplier": class1_multiplier,
                    "cv_accuracy": cv_metrics["cv_accuracy"],
                    "cv_f1": cv_metrics["cv_f1"],
                    "cv_class1_recall": cv_metrics[
                        "cv_class1_recall"
                    ],
                    "cv_accuracy_std": cv_metrics[
                        "cv_accuracy_std"
                    ],
                    "val_accuracy": val_metrics[
                        "accuracy"
                    ],
                    "val_f1": val_metrics["f1"],
                    "val_recall": val_metrics[
                        "recall"
                    ],
                    "val_class1_recall": val_metrics[
                        "class1_recall"
                    ],
                }
            )

    # --------------------------------------------------------
    # SELECT BEST CONFIGURATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 2 — SELECT BEST CONFIGURATION")
    print("=" * 70)

    # First preference:
    # validation accuracy >= 95%
    #
    # Among those, maximize Class-1 recall,
    # then Macro F1, then accuracy.

    acceptable = [
        r
        for r in results
        if r["val_accuracy"]
        >= MIN_ACCEPTABLE_ACCURACY
    ]

    if acceptable:

        best = max(
            acceptable,
            key=lambda r: (
                r["val_class1_recall"],
                r["val_f1"],
                r["val_accuracy"],
                r["cv_accuracy"],
            ),
        )

        selection_rule = (
            "Validation accuracy >= 95%, "
            "then Class-1 recall, Macro F1, accuracy"
        )

    else:

        # If no candidate maintains 95% validation
        # accuracy, do not fake the target.
        #
        # Select based on Macro F1 + Class-1 recall.

        best = max(
            results,
            key=lambda r: (
                r["val_f1"],
                r["val_class1_recall"],
                r["val_accuracy"],
            ),
        )

        selection_rule = (
            "No configuration reached 95% validation accuracy; "
            "selected by Macro F1 + Class-1 recall"
        )

    print()
    print(
        f"Selection rule: {selection_rule}"
    )

    print()
    print("BEST CONFIGURATION")

    print(
        f"Validation Accuracy : "
        f"{best['val_accuracy']:.4f}"
    )

    print(
        f"Validation Macro F1 : "
        f"{best['val_f1']:.4f}"
    )

    print(
        f"Validation Class-1 Recall : "
        f"{best['val_class1_recall']:.4f}"
    )

    print(
        f"CV Accuracy : "
        f"{best['cv_accuracy']:.4f}"
    )

    print(
        f"CV Macro F1 : "
        f"{best['cv_f1']:.4f}"
    )

    print()
    print("PARAMETERS")

    for key, value in best["params"].items():

        print(
            f"  {key}: {value}"
        )

    print(
        f"  class1_multiplier: "
        f"{best['class1_multiplier']}"
    )

    # --------------------------------------------------------
    # FINAL TRAINING
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 3 — FINAL MODEL TRAINING")
    print("=" * 70)

    print()
    print(
        "Combining TRAIN + VALIDATION."
    )

    print(
        "TEST remains completely untouched."
    )

    X_final = np.concatenate(
        [
            X_train,
            X_val,
        ],
        axis=0,
    )

    y_final = np.concatenate(
        [
            y_train,
            y_val,
        ],
        axis=0,
    )

    print()
    print(
        f"Final X : {X_final.shape}"
    )

    print(
        f"Final y : {y_final.shape}"
    )

    final_class_weights = get_class_weights(
        y_final
    )

    final_sample_weights = create_sample_weights(
        y_final,
        final_class_weights,
        best["class1_multiplier"],
    )

    final_model = create_model(
        best["params"]
    )

    start_time = time.time()

    final_model.fit(
        X_final,
        y_final,
        sample_weight=final_sample_weights,
        verbose=False,
    )

    training_time = time.time() - start_time

    print(
        f"Training time: "
        f"{training_time:.2f}s"
    )

    print(
        f"Model device: "
        f"cuda:0"
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PHASE 4 — FINAL LOCKED TEST")
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This is the FIRST use of the locked test set."
    )

    print(
        "No parameters are changed after this."
    )

    test_predictions = final_model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    print()
    print("FINAL TEST — XGBOOST")
    print("-" * 70)

    print(
        f"Accuracy          : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro Precision   : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Macro Recall      : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Class-1 Recall    : "
        f"{metrics['class1_recall']:.4f}"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        test_predictions,
    )

    print()
    print("CONFUSION MATRIX")
    print("-" * 70)
    print(cm)

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print()
    print("CLASSIFICATION REPORT")
    print("-" * 70)

    print(
        classification_report(
            y_test,
            test_predictions,
            digits=4,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # SAVE ONLY FINAL MODEL
    # --------------------------------------------------------

    print("=" * 70)
    print("PHASE 5 — MODEL SERIALIZATION")
    print("=" * 70)

    # Make absolutely sure an old model cannot survive.
    if MODEL_PATH.exists():

        MODEL_PATH.unlink()

        print(
            f"Removed previous model: "
            f"{MODEL_PATH}"
        )

    final_model.save_model(
        str(MODEL_PATH)
    )

    print()
    print(
        f"Saved final model:"
    )

    print(
        MODEL_PATH.resolve()
    )

    print()
    print(
        "Format: native XGBoost JSON"
    )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"Validation Accuracy : "
        f"{best['val_accuracy']:.4f}"
    )

    print(
        f"CV Accuracy         : "
        f"{best['cv_accuracy']:.4f}"
    )

    print(
        f"CV Macro F1         : "
        f"{best['cv_f1']:.4f}"
    )

    print(
        f"Test Accuracy       : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Test Macro F1       : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Test Class-1 Recall : "
        f"{metrics['class1_recall']:.4f}"
    )

    print()

    if metrics["accuracy"] >= 0.95:

        print(
            "ACCURACY TARGET: PASS"
        )

    else:

        print(
            "ACCURACY TARGET: NOT MET"
        )

    if metrics["class1_recall"] > 0.7727:

        print(
            "CLASS-1 RECALL: IMPROVED"
        )

    else:

        print(
            "CLASS-1 RECALL: NOT IMPROVED"
        )

    print()
    print(
        "TEST SET STATUS: LOCKED / USED ONLY FOR FINAL EVALUATION"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()