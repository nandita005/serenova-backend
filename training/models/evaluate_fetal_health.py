from pathlib import Path

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
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# CONFIG
# ============================================================

DATASET = Path("data/processed/fetal_health")

# FINAL native XGBoost model
MODEL_PATH = Path(
    "data/results/fetal_health_xgboost_best.json"
)


# ============================================================
# LOAD DATA
# ============================================================

X_test = np.load(
    DATASET / "X_test.npy"
)

y_test = np.load(
    DATASET / "y_test.npy"
)


print("=" * 70)
print("SERENOVA — FETAL HEALTH PRODUCTION EVALUATION")
print("=" * 70)


# ============================================================
# TEST DATA
# ============================================================

print()
print("TEST DATA")
print("-" * 70)

print("X shape :", X_test.shape)
print("y shape :", y_test.shape)
print("Samples :", len(y_test))

print()
print("IMPORTANT:")
print("This is the locked test set.")
print("No training or parameter tuning is performed here.")


# ============================================================
# DATA VALIDATION
# ============================================================

print()
print("=" * 70)
print("DATA INTEGRITY")
print("=" * 70)

print(
    "Finite X:",
    np.isfinite(X_test).all()
)

print(
    "Finite y:",
    np.isfinite(y_test).all()
)

assert np.isfinite(X_test).all()
assert np.isfinite(y_test).all()


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 70)
print("MODEL")
print("=" * 70)

print("Loading:", MODEL_PATH)

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nFinal model not found:\n{MODEL_PATH}\n\n"
        "Expected model:\n"
        "data/results/fetal_health_xgboost_best.json"
    )


model = xgb.XGBClassifier()

model.load_model(
    str(MODEL_PATH)
)

print("Model loaded successfully.")


# ============================================================
# PREDICTION
# ============================================================

print()
print("=" * 70)
print("PREDICTION")
print("=" * 70)

y_pred = model.predict(
    X_test
)

y_prob = model.predict_proba(
    X_test
)

print(
    "Prediction shape :",
    y_pred.shape
)

print(
    "Probability shape:",
    y_prob.shape
)


# ============================================================
# BASIC METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred,
)

balanced_acc = balanced_accuracy_score(
    y_test,
    y_pred,
)

precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)


print()
print("=" * 70)
print("OVERALL METRICS")
print("=" * 70)

print(
    f"Accuracy          : {accuracy:.4f}"
)

print(
    f"Balanced Accuracy : {balanced_acc:.4f}"
)

print(
    f"Macro Precision   : {precision:.4f}"
)

print(
    f"Macro Recall      : {recall:.4f}"
)

print(
    f"Macro F1          : {f1:.4f}"
)


# ============================================================
# PER-CLASS METRICS
# ============================================================

print()
print("=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)

report = classification_report(
    y_test,
    y_pred,
    digits=4,
    zero_division=0,
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
)

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# CLASS-WISE ERROR ANALYSIS
# ============================================================

print()
print("=" * 70)
print("CLASS-WISE ERROR ANALYSIS")
print("=" * 70)

classes = np.unique(
    y_test
)

for cls in classes:

    actual = (
        y_test == cls
    )

    predicted = (
        y_pred == cls
    )

    tp = np.sum(
        actual & predicted
    )

    fn = np.sum(
        actual & ~predicted
    )

    fp = np.sum(
        ~actual & predicted
    )

    tn = np.sum(
        ~actual & ~predicted
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    print()
    print(f"CLASS {cls}")

    print(
        f"  TP          : {tp}"
    )

    print(
        f"  FN          : {fn}"
    )

    print(
        f"  FP          : {fp}"
    )

    print(
        f"  TN          : {tn}"
    )

    print(
        f"  Sensitivity : {sensitivity:.4f}"
    )

    print(
        f"  Specificity : {specificity:.4f}"
    )


# ============================================================
# ROC-AUC
# ============================================================

print()
print("=" * 70)
print("ROC-AUC")
print("=" * 70)

try:

    roc_auc = roc_auc_score(
        y_test,
        y_prob,
        multi_class="ovr",
        average="macro",
    )

    print(
        f"Macro ROC-AUC : {roc_auc:.4f}"
    )

except Exception as e:

    print(
        "ROC-AUC could not be calculated:"
    )

    print(e)


# ============================================================
# PR-AUC
# ============================================================

print()
print("=" * 70)
print("PRECISION-RECALL AUC")
print("=" * 70)

try:

    y_onehot = np.eye(
        len(classes)
    )[y_test]

    pr_auc = average_precision_score(
        y_onehot,
        y_prob,
        average="macro",
    )

    print(
        f"Macro PR-AUC : {pr_auc:.4f}"
    )

except Exception as e:

    print(
        "PR-AUC could not be calculated:"
    )

    print(e)


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

print()
print("=" * 70)
print("PREDICTION CONFIDENCE")
print("=" * 70)

confidence = np.max(
    y_prob,
    axis=1,
)

print(
    f"Mean confidence : "
    f"{confidence.mean():.4f}"
)

print(
    f"Min confidence  : "
    f"{confidence.min():.4f}"
)

print(
    f"Max confidence  : "
    f"{confidence.max():.4f}"
)

for threshold in [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]:

    mask = (
        confidence >= threshold
    )

    if np.sum(mask) == 0:
        continue

    acc = accuracy_score(
        y_test[mask],
        y_pred[mask],
    )

    coverage = np.mean(
        mask
    )

    print(
        f"Confidence >= {threshold:.2f} | "
        f"Accuracy={acc:.4f} | "
        f"Coverage={coverage:.4f}"
    )


# ============================================================
# MISCLASSIFICATION ANALYSIS
# ============================================================

print()
print("=" * 70)
print("MISCLASSIFICATIONS")
print("=" * 70)

wrong = np.where(
    y_test != y_pred
)[0]

print(
    f"Incorrect predictions: "
    f"{len(wrong)}/{len(y_test)}"
)

print(
    f"Error rate: "
    f"{len(wrong) / len(y_test):.4f}"
)

print()
print("Actual -> Predicted")

for i in wrong:

    print(
        f"Sample {i:3d}: "
        f"{y_test[i]} -> {y_pred[i]} "
        f"| confidence={confidence[i]:.4f}"
    )


# ============================================================
# FINAL VERDICT
# ============================================================

print()
print("=" * 70)
print("PRODUCTION EVALUATION SUMMARY")
print("=" * 70)

print(
    f"Test Accuracy          : {accuracy:.4f}"
)

print(
    f"Balanced Accuracy      : {balanced_acc:.4f}"
)

print(
    f"Macro Precision        : {precision:.4f}"
)

print(
    f"Macro Recall           : {recall:.4f}"
)

print(
    f"Macro F1               : {f1:.4f}"
)

print()

if accuracy >= 0.95:

    print(
        "Accuracy target        : PASS"
    )

else:

    print(
        "Accuracy target        : FAIL"
    )


if f1 >= 0.90:

    print(
        "Macro F1 target        : PASS"
    )

else:

    print(
        "Macro F1 target        : REVIEW"
    )


print()
print("MODEL STATUS:")
print("Evaluation complete.")
print("Test set remains locked.")
print("=" * 70)