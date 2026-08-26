from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

RAW_FILE = (
    ROOT
    / "data"
    / "raw"
    / "external"
    / "diabetes"
    / "diabetes.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "diabetes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SCHEMA
# ============================================================

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET_COLUMN = "Outcome"


# ============================================================
# LOAD
# ============================================================

def load_dataset():

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{RAW_FILE}"
        )

    df = pd.read_csv(RAW_FILE)

    print(f"Loaded dataset: {df.shape}")

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df[required_columns].copy()


# ============================================================
# CLEAN
# ============================================================

def clean_dataset(df):

    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    before = len(df)

    df.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN],
        inplace=True
    )

    removed = before - len(df)

    print(f"Rows removed during cleaning: {removed}")
    print(f"Clean dataset: {df.shape}")

    return df


# ============================================================
# PREPARE
# ============================================================

def prepare_data(df):

    X = df[FEATURE_COLUMNS].copy()

    y = df[TARGET_COLUMN].astype(int)

    # Expected:
    # 0 = No diabetes
    # 1 = Diabetes

    invalid_labels = set(y.unique()) - {0, 1}

    if invalid_labels:
        raise ValueError(
            f"Unexpected Outcome values: {invalid_labels}"
        )

    return X, y


# ============================================================
# SPLIT
# ============================================================

def split_data(X, y):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(X_train, X_val, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler,
    )


# ============================================================
# SAVE
# ============================================================

def save_data(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    scaler,
):

    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "X_val.npy", X_val)
    np.save(OUTPUT_DIR / "X_test.npy", X_test)

    np.save(
        OUTPUT_DIR / "y_train.npy",
        y_train.to_numpy()
    )

    np.save(
        OUTPUT_DIR / "y_val.npy",
        y_val.to_numpy()
    )

    np.save(
        OUTPUT_DIR / "y_test.npy",
        y_test.to_numpy()
    )

    joblib.dump(
        scaler,
        OUTPUT_DIR / "scaler.joblib"
    )

    metadata = {
        "dataset": "Diabetes",
        "source_file": str(RAW_FILE),
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "classes": {
            "0": "No diabetes",
            "1": "Diabetes",
        },
        "split": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
        "random_state": 42,
        "preprocessing": "StandardScaler",
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("SERENOVA — DIABETES PREPROCESSING")
    print("=" * 60)

    df = load_dataset()

    df = clean_dataset(df)

    X, y = prepare_data(df)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(X, y)

    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler,
    ) = normalize(
        X_train,
        X_val,
        X_test,
    )

    save_data(
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        scaler,
    )

    print("\nPREPROCESSING COMPLETE")
    print("-" * 60)

    print("Train:", X_train_scaled.shape)
    print("Validation:", X_val_scaled.shape)
    print("Test:", X_test_scaled.shape)

    print("\nClass distribution:")
    print("Train:", np.bincount(y_train))
    print("Validation:", np.bincount(y_val))
    print("Test:", np.bincount(y_test))

    print("\nSaved to:")
    print(OUTPUT_DIR)

    print("=" * 60)


if __name__ == "__main__":
    main()