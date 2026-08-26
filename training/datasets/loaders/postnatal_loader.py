from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[3]

RAW_FILE = (
    ROOT / "data" / "raw" / "external" /
    "postnatal" / "post natal data.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "postnatal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_COLUMNS = [
    "Age",
    "Irritable towards baby & partner",
    "Trouble sleeping at night",
    "Problems concentrating or making decision",
    "Overeating or loss of appetite",
    "Feeling anxious",
    "Feeling of guilt",
    "Problems of bonding with baby",
    "Suicide attempt",
]

TARGET_COLUMN = "Feeling sad or Tearful"


def load_dataset():

    df = pd.read_csv(RAW_FILE)

    print(f"Loaded dataset: {df.shape}")

    required = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df[required].copy()


def clean_dataset(df):

    before = len(df)

    # All columns in this dataset are categorical strings.
    # Do NOT convert Age to numeric.
    for column in df.columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    # Only remove genuine missing values.
    df = df.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    ).copy()

    removed = before - len(df)

    print(f"Rows removed during cleaning: {removed}")
    print(f"Clean dataset: {df.shape}")

    return df


def encode_dataset(df):

    X = pd.DataFrame(index=df.index)

    encoders = {}

    for column in FEATURE_COLUMNS:

        encoder = LabelEncoder()

        X[column] = encoder.fit_transform(
            df[column]
        )

        encoders[column] = {
            str(label): int(i)
            for i, label
            in enumerate(encoder.classes_)
        }

    target_encoder = LabelEncoder()

    y = target_encoder.fit_transform(
        df[TARGET_COLUMN]
    )

    target_mapping = {
        str(label): int(i)
        for i, label
        in enumerate(target_encoder.classes_)
    }

    return (
        X,
        pd.Series(y, index=df.index),
        encoders,
        target_mapping,
    )


def split_dataset(X, y):

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


def normalize(X_train, X_val, X_test):

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, scaler


def save_data(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    scaler,
    encoders,
    target_mapping,
):

    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "X_val.npy", X_val)
    np.save(OUTPUT_DIR / "X_test.npy", X_test)

    np.save(OUTPUT_DIR / "y_train.npy", y_train.to_numpy())
    np.save(OUTPUT_DIR / "y_val.npy", y_val.to_numpy())
    np.save(OUTPUT_DIR / "y_test.npy", y_test.to_numpy())

    joblib.dump(
        scaler,
        OUTPUT_DIR / "scaler.joblib"
    )

    metadata = {
        "dataset": "Postnatal",
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "feature_encoders": encoders,
        "target_mapping": target_mapping,
        "preprocessing": "LabelEncoder + StandardScaler",
        "split": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
        "random_state": 42,
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )


def main():

    print("\n" + "=" * 60)
    print("SERENOVA — POSTNATAL PREPROCESSING")
    print("=" * 60)

    df = load_dataset()

    df = clean_dataset(df)

    X, y, encoders, target_mapping = encode_dataset(df)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(X, y)

    (
        X_train,
        X_val,
        X_test,
        scaler,
    ) = normalize(
        X_train,
        X_val,
        X_test
    )

    save_data(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
        encoders,
        target_mapping,
    )

    print("\nPREPROCESSING COMPLETE")
    print("-" * 60)

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    print("\nTarget mapping:")
    print(target_mapping)

    print("\nClass distribution:")
    print("Train:", np.bincount(y_train))
    print("Validation:", np.bincount(y_val))
    print("Test:", np.bincount(y_test))

    print("\nSaved to:")
    print(OUTPUT_DIR)

    print("=" * 60)


if __name__ == "__main__":
    main()