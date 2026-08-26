from pathlib import Path

import numpy as np
import wfdb
from sklearn.model_selection import train_test_split


ROOT = Path(
    r"D:\serenova_backend\data\raw\external\mit_bih_nsr"
)

DATASET_DIR = (
    ROOT
    / "mit-bih-normal-sinus-rhythm-database-1.0.0"
)

OUTPUT_DIR = Path(
    r"D:\serenova_backend\data\processed\mit_bih_nsr"
)

FS = 128
WINDOW_SECONDS = 10
WINDOW_SAMPLES = FS * WINDOW_SECONDS

RANDOM_STATE = 42


def find_records():
    records = []

    for hea_file in DATASET_DIR.glob("*.hea"):
        name = hea_file.stem

        if name.endswith("-"):
            continue

        records.append(name)

    return sorted(records)


def normalize_window(window):
    window = window.astype(np.float32)

    mean = np.mean(window, axis=1, keepdims=True)
    std = np.std(window, axis=1, keepdims=True)

    std = np.maximum(std, 1e-8)

    window = (window - mean) / std

    return window


def process_record(record_name):
    record_path = DATASET_DIR / record_name

    record = wfdb.rdrecord(
        str(record_path),
        channels=[0, 1]
    )

    signal = record.p_signal.T

    total_samples = signal.shape[1]

    windows = []

    for start in range(
        0,
        total_samples - WINDOW_SAMPLES + 1,
        WINDOW_SAMPLES
    ):
        end = start + WINDOW_SAMPLES

        window = signal[:, start:end]

        if window.shape != (2, WINDOW_SAMPLES):
            continue

        if not np.isfinite(window).all():
            continue

        window = normalize_window(window)

        windows.append(window)

    return windows


def save_split(name, X):
    np.save(
        OUTPUT_DIR / f"X_{name}.npy",
        X.astype(np.float32)
    )


def main():

    print("=" * 60)
    print("SERENOVA — MIT-BIH NSR PREPROCESSING")
    print("=" * 60)

    records = find_records()

    print(f"Records: {len(records)}")
    print(f"Sampling rate: {FS} Hz")
    print(f"Window: {WINDOW_SECONDS} seconds")
    print(f"Samples/window: {WINDOW_SAMPLES}")
    print("Channels: ECG1, ECG2")

    all_windows = []

    for record_name in records:

        windows = process_record(record_name)

        print(
            f"{record_name}: "
            f"{len(windows)} windows"
        )

        all_windows.extend(windows)

    if not all_windows:
        raise RuntimeError(
            "No valid windows were generated."
        )

    X = np.stack(all_windows).astype(np.float32)

    print("\nTotal windows:", len(X))
    print("Final shape:", X.shape)

    # First split: 70% train, 30% temporary
    X_train, X_temp = train_test_split(
        X,
        test_size=0.30,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # Second split: 15% validation, 15% test
    X_val, X_test = train_test_split(
        X_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    save_split("train", X_train)
    save_split("val", X_val)
    save_split("test", X_test)

    print("\n" + "-" * 60)
    print("PREPROCESSING COMPLETE")
    print("-" * 60)

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    print("\nSaved to:")
    print(OUTPUT_DIR)

    print("=" * 60)


if __name__ == "__main__":
    main()