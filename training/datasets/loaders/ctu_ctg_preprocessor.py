from pathlib import Path
import json
import numpy as np
import wfdb
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = (
    ROOT / "data" / "raw" / "external" / "ctu_uhb_ctg" /
    "ctu-chb-intrapartum-cardiotocography-database-1.0.0"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "ctu_ctg"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 4
WINDOW_SECONDS = 300
WINDOW_SIZE = FS * WINDOW_SECONDS


def clean_signal(signal):
    signal = np.asarray(signal, dtype=np.float32)

    finite = np.isfinite(signal)

    if not finite.any():
        return None

    valid = signal[finite]

    median = np.median(valid)

    signal[~finite] = median

    # Remove obviously invalid CTG values.
    signal = np.clip(signal, 50.0, 220.0)

    return signal


def normalize(signal):

    mean = np.mean(signal)
    std = np.std(signal)

    if std < 1e-8:
        return signal - mean

    return (signal - mean) / std


def find_records():

    return sorted(
        set(
            hea.stem
            for hea in DATASET_DIR.glob("*.hea")
        )
    )


def process_record(record_name):

    record = wfdb.rdrecord(
        str(DATASET_DIR / record_name)
    )

    names = [
        name.strip().upper()
        for name in record.sig_name
    ]

    if "FHR" not in names:
        raise ValueError(
            f"FHR not found in {record_name}"
        )

    if "UC" not in names:
        raise ValueError(
            f"UC not found in {record_name}"
        )

    fhr_index = names.index("FHR")
    uc_index = names.index("UC")

    fhr = clean_signal(
        record.p_signal[:, fhr_index]
    )

    uc = clean_signal(
        record.p_signal[:, uc_index]
    )

    if fhr is None or uc is None:
        return []

    windows = []

    number_of_windows = (
        min(len(fhr), len(uc)) // WINDOW_SIZE
    )

    for i in range(number_of_windows):

        start = i * WINDOW_SIZE
        end = start + WINDOW_SIZE

        fhr_window = fhr[start:end]
        uc_window = uc[start:end]

        if len(fhr_window) != WINDOW_SIZE:
            continue

        if len(uc_window) != WINDOW_SIZE:
            continue

        # Normalize each channel independently.
        fhr_window = normalize(fhr_window)
        uc_window = normalize(uc_window)

        window = np.stack(
            [fhr_window, uc_window],
            axis=0
        )

        if not np.isfinite(window).all():
            continue

        windows.append(window)

    return windows


def main():

    print("\n" + "=" * 60)
    print("SERENOVA — CTU-UHB CTG PREPROCESSING")
    print("=" * 60)

    records = find_records()

    print("Records:", len(records))
    print("Sampling rate:", FS, "Hz")
    print("Window:", WINDOW_SECONDS, "seconds")
    print("Samples/window:", WINDOW_SIZE)

    all_windows = []
    record_ids = []

    for record_name in records:

        windows = process_record(record_name)

        print(
            f"{record_name}: "
            f"{len(windows)} windows"
        )

        all_windows.extend(windows)

        record_ids.extend(
            [record_name] * len(windows)
        )

    X = np.asarray(
        all_windows,
        dtype=np.float32
    )

    record_ids = np.asarray(record_ids)

    print("\nTotal windows:", len(X))
    print("Final shape:", X.shape)

    if len(X) == 0:
        raise RuntimeError(
            "No valid CTG windows generated."
        )

    unique_records = np.unique(record_ids)

    train_records, temp_records = train_test_split(
        unique_records,
        test_size=0.30,
        random_state=42
    )

    val_records, test_records = train_test_split(
        temp_records,
        test_size=0.50,
        random_state=42
    )

    train_mask = np.isin(
        record_ids,
        train_records
    )

    val_mask = np.isin(
        record_ids,
        val_records
    )

    test_mask = np.isin(
        record_ids,
        test_records
    )

    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]

    np.save(
        OUTPUT_DIR / "X_train.npy",
        X_train
    )

    np.save(
        OUTPUT_DIR / "X_val.npy",
        X_val
    )

    np.save(
        OUTPUT_DIR / "X_test.npy",
        X_test
    )

    metadata = {
        "dataset": "CTU-UHB Intrapartum CTG",
        "sampling_frequency_hz": FS,
        "window_seconds": WINDOW_SECONDS,
        "window_samples": WINDOW_SIZE,
        "channels": [
            "FHR",
            "UC"
        ],
        "normalization": "per-window z-score",
        "split_strategy": "record-level",
        "train_records": train_records.tolist(),
        "validation_records": val_records.tolist(),
        "test_records": test_records.tolist(),
        "shapes": {
            "train": list(X_train.shape),
            "validation": list(X_val.shape),
            "test": list(X_test.shape)
        }
    }

    with open(
        OUTPUT_DIR / "preprocessing_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

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