from pathlib import Path
import json
import numpy as np
import pyedflib
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = (
    ROOT
    / "data"
    / "raw"
    / "external"
    / "fetal_ecg"
    / "nifeadb"
    / "non-invasive-fetal-ecg-database-1.0.0"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "fetal_ecg"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 1000
WINDOW_SECONDS = 10
WINDOW_SIZE = FS * WINDOW_SECONDS

CHANNELS = [
    "Abdomen_1",
    "Abdomen_2",
    "Abdomen_3",
]


def normalize(signal):

    signal = np.asarray(signal, dtype=np.float32)

    finite = np.isfinite(signal)

    if not finite.any():
        return None

    median = np.median(signal[finite])
    signal[~finite] = median

    mean = np.mean(signal)
    std = np.std(signal)

    if std < 1e-8:
        return signal - mean

    return (signal - mean) / std


def process_record(edf_file):

    reader = pyedflib.EdfReader(str(edf_file))

    try:

        labels = reader.getSignalLabels()

        channel_indices = []

        for channel in CHANNELS:

            if channel not in labels:
                raise ValueError(
                    f"{channel} not found in {edf_file.name}"
                )

            channel_indices.append(
                labels.index(channel)
            )

        signals = []

        for index in channel_indices:

            signal = reader.readSignal(index)

            signal = normalize(signal)

            if signal is None:
                return []

            signals.append(signal)

        min_length = min(
            len(signal)
            for signal in signals
        )

        number_of_windows = (
            min_length // WINDOW_SIZE
        )

        windows = []

        for i in range(number_of_windows):

            start = i * WINDOW_SIZE
            end = start + WINDOW_SIZE

            window_channels = []

            for signal in signals:

                segment = signal[start:end]

                if len(segment) != WINDOW_SIZE:
                    continue

                window_channels.append(segment)

            if len(window_channels) != len(CHANNELS):
                continue

            window = np.stack(
                window_channels,
                axis=0
            )

            if np.isfinite(window).all():
                windows.append(window)

        return windows

    finally:
        reader.close()


def main():

    print("\n" + "=" * 60)
    print("SERENOVA — FETAL ECG PREPROCESSING")
    print("=" * 60)

    files = sorted(
        DATASET_DIR.glob("*.edf")
    )

    print("Records:", len(files))
    print("Sampling rate:", FS, "Hz")
    print("Window:", WINDOW_SECONDS, "seconds")
    print("Samples/window:", WINDOW_SIZE)
    print("Channels:", CHANNELS)

    all_windows = []
    record_ids = []

    for edf_file in files:

        windows = process_record(edf_file)

        print(
            f"{edf_file.stem}: "
            f"{len(windows)} windows"
        )

        all_windows.extend(windows)

        record_ids.extend(
            [edf_file.stem] * len(windows)
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
            "No valid fetal ECG windows generated."
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
        "dataset": "Non-invasive Fetal ECG Database",
        "sampling_frequency_hz": FS,
        "window_seconds": WINDOW_SECONDS,
        "window_samples": WINDOW_SIZE,
        "channels": CHANNELS,
        "normalization": "per-channel z-score",
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