from pathlib import Path
import json
import numpy as np
# pyrefly: ignore [missing-import]
import wfdb

from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = (
    ROOT / "data" / "raw" / "external" / "bidmc" /
    "bidmc-ppg-and-respiration-dataset-1.0.0"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "bidmc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FS = 125
WINDOW_SECONDS = 10
WINDOW_SIZE = FS * WINDOW_SECONDS


def bandpass(signal, lowcut, highcut, fs, order=4):

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype="band"
    )

    return filtfilt(b, a, signal)


def normalize(signal):

    mean = np.mean(signal)
    std = np.std(signal)

    if std < 1e-8:
        return signal - mean

    return (signal - mean) / std


def find_records():

    records = []

    for hea in DATASET_DIR.glob("*.hea"):

        name = hea.stem

        if not name.endswith("n"):
            records.append(name)

    return sorted(set(records))


def process_record(record_name):

    record_path = str(
        DATASET_DIR / record_name
    )

    record = wfdb.rdrecord(record_path)

    signal_names = [
    name.strip().strip(",").upper()
    for name in record.sig_name
]

    if "PLETH" not in signal_names:
        raise ValueError(
            f"PLETH not found in {record_name}"
        )

    if "RESP" not in signal_names:
        raise ValueError(
            f"RESP not found in {record_name}"
        )

    ppg_index = signal_names.index("PLETH")
    resp_index = signal_names.index("RESP")

    ppg = record.p_signal[:, ppg_index]
    resp = record.p_signal[:, resp_index]

    # PPG physiological band.
    ppg = bandpass(
        ppg,
        lowcut=0.5,
        highcut=5.0,
        fs=FS
    )

    # Respiration band.
    resp = bandpass(
        resp,
        lowcut=0.1,
        highcut=0.7,
        fs=FS
    )

    ppg = normalize(ppg)
    resp = normalize(resp)

    windows = []

    number_of_windows = (
        len(ppg) // WINDOW_SIZE
    )

    for i in range(number_of_windows):

        start = i * WINDOW_SIZE
        end = start + WINDOW_SIZE

        ppg_window = ppg[start:end]
        resp_window = resp[start:end]

        if len(ppg_window) != WINDOW_SIZE:
            continue

        if len(resp_window) != WINDOW_SIZE:
            continue

        if not (
            np.isfinite(ppg_window).all()
            and np.isfinite(resp_window).all()
        ):
            continue

        # Shape:
        # [2, WINDOW_SIZE]
        window = np.stack(
            [ppg_window, resp_window],
            axis=0
        )

        windows.append(window)

    return windows


def main():

    print("\n" + "=" * 60)
    print("SERENOVA — BIDMC PREPROCESSING")
    print("=" * 60)

    records = find_records()

    print("Records:", len(records))
    print("Sampling rate:", FS, "Hz")
    print("Window:", WINDOW_SECONDS, "seconds")
    print("Samples/window:", WINDOW_SIZE)

    all_windows = []
    record_ids = []

    for index, record_name in enumerate(records):

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

    record_ids = np.asarray(
        record_ids
    )

    print("\nTotal windows:", len(X))
    print("Final shape:", X.shape)

    if len(X) == 0:
        raise RuntimeError(
            "No valid BIDMC windows generated."
        )

    # Split by RECORD, not individual windows.
    # This prevents windows from the same patient
    # appearing in both train and test sets.

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
        "dataset": "BIDMC PPG and Respiration",
        "sampling_frequency_hz": FS,
        "window_seconds": WINDOW_SECONDS,
        "window_samples": WINDOW_SIZE,
        "channels": [
            "PLETH_PPG",
            "RESP"
        ],
        "channel_order": [
            0,
            1
        ],
        "filtering": {
            "PPG": "0.5-5.0 Hz bandpass",
            "RESP": "0.1-0.7 Hz bandpass"
        },
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