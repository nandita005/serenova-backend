"""
SERENOVA — Shiraz Fetal Heart Sound Preprocessor

Dataset:
    Shiraz Fetal Heart Sound (FHS)

Input:
    WAV records containing:
        PCG
        PCG1

Processing:
    - Handles mixed sampling rates
    - Resamples all records to 16 kHz
    - 5-second windows
    - 80,000 samples/window
    - Per-channel normalization
    - Removes non-finite values
    - Saves train/validation/test NumPy arrays

Output:
    data/processed/shiraz_fhs/
        X_train.npy
        X_val.npy
        X_test.npy
        metadata.json
"""

from pathlib import Path
import json

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "external"
    / "shiraz_fetal_heart_sound"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "shiraz_fhs"
)

TARGET_FS = 16000

WINDOW_SECONDS = 5

WINDOW_SAMPLES = TARGET_FS * WINDOW_SECONDS

CHANNEL_NAMES = [
    "PCG",
    "PCG1",
]

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# DIRECTORY / FILE DISCOVERY
# ============================================================

def find_audio_files():
    """
    Find all WAV files recursively.
    """

    files = sorted(RAW_DIR.rglob("*.wav"))

    if not files:
        files = sorted(RAW_DIR.rglob("*.WAV"))

    return files


# ============================================================
# RESAMPLING
# ============================================================

def resample_signal(signal, original_fs, target_fs):
    """
    Resample signal from original_fs to target_fs.

    Uses polyphase resampling for efficient and high-quality
    conversion.
    """

    if original_fs == target_fs:
        return signal.astype(np.float32)

    gcd = np.gcd(int(original_fs), int(target_fs))

    up = int(target_fs // gcd)
    down = int(original_fs // gcd)

    resampled = resample_poly(
        signal,
        up,
        down,
        axis=0,
    )

    return resampled.astype(np.float32)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_channel(signal):
    """
    Robust per-channel standardization.

    Result approximately:
        mean = 0
        std  = 1
    """

    signal = np.asarray(signal, dtype=np.float32)

    # Replace invalid values
    signal = np.nan_to_num(
        signal,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    mean = np.mean(signal)

    std = np.std(signal)

    if std < 1e-8:
        return signal - mean

    return (signal - mean) / std


# ============================================================
# RECORD PROCESSING
# ============================================================

def process_record(audio_path):
    """
    Load and preprocess one Shiraz FHS record.

    Returns:
        windows:
            shape = (N, 2, 80000)
    """

    signal, fs = sf.read(
        audio_path,
        always_2d=True,
    )

    signal = np.asarray(
        signal,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Validate channels
    # --------------------------------------------------------

    if signal.shape[1] < 2:
        raise ValueError(
            f"{audio_path.stem}: expected at least 2 channels, "
            f"got {signal.shape[1]}"
        )

    # Keep first two channels:
    # PCG, PCG1
    signal = signal[:, :2]

    # --------------------------------------------------------
    # Handle sampling rate
    # --------------------------------------------------------

    original_fs = int(round(fs))

    if original_fs != TARGET_FS:

        print(
            f"{audio_path.stem}: "
            f"resampling {original_fs} Hz -> {TARGET_FS} Hz"
        )

        signal = resample_signal(
            signal,
            original_fs,
            TARGET_FS,
        )

    # --------------------------------------------------------
    # Clean invalid values
    # --------------------------------------------------------

    signal = np.nan_to_num(
        signal,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # --------------------------------------------------------
    # Normalize each channel
    # --------------------------------------------------------

    for channel in range(2):
        signal[:, channel] = normalize_channel(
            signal[:, channel]
        )

    # --------------------------------------------------------
    # Create windows
    # --------------------------------------------------------

    total_samples = signal.shape[0]

    n_windows = total_samples // WINDOW_SAMPLES

    if n_windows == 0:
        return np.empty(
            (0, 2, WINDOW_SAMPLES),
            dtype=np.float32,
        ), original_fs

    usable_samples = (
        n_windows * WINDOW_SAMPLES
    )

    signal = signal[:usable_samples]

    # Current shape:
    # (samples, channels)

    windows = signal.reshape(
        n_windows,
        WINDOW_SAMPLES,
        2,
    )

    # Convert to:
    # (windows, channels, samples)

    windows = np.transpose(
        windows,
        (0, 2, 1),
    )

    windows = windows.astype(
        np.float32
    )

    return windows, original_fs


# ============================================================
# DATASET SPLITTING
# ============================================================

def split_data(X):
    """
    Split windows into:
        70% train
        15% validation
        15% test
    """

    X_train, X_temp = train_test_split(
        X,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    relative_test_size = (
        TEST_RATIO
        / (VAL_RATIO + TEST_RATIO)
    )

    X_val, X_test = train_test_split(
        X_temp,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    return X_train, X_val, X_test


# ============================================================
# METADATA
# ============================================================

def save_metadata(
    audio_files,
    record_statistics,
    total_windows,
    train_shape,
    val_shape,
    test_shape,
):
    """
    Save preprocessing metadata.
    """

    metadata = {
        "dataset": "Shiraz Fetal Heart Sound",
        "target_sampling_rate_hz": TARGET_FS,
        "window_seconds": WINDOW_SECONDS,
        "window_samples": WINDOW_SAMPLES,
        "channels": CHANNEL_NAMES,
        "total_records": len(audio_files),
        "total_windows": total_windows,
        "split": {
            "train": list(train_shape),
            "validation": list(val_shape),
            "test": list(test_shape),
        },
        "random_state": RANDOM_STATE,
        "records": record_statistics,
    }

    metadata_path = OUTPUT_DIR / "metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SERENOVA — SHIRAZ FETAL HEART SOUND PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # Find records
    # --------------------------------------------------------

    audio_files = find_audio_files()

    print(
        f"Records: {len(audio_files)}"
    )

    print(
        f"Target sampling rate: {TARGET_FS} Hz"
    )

    print(
        f"Window: {WINDOW_SECONDS} seconds"
    )

    print(
        f"Samples/window: {WINDOW_SAMPLES}"
    )

    print(
        f"Channels: {', '.join(CHANNEL_NAMES)}"
    )

    print()

    if not audio_files:
        raise FileNotFoundError(
            f"No WAV files found in:\n{RAW_DIR}"
        )

    # --------------------------------------------------------
    # Process records
    # --------------------------------------------------------

    all_windows = []

    record_statistics = []

    total_windows = 0

    for audio_path in audio_files:

        try:

            windows, original_fs = process_record(
                audio_path
            )

            n_windows = len(windows)

            print(
                f"{audio_path.stem}: "
                f"{n_windows} windows "
                f"({original_fs} Hz)"
            )

            if n_windows == 0:

                print(
                    f"WARNING: {audio_path.stem} "
                    f"is shorter than {WINDOW_SECONDS} seconds"
                )

                continue

            all_windows.append(windows)

            total_windows += n_windows

            record_statistics.append(
                {
                    "record": audio_path.stem,
                    "original_sampling_rate_hz": original_fs,
                    "target_sampling_rate_hz": TARGET_FS,
                    "windows": n_windows,
                }
            )

        except Exception as e:

            print(
                f"ERROR processing {audio_path.stem}: {e}"
            )

    # --------------------------------------------------------
    # Make sure data exists
    # --------------------------------------------------------

    if not all_windows:

        raise RuntimeError(
            "No usable windows were generated."
        )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    X = np.concatenate(
        all_windows,
        axis=0,
    )

    X = X.astype(
        np.float32
    )

    print()

    print(
        f"Total windows: {len(X)}"
    )

    print(
        f"Final shape: {X.shape}"
    )

    # --------------------------------------------------------
    # Finite-value verification
    # --------------------------------------------------------

    finite = np.isfinite(X).all()

    print(
        f"All values finite: {finite}"
    )

    if not finite:

        raise ValueError(
            "Dataset contains NaN or infinite values."
        )

    # --------------------------------------------------------
    # Dataset split
    # --------------------------------------------------------

    X_train, X_val, X_test = split_data(X)

    print()

    print("-" * 60)
    print("PREPROCESSING COMPLETE")
    print("-" * 60)

    print(
        f"Train: {X_train.shape}"
    )

    print(
        f"Validation: {X_val.shape}"
    )

    print(
        f"Test: {X_test.shape}"
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save arrays
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR / "X_train.npy",
        X_train,
    )

    np.save(
        OUTPUT_DIR / "X_val.npy",
        X_val,
    )

    np.save(
        OUTPUT_DIR / "X_test.npy",
        X_test,
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    save_metadata(
        audio_files=audio_files,
        record_statistics=record_statistics,
        total_windows=total_windows,
        train_shape=X_train.shape,
        val_shape=X_val.shape,
        test_shape=X_test.shape,
    )

    print()

    print("Saved to:")
    print(OUTPUT_DIR)

    print("=" * 60)


if __name__ == "__main__":
    main()