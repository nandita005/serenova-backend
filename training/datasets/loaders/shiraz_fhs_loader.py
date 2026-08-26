from pathlib import Path
import wfdb
import soundfile as sf


ROOT = Path(
    r"D:\serenova_backend\data\raw\external\shiraz_fetal_heart_sound"
)

DATASET_DIR = (
    ROOT
    / "shiraz-university-fetal-heart-sounds-database-1.0.1"
)


def find_records():
    records = []

    for wav_file in DATASET_DIR.glob("*.wav"):
        records.append(wav_file.stem)

    return sorted(records)


def inspect_record(record_name):
    wav_path = DATASET_DIR / f"{record_name}.wav"

    signal, sampling_rate = sf.read(wav_path)

    print(f"Record: {record_name}")
    print(f"Sampling frequency: {sampling_rate} Hz")
    print(f"Signal shape: {signal.shape}")
    print(f"Duration: {len(signal) / sampling_rate:.2f} seconds")
    print(f"Data type: {signal.dtype}")

    print("\nFirst 20 samples:")
    print(signal[:20])

    hea_path = DATASET_DIR / f"{record_name}.hea"

    if hea_path.exists():
        print("\nHeader:")
        print(hea_path.read_text(errors="ignore")[:2000])


def main():

    print("=" * 60)
    print("SERENOVA — SHIRAZ FETAL HEART SOUND INSPECTION")
    print("=" * 60)

    records = find_records()

    print(f"Records found: {len(records)}")

    if not records:
        raise RuntimeError(
            f"No WAV records found in:\n{DATASET_DIR}"
        )

    print("\nInspecting first record:\n")

    inspect_record(records[0])

    print("\n" + "=" * 60)
    print("SHIRAZ FETAL HEART SOUND INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()