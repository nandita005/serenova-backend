from pathlib import Path
import wfdb


ROOT = Path(
    r"D:\serenova_backend\data\raw\external\mit_bih_nsr"
)

DATASET_DIR = (
    ROOT
    / "mit-bih-normal-sinus-rhythm-database-1.0.0"
)


def find_records():
    records = []

    for hea_file in DATASET_DIR.glob("*.hea"):
        name = hea_file.stem

        # Ignore backup/header variants
        if name.endswith("-"):
            continue

        records.append(name)

    return sorted(records)


def inspect_record(record_name):
    record_path = DATASET_DIR / record_name

    record = wfdb.rdrecord(str(record_path))

    print(f"Record: {record_name}")
    print(f"Sampling frequency: {record.fs}")
    print(f"Samples: {record.sig_len}")

    print("\nSignal names:")
    print(record.sig_name)

    print("\nSignal units:")
    print(record.units)

    print("\nSignal shape:")
    print(record.p_signal.shape)

    print("\nFirst 10 samples:")
    print(record.p_signal[:10])


def main():
    print("=" * 60)
    print("SERENOVA — MIT-BIH NORMAL SINUS RHYTHM INSPECTION")
    print("=" * 60)

    records = find_records()

    print(f"Records found: {len(records)}")

    if not records:
        raise RuntimeError(
            f"No records found in:\n{DATASET_DIR}"
        )

    print("\nInspecting first record:\n")

    inspect_record(records[0])

    print("\n" + "=" * 60)
    print("MIT-BIH NSR INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()