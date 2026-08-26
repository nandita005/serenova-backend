from pathlib import Path
import json
import numpy as np
import wfdb


ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = (
    ROOT
    / "data"
    / "raw"
    / "external"
    / "bidmc"
    / "bidmc-ppg-and-respiration-dataset-1.0.0"
)

OUTPUT_DIR = ROOT / "data" / "processed" / "bidmc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_records():

    records = []

    for hea in DATASET_DIR.glob("*.hea"):
        name = hea.stem

        # BIDMC has normal signal records and
        # numerics records ending with "n".
        if not name.endswith("n"):
            records.append(name)

    return sorted(set(records))


def inspect_record(record_name):

    record_path = str(DATASET_DIR / record_name)

    record = wfdb.rdrecord(record_path)

    print("\n" + "-" * 60)
    print("Record:", record_name)
    print("Sampling frequency:", record.fs)
    print("Samples:", record.sig_len)

    print("\nSignal names:")
    print(record.sig_name)

    print("\nSignal units:")
    print(record.units)

    print("Signal shape:", record.p_signal.shape)

    return record


def main():

    print("\n" + "=" * 60)
    print("SERENOVA — BIDMC SIGNAL DATASET")
    print("=" * 60)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"BIDMC directory not found:\n{DATASET_DIR}"
        )

    records = find_records()

    print(f"Records found: {len(records)}")

    if not records:
        raise RuntimeError("No BIDMC records found.")

    # Inspect first record only.
    # We are NOT preprocessing all recordings yet.
    record = inspect_record(records[0])

    metadata = {
        "dataset": "BIDMC PPG and Respiration",
        "records_found": len(records),
        "first_record": records[0],
        "sampling_frequency": float(record.fs),
        "signal_names": record.sig_name,
        "units": record.units,
        "signal_length": int(record.sig_len),
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

    print("\n" + "=" * 60)
    print("BIDMC INSPECTION COMPLETE")
    print("=" * 60)

    print("Metadata saved to:")
    print(OUTPUT_DIR / "metadata.json")


if __name__ == "__main__":
    main()