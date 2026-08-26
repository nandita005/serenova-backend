from pathlib import Path
import wfdb


ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = (
    ROOT
    / "data"
    / "raw"
    / "external"
    / "ctu_uhb_ctg"
    / "ctu-chb-intrapartum-cardiotocography-database-1.0.0"
)


def find_records():
    records = []

    for hea in DATASET_DIR.glob("*.hea"):
        records.append(hea.stem)

    return sorted(set(records))


def main():

    print("\n" + "=" * 60)
    print("SERENOVA — CTU-UHB CTG DATASET INSPECTION")
    print("=" * 60)

    records = find_records()

    print("Records found:", len(records))

    if not records:
        raise RuntimeError("No CTG records found.")

    record_name = records[0]

    print("\nInspecting:", record_name)

    record = wfdb.rdrecord(
        str(DATASET_DIR / record_name)
    )

    print("Sampling frequency:", record.fs)
    print("Samples:", record.sig_len)

    print("\nSignal names:")
    print(record.sig_name)

    print("\nSignal units:")
    print(record.units)

    print("\nSignal shape:")
    print(record.p_signal.shape)

    print("\nFirst 10 samples:")
    print(record.p_signal[:10])

    print("\n" + "=" * 60)
    print("CTU-UHB CTG INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()