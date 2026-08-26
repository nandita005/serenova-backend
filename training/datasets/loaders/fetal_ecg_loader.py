from pathlib import Path
import pyedflib


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


def main():

    print("\n" + "=" * 60)
    print("SERENOVA — NON-INVASIVE FETAL ECG INSPECTION")
    print("=" * 60)

    files = sorted(DATASET_DIR.glob("*.edf"))

    print("EDF records found:", len(files))

    if not files:
        raise RuntimeError("No EDF files found.")

    file = files[0]

    print("\nInspecting:", file.name)

    reader = pyedflib.EdfReader(str(file))

    try:
        labels = reader.getSignalLabels()
        sample_rates = reader.getSampleFrequencies()
        samples = reader.getNSamples()

        print("\nChannels:")
        for i, label in enumerate(labels):
            print(
                f"{i}: {label} | "
                f"sampling={sample_rates[i]} Hz | "
                f"samples={samples[i]}"
            )

        print("\nNumber of channels:", reader.signals_in_file)

        print("\nFirst channel first 20 samples:")

        signal = reader.readSignal(0)

        print(signal[:20])

    finally:
        reader.close()

    print("\n" + "=" * 60)
    print("FETAL ECG INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()