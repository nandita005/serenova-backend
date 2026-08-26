from pathlib import Path
import numpy as np


ROOT = Path(r"D:\serenova_backend\data\processed")


def inspect_array(path):
    x = np.load(path, mmap_mode="r")

    finite = np.isfinite(x).all()

    print(f"    {path.name}")
    print(f"      Shape : {x.shape}")
    print(f"      Dtype : {x.dtype}")
    print(f"      Finite: {finite}")

    if finite:
        print(f"      Min   : {np.min(x):.6f}")
        print(f"      Max   : {np.max(x):.6f}")
        print(f"      Mean  : {np.mean(x):.6f}")
        print(f"      Std   : {np.std(x):.6f}")
    else:
        print("      WARNING: NaN/Inf detected!")

    return x.shape, finite


def main():

    print("=" * 70)
    print("SERENOVA — PROCESSED DATASET VALIDATION")
    print("=" * 70)
    print(f"Root: {ROOT}")
    print()

    if not ROOT.exists():
        raise FileNotFoundError(f"Processed directory not found: {ROOT}")

    datasets = []

    for directory in sorted(ROOT.iterdir()):
        if not directory.is_dir():
            continue

        train = directory / "X_train.npy"
        val = directory / "X_val.npy"
        test = directory / "X_test.npy"

        if train.exists() and val.exists() and test.exists():
            datasets.append(directory)

    print(f"Datasets found: {len(datasets)}")
    print()

    all_valid = True

    for dataset in datasets:

        print("-" * 70)
        print(f"DATASET: {dataset.name}")
        print("-" * 70)

        try:
            train_shape, train_finite = inspect_array(
                dataset / "X_train.npy"
            )

            val_shape, val_finite = inspect_array(
                dataset / "X_val.npy"
            )

            test_shape, test_finite = inspect_array(
                dataset / "X_test.npy"
            )

            dataset_valid = (
                train_finite
                and val_finite
                and test_finite
                and train_shape[0] > 0
                and val_shape[0] > 0
                and test_shape[0] > 0
            )

            if dataset_valid:
                print("    STATUS: PASS")
            else:
                print("    STATUS: FAIL")
                all_valid = False

        except Exception as e:
            print(f"    STATUS: ERROR")
            print(f"    {type(e).__name__}: {e}")
            all_valid = False

        print()

    print("=" * 70)

    if all_valid:
        print("VALIDATION COMPLETE — ALL DATASETS PASSED")
    else:
        print("VALIDATION COMPLETE — SOME DATASETS NEED ATTENTION")

    print("=" * 70)


if __name__ == "__main__":
    main()