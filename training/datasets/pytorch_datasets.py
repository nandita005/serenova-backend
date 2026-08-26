from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ======================================================================
# PATH CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


# ======================================================================
# DATASETS
# ======================================================================

DATASETS = [
    "bidmc",
    "ctu_ctg",
    "diabetes",
    "fetal_ecg",
    "fetal_health",
    "maternal_health_risk",
    "mit_bih_nsr",
    "postnatal",
    "shiraz_fhs",
]


# ======================================================================
# SERENOVA NUMPY DATASET
# ======================================================================

class SerenovaNumpyDataset(Dataset):
    """
    PyTorch Dataset wrapper for Serenova processed NumPy datasets.

    Expected directory structure:

        data/
        └── processed/
            └── dataset_name/
                ├── X_train.npy
                ├── X_val.npy
                ├── X_test.npy
                ├── y_train.npy       # optional
                ├── y_val.npy         # optional
                └── y_test.npy        # optional

    Supports:

        - Signal datasets without labels
        - Classification datasets with labels
        - NumPy memory mapping
        - float32 input tensors
        - int64 classification labels
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
    ):

        if dataset_name not in DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}\n"
                f"Available datasets: {DATASETS}"
            )

        if split not in {"train", "val", "test"}:
            raise ValueError(
                "split must be one of: train, val, test"
            )

        self.dataset_name = dataset_name
        self.split = split

        # --------------------------------------------------------------
        # DATASET DIRECTORY
        # --------------------------------------------------------------

        self.dataset_dir = PROCESSED_ROOT / dataset_name

        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found:\n"
                f"{self.dataset_dir}"
            )

        # --------------------------------------------------------------
        # LOAD X
        # --------------------------------------------------------------

        x_path = self.dataset_dir / f"X_{split}.npy"

        if not x_path.exists():
            raise FileNotFoundError(
                f"Missing feature file:\n{x_path}"
            )

        # mmap_mode avoids loading the entire dataset into RAM
        self.X = np.load(
            x_path,
            mmap_mode="r"
        )

        # --------------------------------------------------------------
        # CHECK X
        # --------------------------------------------------------------

        if self.X.ndim < 2:
            raise ValueError(
                f"{dataset_name}/{split}: "
                f"X must have at least 2 dimensions, "
                f"got {self.X.shape}"
            )

        # --------------------------------------------------------------
        # LOAD LABELS IF AVAILABLE
        # --------------------------------------------------------------

        y_path = self.dataset_dir / f"y_{split}.npy"

        if y_path.exists():

            self.y = np.load(
                y_path,
                mmap_mode="r"
            )

            self.has_labels = True

            # Make sure number of samples matches
            if len(self.X) != len(self.y):
                raise ValueError(
                    f"{dataset_name}/{split}: "
                    f"X samples ({len(self.X)}) != "
                    f"y samples ({len(self.y)})"
                )

        else:

            self.y = None
            self.has_labels = False

        # --------------------------------------------------------------
        # CLASS INFORMATION
        # --------------------------------------------------------------

        if self.has_labels:

            classes, counts = np.unique(
                self.y,
                return_counts=True
            )

            self.classes = classes.astype(np.int64)
            self.class_counts = counts.astype(np.int64)

            # ----------------------------------------------------------
            # CLASS WEIGHTS
            #
            # weight = N / (num_classes * class_count)
            #
            # This gives:
            #
            # fetal_health:
            # class 0 -> ~0.4283
            # class 1 -> ~2.3961
            # class 2 -> ~4.0325
            # ----------------------------------------------------------

            total_samples = len(self.y)
            num_classes = len(self.classes)

            weights = (
                total_samples
                /
                (
                    num_classes
                    *
                    self.class_counts
                )
            )

            # Tensor instead of Python list
            self.class_weights = torch.tensor(
                weights,
                dtype=torch.float32
            )

        else:

            self.classes = None
            self.class_counts = None
            self.class_weights = None

    # ==================================================================
    # LENGTH
    # ==================================================================

    def __len__(self):

        return len(self.X)

    # ==================================================================
    # GET ITEM
    # ==================================================================

    def __getitem__(self, index):

        # --------------------------------------------------------------
        # X
        # --------------------------------------------------------------

        x = np.array(
    self.X[index],
    dtype=np.float32,
    copy=True
)

        x_tensor = torch.from_numpy(x)

        # --------------------------------------------------------------
        # LABELED DATASET
        # --------------------------------------------------------------

        if self.has_labels:

            y = int(self.y[index])

            y_tensor = torch.tensor(
                y,
                dtype=torch.int64
            )

            return x_tensor, y_tensor

        # --------------------------------------------------------------
        # UNLABELED SIGNAL DATASET
        # --------------------------------------------------------------

        return x_tensor


# ======================================================================
# DATALOADER CREATION
# ======================================================================

def create_dataloader(
    dataset_name: str,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """
    Create a PyTorch DataLoader for a Serenova dataset.
    """

    dataset = SerenovaNumpyDataset(
        dataset_name=dataset_name,
        split=split,
    )

    # pin_memory is useful when transferring CPU tensors to CUDA.
    # It is disabled automatically when CUDA is unavailable.
    if not torch.cuda.is_available():
        pin_memory = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader


# ======================================================================
# DATASET TEST
# ======================================================================

def test_dataset(name: str):

    print("\n" + "-" * 70)
    print(f"DATASET: {name}")
    print("-" * 70)

    # --------------------------------------------------------------
    # LOAD TRAIN DATASET
    # --------------------------------------------------------------

    train_dataset = SerenovaNumpyDataset(
        dataset_name=name,
        split="train",
    )

    print(f"Samples : {len(train_dataset)}")
    print(f"X shape: {train_dataset.X.shape}")
    print(f"X dtype: {train_dataset.X.dtype}")
    print(f"Labels : {train_dataset.has_labels}")

    # --------------------------------------------------------------
    # FINITE CHECK
    # --------------------------------------------------------------

    finite = np.isfinite(train_dataset.X).all()

    print(f"Finite  : {finite}")

    if not finite:
        raise ValueError(
            f"{name}: X contains NaN or Inf values"
        )

    # --------------------------------------------------------------
    # LABEL INFORMATION
    # --------------------------------------------------------------

    if train_dataset.has_labels:

        print(
            f"y shape: {train_dataset.y.shape}"
        )

        print(
            f"y dtype: {train_dataset.y.dtype}"
        )

        print(
            f"Classes: {train_dataset.classes.tolist()}"
        )

        print(
            f"Counts : {train_dataset.class_counts.tolist()}"
        )

        print(
            f"Class weights: "
            f"{train_dataset.class_weights.tolist()}"
        )

        if not np.isfinite(train_dataset.y).all():
            raise ValueError(
                f"{name}: y contains NaN or Inf values"
            )

    else:

        print("Class weights: N/A")

    # --------------------------------------------------------------
    # SINGLE SAMPLE
    # --------------------------------------------------------------

    sample = train_dataset[0]

    if train_dataset.has_labels:

        x, y = sample

        print(
            f"Sample X: {x.shape}"
        )

        print(
            f"Sample y: {y.item()}"
        )

        print(
            f"X dtype : {x.dtype}"
        )

        print(
            f"y dtype : {y.dtype}"
        )

        assert x.dtype == torch.float32
        assert y.dtype == torch.int64

    else:

        x = sample

        print(
            f"Sample X: {x.shape}"
        )

        print(
            f"X dtype : {x.dtype}"
        )

        assert x.dtype == torch.float32

    # --------------------------------------------------------------
    # DATALOADER
    # --------------------------------------------------------------

    loader = create_dataloader(
        dataset_name=name,
        split="train",
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    # --------------------------------------------------------------
    # LABELED BATCH
    # --------------------------------------------------------------

    if train_dataset.has_labels:

        X_batch, y_batch = batch

        print(
            f"Batch X : {X_batch.shape}"
        )

        print(
            f"Batch y : {y_batch.shape}"
        )

        print(
            f"X dtype : {X_batch.dtype}"
        )

        print(
            f"y dtype : {y_batch.dtype}"
        )

        print(
            f"Classes : "
            f"{torch.unique(y_batch).tolist()}"
        )

        assert X_batch.dtype == torch.float32
        assert y_batch.dtype == torch.int64

    # --------------------------------------------------------------
    # UNLABELED BATCH
    # --------------------------------------------------------------

    else:

        X_batch = batch

        print(
            f"Batch X : {X_batch.shape}"
        )

        print(
            f"X dtype : {X_batch.dtype}"
        )

        assert X_batch.dtype == torch.float32

    # --------------------------------------------------------------
    # GPU TEST
    # --------------------------------------------------------------

    if torch.cuda.is_available():

        X_gpu = X_batch.to(
            device="cuda",
            non_blocking=True
        )

        print(
            f"GPU X   : {X_gpu.shape}"
        )

        print(
            f"Device  : {X_gpu.device}"
        )

        # Make sure GPU tensor is actually CUDA
        assert X_gpu.is_cuda

        del X_gpu

        torch.cuda.empty_cache()

    else:

        print("GPU     : CUDA unavailable")

    print("STATUS  : PASS")


# ======================================================================
# MAIN TEST
# ======================================================================

def main():

    print("=" * 70)
    print("SERENOVA — ALL PYTORCH DATASET TEST")
    print("=" * 70)

    print(
        f"PyTorch version: {torch.__version__}"
    )

    print(
        f"PyTorch CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    passed = 0
    failed = 0

    # --------------------------------------------------------------
    # TEST EVERY DATASET
    # --------------------------------------------------------------

    for dataset_name in DATASETS:

        try:

            test_dataset(dataset_name)

            passed += 1

        except Exception as e:

            failed += 1

            print("\nSTATUS  : FAIL")

            print(
                f"ERROR   : "
                f"{type(e).__name__}: {e}"
            )

    # --------------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"Passed : {passed}/{len(DATASETS)}"
    )

    print(
        f"Failed : {failed}/{len(DATASETS)}"
    )

    if failed == 0:

        print(
            "\nALL PYTORCH DATASETS PASSED"
        )

    else:

        print(
            "\nSOME DATASETS FAILED "
            "— FIX BEFORE MODEL TRAINING"
        )

    print("=" * 70)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()