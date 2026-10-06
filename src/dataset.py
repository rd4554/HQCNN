import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = os.path.join(
    "data",
    "UWB-LOS-NLOS-Data-Set-master",
    "dataset"
)

BATCH_SIZE = 64
RANDOM_STATE = 42


# ============================================================
# LOAD ALL CSV FILES
# ============================================================

def load_uwb_dataset():
    """
    Load all 7 UWB CSV files.

    Returns
    -------
    X : numpy array
        Shape: (42000, 1016)

    y : numpy array
        Shape: (42000,)
        0 = LOS
        1 = NLOS
    """

    csv_files = sorted(
        glob.glob(
            os.path.join(DATA_DIR, "uwb_dataset_part*.csv")
        )
    )

    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No UWB CSV files found in: {DATA_DIR}"
        )

    print("=" * 60)
    print("LOADING UWB DATASET")
    print("=" * 60)

    print(f"Found {len(csv_files)} CSV files")

    dataframes = []

    for file in csv_files:
        print(f"Loading: {os.path.basename(file)}")

        df = pd.read_csv(file)

        print(f"  Shape: {df.shape}")

        dataframes.append(df)

    # Combine all 7 parts
    df = pd.concat(
        dataframes,
        ignore_index=True
    )

    print("\nCombined dataset shape:")
    print(df.shape)

    # --------------------------------------------------------
    # Extract CIR features
    # --------------------------------------------------------

    cir_columns = [
        f"CIR{i}"
        for i in range(1016)
    ]

    # Verify all CIR columns exist
    missing_columns = [
        col
        for col in cir_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing CIR columns: {missing_columns[:10]}"
        )

    X = df[cir_columns].values.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Extract labels
    # --------------------------------------------------------

    y = df["NLOS"].values.astype(
        np.int64
    )

    print("\nFinal feature shape:")
    print(X.shape)

    print("\nFinal label shape:")
    print(y.shape)

    print("\nLabel distribution:")

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    for label, count in zip(unique, counts):

        if label == 0:
            name = "LOS"
        else:
            name = "NLOS"

        print(
            f"  {int(label)} ({name}): {count}"
        )

    return X, y


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def create_datasets():

    X, y = load_uwb_dataset()

    print("\n" + "=" * 60)
    print("CREATING TRAIN / VALIDATION / TEST SPLITS")
    print("=" * 60)

    # --------------------------------------------------------
    # First split:
    # 80% train
    # 20% temporary
    # --------------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y
    )

    # --------------------------------------------------------
    # Split remaining 20% into:
    # 10% validation
    # 10% test
    # --------------------------------------------------------

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y_temp
    )

    print("\nSplit sizes:")

    print(
        f"Train:      {X_train.shape}"
    )

    print(
        f"Validation: {X_val.shape}"
    )

    print(
        f"Test:       {X_test.shape}"
    )

    # --------------------------------------------------------
    # Convert to PyTorch tensors
    # --------------------------------------------------------

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    X_val = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    )

    y_val = torch.tensor(
        y_val,
        dtype=torch.long
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.long
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_dataloaders():

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = create_datasets()

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    val_dataset = TensorDataset(
        X_val,
        y_val
    )

    test_dataset = TensorDataset(
        X_test,
        y_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("\n" + "=" * 60)
    print("DATALOADERS CREATED")
    print("=" * 60)

    print(
        f"Train batches: {len(train_loader)}"
    )

    print(
        f"Validation batches: {len(val_loader)}"
    )

    print(
        f"Test batches: {len(test_loader)}"
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    (
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders()

    print("\n" + "=" * 60)
    print("CHECKING ONE BATCH")
    print("=" * 60)

    X_batch, y_batch = next(
        iter(train_loader)
    )

    print(
        "Input batch shape:",
        X_batch.shape
    )

    print(
        "Label batch shape:",
        y_batch.shape
    )

    print(
        "First labels:",
        y_batch[:10]
    )

    print("\nDATASET TEST PASSED ✓")