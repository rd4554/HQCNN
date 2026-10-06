import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def load_data(seed=42):

    X = np.load("data/processed/X.npy")
    y = np.load("data/processed/y.npy")

    # First split:
    # 33600 train
    # 8400 temporary
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=8400,
        random_state=seed,
        stratify=y
    )

    # Second split:
    # 4200 validation
    # 4200 test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=4200,
        random_state=seed,
        stratify=y_temp
    )

    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


def make_loaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size=64
):

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader