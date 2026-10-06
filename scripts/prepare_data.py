import os
import glob
import numpy as np
import pandas as pd

DATASET_DIR = "data/UWB-LOS-NLOS-Data-Set-master/dataset"

OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))

if not files:
    raise FileNotFoundError(
        f"No CSV files found in {DATASET_DIR}"
    )

print(f"Found {len(files)} CSV files.")

frames = []

for file in files:
    print("Reading:", file)

    df = pd.read_csv(file)

    frames.append(df)

data = pd.concat(frames, ignore_index=True)

print("Dataset shape:", data.shape)

# First column is the label.
y = data.iloc[:, 0].values

# Last 1016 columns are CIR values.
X = data.iloc[:, -1016:].values

print("X shape:", X.shape)
print("y shape:", y.shape)

np.save(
    os.path.join(OUTPUT_DIR, "X.npy"),
    X.astype(np.float32)
)

np.save(
    os.path.join(OUTPUT_DIR, "y.npy"),
    y.astype(np.int64)
)

print("Saved processed dataset.")