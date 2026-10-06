import torch

from src.dataset import create_dataloaders
from src.hqcnn import HQCNN


print("=" * 60)
print("DATASET + HQCNN INTEGRATION TEST")
print("=" * 60)


# ============================================================
# 1. LOAD DATASET
# ============================================================

train_loader, val_loader, test_loader = create_dataloaders()


# ============================================================
# 2. CREATE MODEL
# ============================================================

model = HQCNN()


# ============================================================
# 3. GET ONE BATCH
# ============================================================

X_batch, y_batch = next(iter(train_loader))

print("\nInput batch:")
print(X_batch.shape)

print("\nLabel batch:")
print(y_batch.shape)


# ============================================================
# 4. TEST MODEL ON ONE BATCH
# ============================================================

print("\nRunning HQCNN on one batch...")

# IMPORTANT:
# Start with only 1 sample.
# Quantum simulation is computationally expensive.

X_test = X_batch[:1]
y_test = y_batch[:1]

output = model(X_test)


# ============================================================
# 5. PRINT RESULTS
# ============================================================

print("\nModel output:")
print(output)

print("\nOutput shape:")
print(output.shape)

print("\nTrue label:")
print(y_test)

print("\nPredicted class:")
print(torch.argmax(output, dim=1))


# ============================================================
# 6. TEST LOSS
# ============================================================

criterion = torch.nn.CrossEntropyLoss()

loss = criterion(
    output,
    y_test
)

print("\nLoss:")
print(loss)


# ============================================================
# 7. TEST BACKPROPAGATION
# ============================================================

print("\nTesting backward pass...")

loss.backward()

print("Backward pass successful ✓")


print("\n" + "=" * 60)
print("DATASET + HQCNN INTEGRATION TEST PASSED ✓")
print("=" * 60)