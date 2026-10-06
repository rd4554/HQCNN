import torch

from src.hqcnn import HQCNN


# ============================================================
# Create model
# ============================================================

model = HQCNN()

print("=" * 60)
print("HQCNN MODEL TEST")
print("=" * 60)


# ============================================================
# Create dummy input
#
# One sample with 1016 classical features
# ============================================================

x = torch.randn(
    1,
    1016,
    dtype=torch.float32
)

print("\nInput shape:")
print(x.shape)


# ============================================================
# Forward pass
# ============================================================

output = model(x)


print("\nOutput:")
print(output)

print("\nOutput shape:")
print(output.shape)


# ============================================================
# Check output shape
#
# Expected:
#
# [1, 2]
#
# because we have:
#
# 1 sample
# 2 classes
# ============================================================

assert output.shape == (1, 2), (
    f"Expected output shape (1, 2), "
    f"got {output.shape}"
)


# ============================================================
# Check that output contains valid numbers
# ============================================================

assert torch.all(torch.isfinite(output)), (
    "Model output contains NaN or infinity."
)


# ============================================================
# Test backward pass
# ============================================================

loss = output.sum()

loss.backward()


print("\nBackward pass:")
print("Successful")


# ============================================================
# Check trainable parameters
# ============================================================

print("\nTrainable parameters:")

for name, parameter in model.named_parameters():

    if parameter.requires_grad:

        print(
            f"{name}: "
            f"shape={tuple(parameter.shape)}"
        )


# ============================================================
# Final result
# ============================================================

print("\n" + "=" * 60)
print("HQCNN END-TO-END TEST PASSED ✓")
print("=" * 60)