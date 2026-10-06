import torch

from src.model import HQCNN


# ---------------------------------------------
# Create model
# ---------------------------------------------

model = HQCNN()


# ---------------------------------------------
# Create fake input
# ---------------------------------------------
#
# Pretend we have 2 UWB samples.
#
# Each sample has 1016 CIR values.
#

x = torch.randn(
    2,
    1016
)


# ---------------------------------------------
# Forward pass
# ---------------------------------------------

output = model(x)


# ---------------------------------------------
# Print results
# ---------------------------------------------

print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)

print("\nModel output:")
print(output)

print("\nTrainable parameters:")

total_params = 0

for name, parameter in model.named_parameters():

    if parameter.requires_grad:

        print(
            name,
            parameter.shape,
            parameter.numel()
        )

        total_params += parameter.numel()


print("\nTotal trainable parameters:")
print(total_params)