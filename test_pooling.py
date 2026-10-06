import torch

from src.quantum import QuantumHQCNNLayer


layer = QuantumHQCNNLayer()

angles = torch.tensor(
    [0.1, 0.2, 0.3, 0.4,
     0.5, 0.6, 0.7, 0.8],
    dtype=torch.float32
)

output = layer(angles)

print("Input:")
print(angles)

print("\nInput shape:")
print(angles.shape)

print("\nPooling / HQCNN quantum output:")
print(output)

print("\nOutput shape:")
print(output.shape)

print("\nConvolution parameters:")
print(layer.conv_weights.shape)

print("\nPooling parameters:")
print(layer.pool_weights.shape)

print("\nTotal quantum parameters:")
print(
    layer.conv_weights.numel()
    + layer.pool_weights.numel()
)