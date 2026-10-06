
import torch

from src.quantum import QuantumConvolutionLayer


# Create quantum convolution layer
conv = QuantumConvolutionLayer()

# Eight input rotation angles
angles = torch.tensor(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    dtype=torch.float32
)

# Forward pass
output = conv(angles)

print("Input shape:")
print(angles.shape)

print("\nQuantum convolution output shape:")
print(output.shape)

print("\nTrainable weights:")
print(conv.weights)

print("\nNumber of trainable quantum parameters:")
print(conv.weights.numel())

# Test gradient flow
loss = torch.abs(output[1]) ** 2
loss.backward()

print("\nGradients:")
print(conv.weights.grad)

print("\nGradients available:")
print(conv.weights.grad is not None)