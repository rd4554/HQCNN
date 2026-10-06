import torch

from src.model import HQCNNInput


model = HQCNNInput()

x = torch.randn(
    1,
    1016
)

output = model(x)

print("Output shape:")
print(output.shape)