import torch

from src.quantum import quantum_embedding


# Eight example angles
angles = torch.tensor(
    [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8
    ],
    dtype=torch.float32
)


result = quantum_embedding(angles)


print("Quantum state shape:")
print(result.shape)

print("\nQuantum state:")
print(result)