import torch
import torch.nn as nn

from src.quantum import QuantumHQCNNLayer


class HQCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # 1016 CIR features -> 8 quantum features
        self.input_layer = nn.Linear(1016, 8)

        # Quantum convolution + pooling + measurement
        self.quantum_layer = QuantumHQCNNLayer()

        # 4 quantum measurements -> 2 classes
        self.output_layer = nn.Linear(4, 2)

    def forward(self, x):

        # 1016 -> 8
        angles = self.input_layer(x)

        # Quantum processing
        quantum_output = self.quantum_layer(angles)

        # Quantum simulator currently returns float64
        # PyTorch Linear layers use float32
        quantum_output = quantum_output.float()

        # 4 -> 2
        output = self.output_layer(quantum_output)

        return output