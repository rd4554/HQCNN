import torch
import torch.nn as nn

from src.quantum import QuantumHQCNNLayer


class HQCNN(nn.Module):
    """
    Hybrid Quantum Convolutional Neural Network

    Architecture:

        Input: 1016 features
              ↓
        Classical feature reduction
              1016 → 8
              ↓
        Tanh
              ↓
        8-qubit quantum layer
              ↓
        Quantum pooling
              ↓
        4 expectation values
              ↓
        Classical classifier
              4 → 2
              ↓
        Output: 2 class logits
    """

    def __init__(self):
        super().__init__()

        # --------------------------------------------------
        # 1. Classical feature reduction
        # --------------------------------------------------
        self.feature_reduction = nn.Linear(
            1016,
            8
        )

        # --------------------------------------------------
        # 2. Quantum layer
        # --------------------------------------------------
        self.quantum_layer = QuantumHQCNNLayer()

        # --------------------------------------------------
        # 3. Classical output classifier
        # --------------------------------------------------
        self.classifier = nn.Linear(
            4,
            2
        )

    def forward(self, x):

        # ----------------------------------------------
        # Input
        # ----------------------------------------------
        # x shape:
        # [batch_size, 1016]
        # ----------------------------------------------

        x = self.feature_reduction(x)

        # x shape:
        # [batch_size, 8]

        # ----------------------------------------------
        # Convert the 8 classical features into
        # bounded values for quantum angle encoding
        # ----------------------------------------------

        # x = torch.tanh(x)

        # x shape:
        # [batch_size, 8]

        # ----------------------------------------------
        # Quantum convolution + quantum pooling
        # ----------------------------------------------

        x = self.quantum_layer(x)

        # PennyLane returns float64 by default.
        # PyTorch Linear layers normally use float32.
        # Convert before the classifier.
        x = x.float()

        # x shape:
        # [batch_size, 4]

        # ----------------------------------------------
        # Classical classifier
        # ----------------------------------------------

        x = self.classifier(x)

        # x shape:
        # [batch_size, 2]

        return x