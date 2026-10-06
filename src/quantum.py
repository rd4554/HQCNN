import math

import pennylane as qml
import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

N_QUBITS = 8


# ============================================================
# QUANTUM DEVICE
# ============================================================

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


# ============================================================
# QUANTUM CONVOLUTION PAIRS
# ============================================================

# Paper's 8-qubit HQCNN:
#
# {1,2}, {3,4}, {5,6}, {7,8},
# {2,3}, {4,5}, {6,7}, {8,1}
#
# Python uses 0-based indexing:
#
# {1,2} -> (0,1)
# {3,4} -> (2,3)
# {5,6} -> (4,5)
# {7,8} -> (6,7)
# {2,3} -> (1,2)
# {4,5} -> (3,4)
# {6,7} -> (5,6)
# {8,1} -> (7,0)

CONV_PAIRS = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 0),
]


# ============================================================
# QUANTUM POOLING PAIRS
# ============================================================

# Paper:
#
# {1,2}, {3,4}, {5,6}, {7,8}
#
# Python 0-based:
#
# {1,2} -> (0,1)
# {3,4} -> (2,3)
# {5,6} -> (4,5)
# {7,8} -> (6,7)

POOL_PAIRS = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
]


# ============================================================
# VQC3
# ============================================================

# This is the explicit VQC3 circuit shown in the paper's
# Figure 7.
#
# VQC3 has 3 trainable parameters:
#
#     theta1
#     theta2
#     theta3
#
# Gate sequence:
#
#     RZ(-pi/2) on q2
#     CNOT q2 -> q1
#     RZ(theta1) on q1
#     RY(theta2) on q2
#     CNOT q1 -> q2
#     RY(theta3) on q2
#     CNOT q2 -> q1
#     RZ(pi/2) on q1
#
# q1 and q2 are the two qubits in the pair.


def vqc3(theta1, theta2, theta3, wires):

    q1, q2 = wires

    # --------------------------------------------------------
    # 1. RZ(-pi/2) on second qubit
    # --------------------------------------------------------

    qml.RZ(
        -math.pi / 2,
        wires=q2
    )

    # --------------------------------------------------------
    # 2. CNOT: q2 -> q1
    # --------------------------------------------------------

    qml.CNOT(
        wires=[q2, q1]
    )

    # --------------------------------------------------------
    # 3. RZ(theta1) on q1
    # --------------------------------------------------------

    qml.RZ(
        theta1,
        wires=q1
    )

    # --------------------------------------------------------
    # 4. RY(theta2) on q2
    # --------------------------------------------------------

    qml.RY(
        theta2,
        wires=q2
    )

    # --------------------------------------------------------
    # 5. CNOT: q1 -> q2
    # --------------------------------------------------------

    qml.CNOT(
        wires=[q1, q2]
    )

    # --------------------------------------------------------
    # 6. RY(theta3) on q2
    # --------------------------------------------------------

    qml.RY(
        theta3,
        wires=q2
    )

    # --------------------------------------------------------
    # 7. CNOT: q2 -> q1
    # --------------------------------------------------------

    qml.CNOT(
        wires=[q2, q1]
    )

    # --------------------------------------------------------
    # 8. RZ(pi/2) on q1
    # --------------------------------------------------------

    qml.RZ(
        math.pi / 2,
        wires=q1
    )


# ============================================================
# VQC3 TEST CIRCUIT
# ============================================================

# This circuit is ONLY for testing VQC3 independently.
#
# Input:
#     3 trainable parameters
#
# Output:
#     probabilities of the 4 possible states
#
# Why 4?
#
#     2 qubits
#         ↓
#     2^2 = 4 basis states
#
# The four states are:
#
#     |00>
#     |01>
#     |10>
#     |11>


dev_vqc3 = qml.device(
    "default.qubit",
    wires=2
)


@qml.qnode(
    dev_vqc3,
    interface="torch",
    diff_method="backprop"
)
def test_vqc3_circuit(params):

    theta1 = params[0]
    theta2 = params[1]
    theta3 = params[2]

    vqc3(
        theta1,
        theta2,
        theta3,
        wires=[0, 1]
    )

    return qml.probs(
        wires=[0, 1]
    )


# ============================================================
# QUANTUM CONVOLUTION CIRCUIT
# ============================================================

# Pipeline:
#
# 8 classical values
#        ↓
#   RX embedding
#        ↓
#  8 VQC3 blocks
#        ↓
#  quantum state
#
# This is the separate convolution-only circuit.
#
# IMPORTANT:
# This layer uses 8 DIFFERENT VQC3 parameter sets.
#
# Therefore:
#
#     8 VQC3 blocks × 3 parameters
#                         = 24 parameters
#
# This layer is kept for independent testing.
#
# The COMPLETE HQCNN below uses the paper's shared
# 3 convolution parameters instead.


@qml.qnode(
    dev,
    interface="torch",
    diff_method="backprop"
)
def quantum_conv_circuit(
    angles,
    weights
):

    # --------------------------------------------------------
    # STEP 1: Quantum embedding
    # --------------------------------------------------------

    # Each of the 8 classical values becomes
    # an RX rotation angle on one qubit.

    for i in range(N_QUBITS):

        qml.RX(
            angles[i],
            wires=i
        )

    # --------------------------------------------------------
    # STEP 2: Quantum convolution
    # --------------------------------------------------------

    # Apply VQC3 to all 8 convolution pairs.

    for i, (q1, q2) in enumerate(CONV_PAIRS):

        vqc3(
            weights[i, 0],
            weights[i, 1],
            weights[i, 2],
            wires=[q1, q2]
        )

    # --------------------------------------------------------
    # STEP 3: Return complete quantum state
    # --------------------------------------------------------

    # 8 qubits -> 2^8 = 256 amplitudes

    return qml.state()


# ============================================================
# PYTORCH QUANTUM CONVOLUTION LAYER
# ============================================================

class QuantumConvolutionLayer(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # 8 VQC3 blocks
        #
        # Each VQC3 has 3 parameters.
        #
        # 8 × 3 = 24 parameters
        # ----------------------------------------------------

        self.weights = nn.Parameter(
            0.01 * torch.randn(8, 3)
        )

    def forward(self, angles):

        # ----------------------------------------------------
        # SINGLE SAMPLE
        #
        # angles shape:
        #
        # [8]
        # ----------------------------------------------------

        if angles.dim() == 1:

            return quantum_conv_circuit(
                angles,
                self.weights
            )

        # ----------------------------------------------------
        # BATCH OF SAMPLES
        #
        # angles shape:
        #
        # [batch_size, 8]
        # ----------------------------------------------------

        outputs = []

        for sample in angles:

            result = quantum_conv_circuit(
                sample,
                self.weights
            )

            outputs.append(result)

        return torch.stack(outputs)


# ============================================================
# QUANTUM POOLING CIRCUIT
# ============================================================

# Paper's proposed pooling circuit 2,
# shown in Figure 8(b).
#
# It has 3 trainable parameters:
#
#     theta1
#     theta2
#     theta3
#
# Gate sequence:
#
#     RZ(-pi/2) on q2
#     CNOT q2 -> q1
#     RZ(theta1) on q1
#     RY(theta2) on q2
#     CNOT q1 -> q2
#     RY(theta3) on q2
#
# IMPORTANT:
#
# This replaces the previous implementation that used:
#
#     RY(-pi/2)
#     controlled-RZ
#     RY
#     RY
#
# The version below follows the explicit circuit in
# Figure 8(b).


def pooling_circuit(
    theta1,
    theta2,
    theta3,
    wires
):

    q1, q2 = wires

    # --------------------------------------------------------
    # 1. RZ(-pi/2) on q2
    # --------------------------------------------------------

    qml.RZ(
        -math.pi / 2,
        wires=q2
    )

    # --------------------------------------------------------
    # 2. CNOT: q2 -> q1
    # --------------------------------------------------------

    qml.CNOT(
        wires=[q2, q1]
    )

    # --------------------------------------------------------
    # 3. RZ(theta1) on q1
    # --------------------------------------------------------

    qml.RZ(
        theta1,
        wires=q1
    )

    # --------------------------------------------------------
    # 4. RY(theta2) on q2
    # --------------------------------------------------------

    qml.RY(
        theta2,
        wires=q2
    )

    # --------------------------------------------------------
    # 5. CNOT: q1 -> q2
    # --------------------------------------------------------

    qml.CNOT(
        wires=[q1, q2]
    )

    # --------------------------------------------------------
    # 6. RY(theta3) on q2
    # --------------------------------------------------------

    qml.RY(
        theta3,
        wires=q2
    )


# ============================================================
# COMPLETE 8-QUBIT HQCNN QUANTUM CIRCUIT
# ============================================================

# Complete pipeline:
#
# 8 classical values
#        ↓
# RX embedding
#        ↓
# 8 VQC3 convolution blocks
#        ↓
# 4 pooling blocks
#        ↓
# Measure qubits 2,4,6,8
#        ↓
# 4 expectation values
#
#
# IMPORTANT:
#
# The same 3 convolution parameters are shared
# across all 8 VQC3 blocks.
#
# The same 3 pooling parameters are shared
# across all 4 pooling blocks.
#
# Therefore:
#
# Quantum parameters:
#
#     convolution = 3
#     pooling     = 3
#     ----------------
#     total       = 6
#
# This matches the paper's reported parameter count.


@qml.qnode(
    dev,
    interface="torch",
    diff_method="backprop"
)
def quantum_hqcnn_circuit(
    angles,
    conv_weights,
    pool_weights
):

    # --------------------------------------------------------
    # STEP 1: Quantum embedding
    # --------------------------------------------------------

    # 8 classical values
    #        ↓
    # 8 RX rotations
    #        ↓
    # 8 qubits

    for i in range(N_QUBITS):

        qml.RX(
            angles[i],
            wires=i
        )

    # --------------------------------------------------------
    # STEP 2: Quantum convolution
    # --------------------------------------------------------

    # There are 8 VQC3 blocks.
    #
    # BUT they share the same 3 parameters:
    #
    # conv_weights[0]
    # conv_weights[1]
    # conv_weights[2]

    for q1, q2 in CONV_PAIRS:

        vqc3(
            conv_weights[0],
            conv_weights[1],
            conv_weights[2],
            wires=[q1, q2]
        )

    # --------------------------------------------------------
    # STEP 3: Quantum pooling
    # --------------------------------------------------------

    # There are 4 pooling blocks.
    #
    # They share the same 3 parameters:
    #
    # pool_weights[0]
    # pool_weights[1]
    # pool_weights[2]

    for q1, q2 in POOL_PAIRS:

        pooling_circuit(
            pool_weights[0],
            pool_weights[1],
            pool_weights[2],
            wires=[q1, q2]
        )

    # --------------------------------------------------------
    # STEP 4: Measurement
    # --------------------------------------------------------

    # Paper measures:
    #
    # qubit 2
    # qubit 4
    # qubit 6
    # qubit 8
    #
    # Python uses 0-based indexing:
    #
    # qubit 2 -> wire 1
    # qubit 4 -> wire 3
    # qubit 6 -> wire 5
    # qubit 8 -> wire 7
    #
    # Therefore we obtain:
    #
    # 4 quantum output values.

    return (
        qml.expval(
            qml.PauliZ(1)
        ),
        qml.expval(
            qml.PauliZ(3)
        ),
        qml.expval(
            qml.PauliZ(5)
        ),
        qml.expval(
            qml.PauliZ(7)
        )
    )


# ============================================================
# PYTORCH COMPLETE QUANTUM LAYER
# ============================================================

class QuantumHQCNNLayer(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # SHARED CONVOLUTION PARAMETERS
        # ----------------------------------------------------
        #
        # One VQC3 contains:
        #
        #     theta1
        #     theta2
        #     theta3
        #
        # These 3 parameters are shared by all 8
        # convolution blocks.
        #
        # Therefore:
        #
        #     3 quantum convolution parameters
        #

        self.conv_weights = nn.Parameter(
            0.01 * torch.randn(3)
        )

        # ----------------------------------------------------
        # SHARED POOLING PARAMETERS
        # ----------------------------------------------------
        #
        # One pooling block contains:
        #
        #     theta1
        #     theta2
        #     theta3
        #
        # These 3 parameters are shared by all 4
        # pooling blocks.
        #
        # Therefore:
        #
        #     3 quantum pooling parameters
        #

        self.pool_weights = nn.Parameter(
            0.01 * torch.randn(3)
        )

    def forward(self, angles):

        # ----------------------------------------------------
        # SINGLE SAMPLE
        # ----------------------------------------------------
        #
        # Input:
        #
        #     [8]
        #
        # Output:
        #
        #     [4]
        #
        # ----------------------------------------------------

        if angles.dim() == 1:

            output = quantum_hqcnn_circuit(
                angles,
                self.conv_weights,
                self.pool_weights
            )

            return torch.stack(output)

        # ----------------------------------------------------
        # BATCH
        # ----------------------------------------------------
        #
        # Input:
        #
        #     [batch_size, 8]
        #
        # Output:
        #
        #     [batch_size, 4]
        #
        # ----------------------------------------------------

        outputs = []

        for sample in angles:

            output = quantum_hqcnn_circuit(
                sample,
                self.conv_weights,
                self.pool_weights
            )

            outputs.append(
                torch.stack(output)
            )

        return torch.stack(outputs)