import torch
import pennylane as qml

from src.quantum import vqc3


dev = qml.device(
    "default.qubit",
    wires=2
)


@qml.qnode(
    dev,
    interface="torch"
)
def test_circuit(params):
    alpha = params[0]
    beta = params[1]
    gamma = params[2]

    # Start with |00>
    vqc3(
        alpha,
        beta,
        gamma,
        wires=[0, 1]
    )

    return qml.probs(wires=[0, 1])


params = torch.tensor(
    [0.1, 0.2, 0.3],
    dtype=torch.float32,
    requires_grad=True
)

result = test_circuit(params)

print("VQC3 output:")
print(result)

print("\nOutput shape:")
print(result.shape)

print("\nTrainable parameters:")
print(params)


@qml.qnode(
    dev,
    interface="torch"
)
def gradient_test(params):

    # Create a non-trivial input state
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)

    vqc3(
        params[0],
        params[1],
        params[2],
        wires=[0, 1]
    )

    return qml.expval(
        qml.PauliZ(0) @ qml.PauliZ(1)
    )

params = torch.tensor(
    [0.1, 0.2, 0.3],
    dtype=torch.float32,
    requires_grad=True
)

output = gradient_test(params)

output.backward()

print("\nOutput:")
print(output)

print("\nGradients:")
print(params.grad)