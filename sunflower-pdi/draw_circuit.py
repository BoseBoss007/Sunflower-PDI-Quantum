import pennylane as qml
import matplotlib.pyplot as plt
import os

os.makedirs('artifacts/figures', exist_ok=True)

# Define a 6-qubit device for drawing
dev = qml.device('default.qubit', wires=6)

@qml.qnode(dev)
def vqc_circuit(x, weights):
    # Angle Encoding
    for i in range(6):
        qml.RY(x[i], wires=i)
        
    # Variational Layers (showing 2 layers for clean drawing)
    for layer in range(2):
        # Entanglement
        for i in range(6):
            qml.CNOT(wires=[i, (i + 1) % 6])
        # Rotations
        for i in range(6):
            qml.RY(weights[layer][i][0], wires=i)
            qml.RZ(weights[layer][i][1], wires=i)
            
    return qml.expval(qml.PauliZ(0))

# Create dummy inputs
x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
weights = [[[0.1, 0.1] for _ in range(6)] for _ in range(2)]

# Draw using PennyLane's matplotlib drawer
fig, ax = qml.draw_mpl(vqc_circuit, decimals=2)(x, weights)

fig.suptitle("Phase 2: Variational Quantum Circuit (Angle Encoding + VQC)", fontsize=14, y=1.05)
plt.tight_layout()
plt.savefig('artifacts/figures/vqc_architecture.png', dpi=300, bbox_inches='tight')
print("Saved VQC architecture circuit to artifacts/figures/vqc_architecture.png")
