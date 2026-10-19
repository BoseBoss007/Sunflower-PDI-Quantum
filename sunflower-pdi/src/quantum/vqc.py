import pennylane as qml
from pennylane import numpy as pnp
import numpy as np

def create_qnode(n_qubits, n_layers, device="default.qubit"):
    dev = qml.device(device, wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(x, weights):
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return qml.expval(qml.PauliZ(0))
        
    return circuit

def vqc_model(qnode, x, weights, bias, scale):
    z = qnode(x, weights)
    return 1.0 / (1.0 + pnp.exp(-(scale * z + bias)))

def bce_loss(qnode, weights, bias, scale, X, y):
    eps = 1e-7
    loss = 0.0
    for i in range(len(X)):
        p = vqc_model(qnode, X[i], weights, bias, scale)
        p = pnp.clip(p, eps, 1.0 - eps)
        yi = float(y[i])
        loss = loss + (yi * pnp.log(p) + (1.0 - yi) * pnp.log(1.0 - p))
    return -loss / len(X)

def accuracy_of(qnode, weights, bias, scale, X, y):
    preds = np.array([float(vqc_model(qnode, X[i], weights, bias, scale)) for i in range(len(X))])
    yhat = (preds >= 0.5).astype(int)
    from sklearn.metrics import accuracy_score
    return accuracy_score(y, yhat), yhat, preds
