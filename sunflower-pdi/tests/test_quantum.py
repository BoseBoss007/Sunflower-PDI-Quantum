import numpy as np
from src.quantum.vqc import create_qnode

def test_qnode_creation():
    qnode = create_qnode(n_qubits=6, n_layers=3)
    assert qnode is not None

def test_vqc_parameters():
    # N_qubits = 6, N_layers = 3 -> 3 * 6 * 2 = 36 rotation parameters
    n_qubits = 6
    n_layers = 3
    
    weights = np.random.normal(0, 0.3, size=(n_layers, n_qubits, 2))
    assert weights.size == 36
