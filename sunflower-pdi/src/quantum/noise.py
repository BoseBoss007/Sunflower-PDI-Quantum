import os
import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from sklearn.metrics import accuracy_score

from src.quantum.train_vqc import angle_scale

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_noise_experiment(base_config_path, phase2_config_path):
    base_config = load_config(base_config_path)
    phase2_config = load_config(phase2_config_path)
    
    artifacts_dir = base_config['artifacts_dir']
    models_dir = base_config['models_dir']
    
    df_pca = pd.read_csv(os.path.join(artifacts_dir, "features", "phase2_pca_features.csv"))
    
    n_qubits = phase2_config['vqc']['n_qubits']
    pca_cols = [f"pca_{i}" for i in range(n_qubits)]
    
    train_idx = df_pca['split'] == 'train'
    test_idx = df_pca['split'] == 'test'
    
    Xp_train = df_pca[train_idx][pca_cols].values
    Xp_test = df_pca[test_idx][pca_cols].values
    y_test = (df_pca[test_idx]['pdi_label'].values == "High").astype(int)
    
    pmax = np.abs(Xp_train).max(axis=0) + 1e-8
    Xa_test = angle_scale(Xp_test, pmax)
    
    n_layers = phase2_config['vqc']['n_layers']
    
    params = np.load(os.path.join(models_dir, "quantum", "vqc_parameters.npz"))
    weights = params['weights']
    bias = params['bias']
    scale = params['scale']
    
    dev_noisy = qml.device("default.mixed", wires=n_qubits)
    
    @qml.qnode(dev_noisy)
    def circuit_noisy(x, w, p):
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
            qml.DepolarizingChannel(p, wires=i)
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(w[l, i, 0], wires=i)
                qml.RZ(w[l, i, 1], wires=i)
                qml.DepolarizingChannel(p, wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return qml.expval(qml.PauliZ(0))
        
    noise_levels = phase2_config['noise']['levels']
    results = []
    
    n_eval = min(30, len(Xa_test))
    Xa_test_sub = Xa_test[:n_eval]
    y_test_sub = y_test[:n_eval]
    
    print("Running noise simulation...")
    for p in noise_levels:
        preds = []
        for i in range(n_eval):
            z = circuit_noisy(Xa_test_sub[i], weights, p)
            prob = 1 / (1 + np.exp(-(scale * float(z) + bias)))
            preds.append(1 if prob >= 0.5 else 0)
        acc = accuracy_score(y_test_sub, preds)
        results.append({"p": p, "accuracy": acc, "n_eval": n_eval})
        print(f"p={p:.2f} -> acc={acc:.4f}")
        
    df_noise = pd.DataFrame(results)
    df_noise.to_csv(os.path.join(artifacts_dir, "metrics", "noise_sweep.csv"), index=False)
    
    plt.figure()
    plt.plot(df_noise['p'], df_noise['accuracy'], marker='o')
    plt.xlabel("Depolarizing Probability (p)")
    plt.ylabel("Accuracy")
    plt.title("VQC Noise Robustness")
    plt.grid(True)
    plt.savefig(os.path.join(artifacts_dir, "figures", "noise_robustness.png"))
    plt.close()

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    run_noise_experiment("configs/base.yaml", "configs/phase2.yaml")
