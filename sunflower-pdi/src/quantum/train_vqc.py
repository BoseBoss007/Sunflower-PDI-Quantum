import os
import json
import yaml
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pennylane as qml
from pennylane import numpy as pnp

from src.quantum.vqc import create_qnode, bce_loss, accuracy_of

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def angle_scale(Xp, pmax):
    return np.pi * np.clip(Xp / pmax, -1, 1)

def run_train_vqc(base_config_path, phase2_config_path):
    base_config = load_config(base_config_path)
    phase2_config = load_config(phase2_config_path)
    
    artifacts_dir = base_config['artifacts_dir']
    models_dir = base_config['models_dir']
    seed = base_config['random_seed']
    
    df_pca = pd.read_csv(os.path.join(artifacts_dir, "features", "phase2_pca_features.csv"))
    
    n_qubits = phase2_config['vqc']['n_qubits']
    pca_cols = [f"pca_{i}" for i in range(n_qubits)]
    
    train_idx = df_pca['split'] == 'train'
    val_idx = df_pca['split'] == 'val'
    test_idx = df_pca['split'] == 'test'
    
    Xp_train = df_pca[train_idx][pca_cols].values
    Xp_val = df_pca[val_idx][pca_cols].values
    Xp_test = df_pca[test_idx][pca_cols].values
    
    y_train = (df_pca[train_idx]['pdi_label'].values == "High").astype(int)
    y_val = (df_pca[val_idx]['pdi_label'].values == "High").astype(int)
    y_test = (df_pca[test_idx]['pdi_label'].values == "High").astype(int)
    
    pmax = np.abs(Xp_train).max(axis=0) + 1e-8
    Xa_train = angle_scale(Xp_train, pmax)
    Xa_val = angle_scale(Xp_val, pmax)
    Xa_test = angle_scale(Xp_test, pmax)
    
    n_layers = phase2_config['vqc']['n_layers']
    qnode = create_qnode(n_qubits, n_layers)
    
    rng = np.random.default_rng(seed)
    weights = pnp.array(rng.normal(0, 0.3, size=(n_layers, n_qubits, 2)), requires_grad=True)
    bias = pnp.array(0.0, requires_grad=True)
    scale = pnp.array(1.0, requires_grad=True)
    
    lr = phase2_config['vqc']['learning_rate']
    opt = qml.AdamOptimizer(stepsize=lr)
    
    n_epochs = phase2_config['vqc']['epochs']
    batch_size = phase2_config['vqc']['batch_size']
    
    y_train_p = pnp.array(y_train, dtype=float)
    
    history = {"epoch": [], "train_loss": [], "val_acc": []}
    best_val = -1
    best_params = None
    
    n_train = len(Xa_train)
    
    print("Training VQC...")
    for epoch in range(n_epochs):
        perm = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start+batch_size]
            Xb = Xa_train[idx]
            yb = y_train_p[idx]
            (weights, bias, scale), _ = opt.step_and_cost(
                lambda w, b, s: bce_loss(qnode, w, b, s, Xb, yb), weights, bias, scale)
                
        tr_loss = float(bce_loss(qnode, weights, bias, scale, Xa_train[:120], y_train_p[:120]))
        val_acc, _, _ = accuracy_of(qnode, weights, bias, scale, Xa_val, y_val)
        
        history["epoch"].append(epoch+1)
        history["train_loss"].append(tr_loss)
        history["val_acc"].append(val_acc)
        
        if val_acc > best_val:
            best_val = val_acc
            best_params = (np.array(weights), float(bias), float(scale))
            
        print(f"epoch {epoch+1:2d}/{n_epochs}  train_loss={tr_loss:.4f}  val_acc={val_acc:.4f}")
        
    w_best, b_best, s_best = best_params
    w_best_p = pnp.array(w_best, requires_grad=False)
    
    test_acc, test_pred, test_probs = accuracy_of(qnode, w_best_p, b_best, s_best, Xa_test, y_test)
    test_f1 = f1_score(y_test, test_pred, average='macro')
    test_cm = confusion_matrix(y_test, test_pred).tolist()
    
    print(f"\nVQC TEST -> Accuracy={test_acc:.4f}  Macro-F1={test_f1:.4f}")
    
    # Save model
    np.savez(os.path.join(models_dir, "quantum", "vqc_parameters.npz"), weights=w_best, bias=b_best, scale=s_best)
    
    # Fair Classical SVM Comparator
    svm = SVC(kernel='rbf', C=5.0, gamma='scale', random_state=seed)
    svm.fit(Xp_train, y_train)
    svm_pred = svm.predict(Xp_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_f1 = f1_score(y_test, svm_pred, average='macro')
    print(f"SVM Comparator -> Accuracy={svm_acc:.4f}  Macro-F1={svm_f1:.4f}")
    
    # Save metrics
    res = {
        "vqc": {
            "accuracy": test_acc,
            "macro_f1": test_f1,
            "confusion_matrix": test_cm,
            "history": history
        },
        "svm_comparator": {
            "accuracy": svm_acc,
            "macro_f1": svm_f1,
            "confusion_matrix": confusion_matrix(y_test, svm_pred).tolist()
        }
    }
    
    with open(os.path.join(artifacts_dir, "metrics", "quantum_vqc_results.json"), "w") as f:
        json.dump(res, f, indent=2)
        
    with open(os.path.join(artifacts_dir, "metrics", "vqc_training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
        
    # Plot history
    plt.figure()
    plt.plot(history['epoch'], history['train_loss'], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("VQC Training Curve")
    plt.legend()
    plt.savefig(os.path.join(artifacts_dir, "figures", "vqc_training_curve.png"))
    plt.close()

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    run_train_vqc("configs/base.yaml", "configs/phase2.yaml")
