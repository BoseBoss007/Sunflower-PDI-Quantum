"""
6. Quantum Model: Variational Quantum Classifier (VQC)
======================================================
Binary PDI Classification (Low vs High)
6 Qubits, 3 Variational Layers, Ring CNOT Entanglement
Real-time live epoch training via PennyLane Adam optimizer.
"""

import sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import pennylane as qml
from pennylane import numpy as pnp

def main():
    print("\n" + "="*65)
    print("  QUANTUM MODEL: 6-QUBIT VARIATIONAL QUANTUM CLASSIFIER (VQC)")
    print("="*65)

    df = pd.read_csv("artifacts/features/phase1_features.csv")
    from src.pdi.rules import pdi_score
    df["pdi_score"] = df.apply(pdi_score, axis=1)
    df["pdi_label"] = pd.cut(df["pdi_score"], bins=[-1, 35, 65, 101], labels=["Low", "Medium", "High"])

    # Exclude non-features and score-defining features
    SCORE_FEATURES = {"disc_ratio", "sat_mean", "val_mean", "circularity"}
    EXCLUDE = {"path", "class", "pdi_score", "pdi_label"} | SCORE_FEATURES
    feat_cols = [c for c in df.columns if c not in EXCLUDE]

    # Binary balanced subset: Low vs High PDI
    df_bin = df[df["pdi_label"].isin(["Low", "High"])].copy()
    n_per = min(df_bin["pdi_label"].value_counts())
    balanced = pd.concat([
        df_bin[df_bin["pdi_label"] == lbl].sample(n=min(n_per, 110), random_state=42)
        for lbl in ["Low", "High"]
    ]).reset_index(drop=True)

    X_raw = np.nan_to_num(balanced[feat_cols].values)
    y = (balanced["pdi_label"] == "High").astype(int).values

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_raw, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    # Dimensionality Reduction: PCA 6-D to map to 6 qubits
    scaler = StandardScaler().fit(X_tr)
    pca = PCA(n_components=6, random_state=42).fit(scaler.transform(X_tr))
    X_tr_pca = pca.transform(scaler.transform(X_tr))
    X_val_pca = pca.transform(scaler.transform(X_val))
    X_te_pca = pca.transform(scaler.transform(X_te))

    # Angle scaling to [-π, π] for RY rotations
    pmax = np.abs(X_tr_pca).max(axis=0) + 1e-8
    def angle_scale(data): return np.pi * np.clip(data / pmax, -1, 1)
    X_tr_enc, X_val_enc, X_te_enc = angle_scale(X_tr_pca), angle_scale(X_val_pca), angle_scale(X_te_pca)

    N_QUBITS = 6
    N_LAYERS = 3
    print(f"\nQuantum Circuit Specification:")
    print(f"  Physical Qubits     : {N_QUBITS} Wires")
    print(f"  Hilbert Space Dim   : 2^{N_QUBITS} = {2**N_QUBITS} Complex Amplitudes")
    print(f"  Variational Layers  : {N_LAYERS}")
    print(f"  Trainable Parameters: {N_LAYERS * N_QUBITS * 2} variational + 2 classical head (scale, bias)")
    print(f"  Entanglement Topology: Ring CNOT (periodic boundary)")
    print(f"  State Encoding      : RY Pauli Angle Encoding (6-D PCA, explained var = {pca.explained_variance_ratio_.sum()*100:.1f}%)")

    # Define PennyLane Quantum Device & Node
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="autograd")
    def circuit(x, weights):
        for i in range(N_QUBITS):
            qml.RY(x[i], wires=i)
        for l in range(N_LAYERS):
            for i in range(N_QUBITS):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)
            for i in range(N_QUBITS):
                qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
        return qml.expval(qml.PauliZ(0))

    def qmodel(x, weights, bias, scale):
        z = circuit(x, weights)
        return 1.0 / (1.0 + pnp.exp(-(scale * z + bias)))

    def bce_loss(weights, bias, scale, X, y):
        eps = 1e-7
        loss = 0.0
        for i in range(len(X)):
            p = qmodel(X[i], weights, bias, scale)
            p = pnp.clip(p, eps, 1.0 - eps)
            yi = float(y[i])
            loss = loss + (yi * pnp.log(p) + (1.0 - yi) * pnp.log(1.0 - p))
        return -loss / len(X)

    def predict(weights, bias, scale, X):
        probs = np.array([float(qmodel(X[i], weights, bias, scale)) for i in range(len(X))])
        return (probs >= 0.5).astype(int), probs

    # Optimizer initialization
    rng = np.random.default_rng(42)
    weights = pnp.array(rng.normal(0, 0.3, (N_LAYERS, N_QUBITS, 2)), requires_grad=True)
    bias = pnp.array(0.0, requires_grad=True)
    scale = pnp.array(1.0, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.15)

    N_EPOCHS = 15
    BATCH_SIZE = 24
    best_val_acc = -1.0
    best_params = None

    print("\n" + "-"*65)
    print(f"  {'Epoch':>5} │ {'Train Loss':>10} │ {'Train Acc':>9} │ {'Val Acc':>8} │ {'Val F1':>7} │ {'Time':>7} │ {'Best':>5}")
    print("-" * 65)

    for epoch in range(1, N_EPOCHS + 1):
        t_ep = time.time()
        perm = rng.permutation(len(X_tr_enc))

        # Mini-batch gradient descent
        for start in range(0, len(X_tr_enc), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            Xb = X_tr_enc[idx]
            yb = y_tr[idx]
            (weights, bias, scale), _ = opt.step_and_cost(
                lambda w, b, s: bce_loss(w, b, s, Xb, yb), weights, bias, scale
            )

        # Loss and Metrics
        tr_loss = float(bce_loss(weights, bias, scale, X_tr_enc[:80], y_tr[:80]))
        tr_pred, _ = predict(weights, bias, scale, X_tr_enc)
        val_pred, _ = predict(weights, bias, scale, X_val_enc)
        tr_acc = accuracy_score(y_tr, tr_pred)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        ep_time = time.time() - t_ep

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_params = (np.array(weights), float(bias), float(scale))
            star = "★ Best"
        else:
            star = ""

        print(f"  {epoch:5d} │ {tr_loss:10.4f} │ {tr_acc*100:8.2f}% │ {val_acc*100:7.2f}% │ {val_f1:7.4f} │ {ep_time:6.1f}s │ {star:>5}")
        sys.stdout.flush()

    # Final evaluation using best parameters
    w_best, b_best, s_best = best_params
    w_p = pnp.array(w_best, requires_grad=False)
    te_pred, te_probs = predict(w_p, b_best, s_best, X_te_enc)

    te_acc = accuracy_score(y_te, te_pred)
    te_f1 = f1_score(y_te, te_pred, average="macro")
    te_prec = precision_score(y_te, te_pred, average="macro", zero_division=0)
    te_rec = recall_score(y_te, te_pred, average="macro", zero_division=0)
    te_auc = roc_auc_score(y_te, te_probs)
    te_mcc = matthews_corrcoef(y_te, te_pred)
    cm = confusion_matrix(y_te, te_pred)

    print("\n" + "="*65)
    print("  FINAL VQC EVALUATION ON HELD-OUT TEST SET")
    print("="*65)
    print(f"  Test Accuracy  : {te_acc*100:.2f}%")
    print(f"  Macro-F1       : {te_f1:.4f}")
    print(f"  Precision      : {te_prec:.4f}")
    print(f"  Recall         : {te_rec:.4f}")
    print(f"  ROC-AUC        : {te_auc:.4f}")
    print(f"  MCC            : {te_mcc:.4f}")
    print(f"\n  Confusion Matrix (Rows=Actual, Columns=Predicted):")
    print(f"                 Pred Low   Pred High")
    print(f"  Actual Low  :     {cm[0,0]:4d}         {cm[0,1]:4d}")
    print(f"  Actual High :     {cm[1,0]:4d}         {cm[1,1]:4d}")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()
