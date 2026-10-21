"""
7. Quantum Model: 6-Qubit Amplitude-Encoded VQC (NO PCA)
=========================================================
Directly embeds all raw features into a 6-qubit quantum state vector via Amplitude Encoding:
  - 6 Qubits -> 2^6 = 64 computational basis states
  - Features F_1 ... F_m (m <= 64) are mapped as probability amplitudes c_0 ... c_{m-1}
  - Zero PCA dimensionality reduction needed: 100% of feature information is retained!
"""

import os, sys, time
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import pennylane as qml
from pennylane import numpy as pnp
from src.pdi.rules import pdi_proxy_label

def main():
    print("\n" + "="*68)
    print("  QUANTUM MODEL: 6-QUBIT AMPLITUDE-ENCODED VQC (ZERO PCA)")
    print("="*68)

    # 1. Load dataset
    df = pd.read_csv("artifacts/features/phase1_features.csv")
    df["pdi_label"] = [
        pdi_proxy_label(d, s, v, c)
        for d, s, v, c in zip(df["disc_ratio"], df["sat_mean"], df["val_mean"], df["circularity"])
    ]

    # Exclude non-features and score-defining features to avoid circular reasoning
    SCORE_FEATURES = {"disc_ratio", "sat_mean", "val_mean", "circularity"}
    EXCLUDE = {"path", "class", "pdi_score", "pdi_label"} | SCORE_FEATURES
    feat_cols = [c for c in df.columns if c not in EXCLUDE]

    # Balanced binary subset: Low vs High PDI
    df_bin = df[df["pdi_label"].isin(["Low", "High"])].copy()
    n_per = min(df_bin["pdi_label"].value_counts())
    balanced = pd.concat([
        df_bin[df_bin["pdi_label"] == lbl].sample(n=min(n_per, 110), random_state=42)
        for lbl in ["Low", "High"]
    ]).reset_index(drop=True)

    X_raw = np.nan_to_num(balanced[feat_cols].values)
    y = (balanced["pdi_label"] == "High").astype(int).values

    # 70% Train, 15% Val, 15% Test
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_raw, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    # Scale features to strictly positive values for amplitude normalization
    scaler = MinMaxScaler(feature_range=(0.05, 1.0)).fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    N_FEATS = len(feat_cols)
    N_QUBITS = 6
    N_STATES = 2 ** N_QUBITS
    N_LAYERS = 3

    print(f"\nAmplitude Encoding Architecture:")
    print(f"  Raw Input Features      : {N_FEATS} (ALL preserved, NO PCA projection!)")
    print(f"  Physical Qubits         : {N_QUBITS} Wires (ceil(log2({N_FEATS})) = {N_QUBITS})")
    print(f"  Hilbert State Space     : 2^{N_QUBITS} = {N_STATES} Computational Basis States")
    print(f"  Encoding Formula        : |psi> = sum_{{i=0}}^{{{N_FEATS-1}}} c_i |i> + sum_{{k={N_FEATS}}}^{{{N_STATES-1}}} 0 |k>")
    print(f"  Variational Parameters  : {N_LAYERS * N_QUBITS * 2} rotation angles + 2 classical head (scale, bias)")
    print(f"  Samples                 : Train={len(y_tr)}, Val={len(y_val)}, Test={len(y_te)}")

    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="autograd")
    def circuit(x, weights):
        # Embed all features directly into quantum state amplitudes
        qml.AmplitudeEmbedding(features=x, wires=range(N_QUBITS), pad_with=0.0, normalize=True)
        # Variational entanglement layers
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

    # Optimizer & Weights initialization
    rng = np.random.default_rng(42)
    weights = pnp.array(rng.normal(0, 0.3, (N_LAYERS, N_QUBITS, 2)), requires_grad=True)
    bias = pnp.array(0.0, requires_grad=True)
    scale = pnp.array(1.0, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.10)

    N_EPOCHS = 15
    BATCH_SIZE = 24
    best_val_acc = -1.0
    best_params = None

    print("\n" + "-"*68)
    print(f"  {'Epoch':>5} │ {'Train Acc':>10} │ {'Val Acc':>10} │ {'Val F1':>8} │ {'Time':>7} │ {'Status':>6}")
    print("-" * 68)

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        perm = rng.permutation(len(X_tr_s))

        for start in range(0, len(X_tr_s), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            Xb = X_tr_s[idx]
            yb = y_tr[idx]
            (weights, bias, scale), _ = opt.step_and_cost(
                lambda w, b, s: bce_loss(w, b, s, Xb, yb), weights, bias, scale
            )

        tr_pred, _ = predict(weights, bias, scale, X_tr_s)
        val_pred, _ = predict(weights, bias, scale, X_val_s)
        tr_acc = accuracy_score(y_tr, tr_pred)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        elapsed = time.time() - t0

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_params = (np.array(weights), float(bias), float(scale))
            status = "★ Best"
        else:
            status = ""

        print(f"  {epoch:5d} │ {tr_acc*100:9.2f}% │ {val_acc*100:9.2f}% │ {val_f1:8.4f} │ {elapsed:6.1f}s │ {status:>6}")
        sys.stdout.flush()

    # Evaluation on held-out test set
    w_best, b_best, s_best = best_params
    w_p = pnp.array(w_best, requires_grad=False)
    te_pred, te_probs = predict(w_p, b_best, s_best, X_te_s)

    te_acc = accuracy_score(y_te, te_pred)
    te_f1 = f1_score(y_te, te_pred, average="macro")
    te_prec = precision_score(y_te, te_pred, average="macro", zero_division=0)
    te_rec = recall_score(y_te, te_pred, average="macro", zero_division=0)
    te_auc = roc_auc_score(y_te, te_probs)
    te_mcc = matthews_corrcoef(y_te, te_pred)
    cm = confusion_matrix(y_te, te_pred)

    print("\n" + "="*68)
    print("  FINAL EVALUATION ON HELD-OUT TEST SET (ZERO PCA)")
    print("="*68)
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
    print("="*68 + "\n")

if __name__ == "__main__":
    main()
