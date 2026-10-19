"""
Stage 5 - Quantum-Enhanced Refinement (REAL PennyLane simulation, no
fabricated numbers).

Task simplified to BINARY PDI classification (Low deficiency vs High
deficiency, dropping the Medium class) - a standard simplification for
a tractable single-qubit-readout VQC proof-of-concept, and clearly
reported as such (the classical baseline in Stage 4 already covers the
full 3-class task for comparison).

Pipeline:
  1. Take the Stage-3 handcrafted features for the Low/High subset.
  2. Standardize + PCA to 6 components (fits on TRAIN split only).
  3. Angle-encode the 6 PCA components onto 6 qubits (RY rotations).
  4. A trainable variational circuit: 3 entangling layers of
     RY-RZ rotations + a ring of CNOTs (a real, executed circuit,
     not a diagram mock-up).
  5. Read out <Z> on qubit 0, pass through a trainable classical
     sigmoid head (bias + scale) -> P(High deficiency).
  6. Train by gradient descent (Adam, PennyLane autograd) minimizing
     binary cross-entropy on the TRAIN split, model-select on VAL,
     evaluate once on TEST.
  7. A classical baseline (SVM) is trained on the exact same
     train/test split of the SAME reduced 6-D PCA features, for an
     apples-to-apples comparison at equal information budget.
  8. Noise robustness: re-evaluate the trained VQC on the test set
     under a depolarizing-noise channel at several probabilities using
     PennyLane's default.mixed simulator (real density-matrix
     simulation, not an approximation).
"""
import numpy as np
import pandas as pd
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# pyrefly: ignore [missing-import]
import pennylane as qml
# pyrefly: ignore [missing-import]
from pennylane import numpy as pnp

OUT = "/home/claude/pipeline/outputs"
SEED = 42
np.random.seed(SEED)

df = pd.read_csv(f"{OUT}/features.csv")
sub = df[df["pdi_label"].isin(["Low", "High"])].copy()

# Balance classes for a clean binary task (real subsampling, seeded)
n_per_class = min(min(sub["pdi_label"].value_counts()), 110)  # capped for VQC training tractability (documented)
parts = []
for lbl, g in sub.groupby("pdi_label"):
    parts.append(g.sample(n=n_per_class, random_state=SEED))
sub_balanced = pd.concat(parts, axis=0).reset_index(drop=True)
print(f"Balanced Low/High subset: {len(sub_balanced)} real samples ({n_per_class} per class)")

feature_cols = [c for c in df.columns if c not in ("path", "class", "pdi_label")]
X_all = np.nan_to_num(sub_balanced[feature_cols].values)
y_all = (sub_balanced["pdi_label"].values == "High").astype(int)  # 1=High deficiency, 0=Low

X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.30, random_state=SEED, stratify=y_all)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)
print(f"Train/Val/Test: {len(y_train)}/{len(y_val)}/{len(y_test)}")

scaler = StandardScaler().fit(X_train)
X_train_s, X_val_s, X_test_s = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

N_QUBITS = 6
pca = PCA(n_components=N_QUBITS, random_state=SEED).fit(X_train_s)
Xp_train = pca.transform(X_train_s)
Xp_val = pca.transform(X_val_s)
Xp_test = pca.transform(X_test_s)
print(f"PCA explained variance ratio (sum, {N_QUBITS} comps): {pca.explained_variance_ratio_.sum():.4f}")

# scale PCA outputs to [-pi, pi] for angle encoding
pmax = np.abs(Xp_train).max(axis=0) + 1e-8
def angle_scale(Xp):
    return np.pi * np.clip(Xp / pmax, -1, 1)

Xa_train = angle_scale(Xp_train)
Xa_val = angle_scale(Xp_val)
Xa_test = angle_scale(Xp_test)

# ---------------- Quantum circuit (real PennyLane execution) ----------------
N_LAYERS = 3
dev = qml.device("default.qubit", wires=N_QUBITS)

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

qnode = qml.QNode(circuit, dev, interface="autograd")

def model(x, weights, bias, scale):
    z = qnode(x, weights)
    return 1 / (1 + pnp.exp(-(scale * z + bias)))  # sigmoid -> P(High)

def bce_loss(weights, bias, scale, X, y):
    preds = pnp.stack([model(X[i], weights, bias, scale) for i in range(len(X))])
    eps = 1e-7
    preds = pnp.clip(preds, eps, 1 - eps)
    return -pnp.mean(y * pnp.log(preds) + (1 - y) * pnp.log(1 - preds))

def accuracy_of(weights, bias, scale, X, y):
    preds = np.array([float(model(X[i], weights, bias, scale)) for i in range(len(X))])
    yhat = (preds >= 0.5).astype(int)
    return accuracy_score(y, yhat), yhat, preds

# init
rng = np.random.default_rng(SEED)
weights = pnp.array(rng.normal(0, 0.3, size=(N_LAYERS, N_QUBITS, 2)), requires_grad=True)
bias = pnp.array(0.0, requires_grad=True)
scale = pnp.array(1.0, requires_grad=True)

opt = qml.AdamOptimizer(stepsize=0.15)

N_EPOCHS = 15
BATCH = 24
y_train_p = pnp.array(y_train, dtype=float)
y_val_p = pnp.array(y_val, dtype=float)

history = {"epoch": [], "train_loss": [], "val_acc": []}
t0 = time.time()
n_train = len(Xa_train)
best_val = -1
best_params = None
for epoch in range(N_EPOCHS):
    perm = rng.permutation(n_train)
    for start in range(0, n_train, BATCH):
        idx = perm[start:start+BATCH]
        Xb = Xa_train[idx]
        yb = y_train_p[idx]
        (weights, bias, scale), _ = opt.step_and_cost(
            lambda w, b, s: bce_loss(w, b, s, Xb, yb), weights, bias, scale)
    tr_loss = float(bce_loss(weights, bias, scale, Xa_train[:120], y_train_p[:120]))
    val_acc, _, _ = accuracy_of(weights, bias, scale, Xa_val, y_val)
    history["epoch"].append(epoch+1)
    history["train_loss"].append(tr_loss)
    history["val_acc"].append(val_acc)
    if val_acc > best_val:
        best_val = val_acc
        best_params = (np.array(weights), float(bias), float(scale))
    print(f"epoch {epoch+1:2d}/{N_EPOCHS}  train_loss={tr_loss:.4f}  val_acc={val_acc:.4f}  elapsed={time.time()-t0:.1f}s")

w_best, b_best, s_best = best_params
w_best_p = pnp.array(w_best, requires_grad=False)
test_acc, test_pred, test_probs = accuracy_of(w_best_p, b_best, s_best, Xa_test, y_test)
test_f1 = f1_score(y_test, test_pred, average='macro')
test_cm = confusion_matrix(y_test, test_pred).tolist()
print(f"\nQVC TEST -> Accuracy={test_acc:.4f}  Macro-F1={test_f1:.4f}")
print("Confusion matrix (rows=true Low/High, cols=pred):", test_cm)

# ---------------- Classical baseline on the SAME 6-D PCA features ----------------
svm_pca = SVC(kernel='rbf', C=5.0, gamma='scale', random_state=SEED)
svm_pca.fit(Xp_train, y_train)
svm_pred = svm_pca.predict(Xp_test)
svm_acc = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred, average='macro')
svm_cm = confusion_matrix(y_test, svm_pred).tolist()
print(f"\nClassical SVM on SAME 6-D PCA features -> Accuracy={svm_acc:.4f}  Macro-F1={svm_f1:.4f}")

# Also: classical baseline on FULL (unreduced) features for the same train/test split
scaler_full = StandardScaler().fit(X_train)
svm_full = SVC(kernel='rbf', C=5.0, gamma='scale', random_state=SEED)
svm_full.fit(scaler_full.transform(X_train), y_train)
svm_full_pred = svm_full.predict(scaler_full.transform(X_test))
svm_full_acc = accuracy_score(y_test, svm_full_pred)
svm_full_f1 = f1_score(y_test, svm_full_pred, average='macro')
print(f"Classical SVM on FULL (34-D) features   -> Accuracy={svm_full_acc:.4f}  Macro-F1={svm_full_f1:.4f}")

# ---------------- Noise robustness (REAL density-matrix simulation) ----------------
dev_noisy = qml.device("default.mixed", wires=N_QUBITS)

def circuit_noisy(x, weights, p):
    for i in range(N_QUBITS):
        qml.RY(x[i], wires=i)
        qml.DepolarizingChannel(p, wires=i)
    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[l, i, 0], wires=i)
            qml.RZ(weights[l, i, 1], wires=i)
            qml.DepolarizingChannel(p, wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
    return qml.expval(qml.PauliZ(0))

qnode_noisy = qml.QNode(circuit_noisy, dev_noisy)

noise_levels = [0.0, 0.01, 0.03, 0.05, 0.08, 0.12]
noise_results = []
# use a subset of test set for the noisy density-matrix sim (more expensive) - real run, documented
n_noise_eval = min(30, len(Xa_test))
Xa_test_sub = Xa_test[:n_noise_eval]
y_test_sub = y_test[:n_noise_eval]
for p in noise_levels:
    preds = []
    for i in range(n_noise_eval):
        z = qnode_noisy(Xa_test_sub[i], w_best_p, p)
        prob = 1 / (1 + np.exp(-(s_best * float(z) + b_best)))
        preds.append(1 if prob >= 0.5 else 0)
    acc = accuracy_score(y_test_sub, preds)
    noise_results.append({"p": p, "accuracy": acc, "n_eval": n_noise_eval})
    print(f"Depolarizing p={p:.2f}  ->  accuracy on {n_noise_eval} test samples = {acc:.4f}")

# ---------------- Ablation: number of qubits / PCA components ----------------
ablation = []
for nq in [2, 4, 6]:
    pca_a = PCA(n_components=nq, random_state=SEED).fit(X_train_s)
    svm_a = SVC(kernel='rbf', C=5.0, gamma='scale', random_state=SEED)
    svm_a.fit(pca_a.transform(X_train_s), y_train)
    acc_a = accuracy_score(y_test, svm_a.predict(pca_a.transform(X_test_s)))
    var_a = pca_a.explained_variance_ratio_.sum()
    ablation.append({"n_components": nq, "classical_svm_acc_same_dims": acc_a, "explained_var": var_a})
    print(f"[ablation] n_components={nq}  explained_var={var_a:.3f}  classical_svm_acc={acc_a:.4f}")

results = {
    "n_samples_total": int(len(sub_balanced)),
    "n_train": int(len(y_train)), "n_val": int(len(y_val)), "n_test": int(len(y_test)),
    "n_qubits": N_QUBITS, "n_layers": N_LAYERS, "n_epochs": N_EPOCHS,
    "pca_explained_variance_6d": float(pca.explained_variance_ratio_.sum()),
    "vqc_history": history,
    "vqc_test_accuracy": float(test_acc), "vqc_test_macro_f1": float(test_f1),
    "vqc_test_confusion_matrix": test_cm,
    "svm_pca6d_test_accuracy": float(svm_acc), "svm_pca6d_test_macro_f1": float(svm_f1),
    "svm_pca6d_confusion_matrix": svm_cm,
    "svm_full_test_accuracy": float(svm_full_acc), "svm_full_test_macro_f1": float(svm_full_f1),
    "noise_robustness": noise_results,
    "ablation_qubits": ablation,
}
with open(f"{OUT}/quantum_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved quantum_results.json")
