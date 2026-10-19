"""
LIVE TRAINING — Sunflower PDI Quantum Project
============================================================
Phase 1 : Classical Models  (LR, RF, SVM, XGBoost, LightGBM)
Phase 2 : Quantum VQC       (6-qubit PennyLane, live epochs)

Run:
    cd sunflower-pdi
    source venv/bin/activate
    PYTHONPATH=. python scripts/live_train.py
"""

import os, sys, time, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def banner(title, width=65):
    print("\n" + "═"*width)
    pad = (width - len(title) - 2) // 2
    print("║" + " "*pad + title + " "*(width - pad - len(title) - 2) + "║")
    print("═"*width)

def section(title):
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")

def tick():
    """Elapsed time string."""
    return f"[{time.strftime('%H:%M:%S')}]"

# ─────────────────────────────────────────────────────────
# 0. SETUP & DATA LOADING
# ─────────────────────────────────────────────────────────
banner("SUNFLOWER PDI — LIVE MODEL TRAINING")
print(f"{tick()} Loading data and applying rigorous preprocessing …")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib, pennylane as qml
from pennylane import numpy as pnp

SEED = 42
np.random.seed(SEED)

# ── Load features
df = pd.read_csv("artifacts/features/phase1_features.csv")

# ── Exclude the 4 features that define the PDI label (circular-reasoning fix)
SCORE_FEATURES = {"disc_ratio", "sat_mean", "val_mean", "circularity"}
ALL_FEAT = [c for c in df.columns if c not in ({"path","class","pdi_label"} | SCORE_FEATURES)]

# ── Outlier removal (3×IQR)
mask = pd.Series(True, index=df.index)
for col in ALL_FEAT:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask &= df[col].between(Q1 - 3*IQR, Q3 + 3*IQR)
df_c = df[mask].reset_index(drop=True)

# ── Feature engineering
skewed = ['contrast','h_hist_8','h_hist_3','h_hist_9','h_hist_16','h_hist_5','h_hist_2']
for col in skewed:
    df_c[f"{col}_log"] = np.log1p(df_c[col].clip(lower=0))
df_c["contrast_x_extent"]    = df_c["contrast"] * df_c["extent"]
df_c["energy_x_homogeneity"] = df_c["energy"]   * df_c["homogeneity"]
df_c["area_x_solidity"]      = df_c["area_ratio"] * df_c["solidity"]

FEAT_COLS = ALL_FEAT + [f"{c}_log" for c in skewed] + \
            ["contrast_x_extent","energy_x_homogeneity","area_x_solidity"]

# ── Feature selection (variance → correlation → RF importance)
X_raw = np.nan_to_num(df_c[FEAT_COLS].values)
vt    = VarianceThreshold(threshold=0.001)
X_var = vt.fit_transform(X_raw)
fv    = [f for f, m in zip(FEAT_COLS, vt.get_support()) if m]

corr_df = pd.DataFrame(X_var, columns=fv).corr().abs()
upper   = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
drop_c  = [c for c in upper.columns if any(upper[c] > 0.92)]
kf      = [f for f in fv if f not in drop_c]
X_sel   = X_var[:, [fv.index(f) for f in kf]]

rf_sel  = RandomForestClassifier(200, max_depth=8, random_state=SEED)
rf_sel.fit(StandardScaler().fit_transform(X_sel),
           (df_c["pdi_label"] == "High").astype(int).values)
imp = rf_sel.feature_importances_
idx = np.argsort(imp)[::-1]
cumsum = np.cumsum(imp[idx])
k   = max(int(np.searchsorted(cumsum, 0.85)) + 1, 8)
X_FINAL = X_sel[:, idx[:k]]
FINAL_FEATS = [kf[i] for i in idx[:k]]

# ── Labels
le3  = LabelEncoder().fit(["Low", "Medium", "High"])
y3   = le3.transform(df_c["pdi_label"].values)      # 3-class
y_stage = LabelEncoder().fit(["YoungBud","MatureBud","EarlyBloom","Healthy","Wilted"])\
                         .transform(df_c["class"].values)  # 5-class stage

print(f"{tick()} Data ready.")
print(f"  Clean samples  : {len(df_c)} (removed {len(df)-len(df_c)} outliers)")
print(f"  Features used  : {k}  →  {FINAL_FEATS}")
print(f"  3-class dist   : {dict(zip(le3.classes_, np.bincount(y3)))}")
print(f"  5-class dist   : {dict(zip(['EarlyBloom','Healthy','MatureBud','Wilted','YoungBud'], np.bincount(y_stage)))}")

# ═══════════════════════════════════════════════════════════
# PHASE 1 — CLASSICAL MODELS
# ═══════════════════════════════════════════════════════════
banner("PHASE 1 — CLASSICAL MODEL TRAINING")

outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

# Best hyperparameters from previous tuning
MODELS = {
    "Logistic Regression": Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=5.0, penalty="l1", solver="saga",
                                   max_iter=2000, random_state=SEED))
    ]),
    "SVM (RBF, C=5)": Pipeline([
        ("sc", StandardScaler()),
        ("clf", SVC(C=5.0, kernel="rbf", gamma="auto",
                    probability=True, random_state=SEED))
    ]),
    "Random Forest": Pipeline([
        ("sc", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=8,
                                       min_samples_leaf=5, max_features="log2",
                                       random_state=SEED, n_jobs=-1))
    ]),
    "XGBoost": Pipeline([
        ("sc", StandardScaler()),
        ("clf", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                              objective="multi:softmax", num_class=3,
                              random_state=SEED, n_jobs=-1, verbosity=0))
    ]),
    "LightGBM": Pipeline([
        ("sc", StandardScaler()),
        ("clf", LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                               num_leaves=63, reg_lambda=1.0, min_child_samples=20,
                               random_state=SEED, n_jobs=-1, verbose=-1))
    ]),
}

# Stacking ensemble (built on top of tuned base models)
MODELS["Stacking Ensemble"] = StackingClassifier(
    estimators=[(n, p) for n, p in list(MODELS.items())],
    final_estimator=LogisticRegression(C=0.5, max_iter=2000, random_state=SEED),
    cv=5, passthrough=False, n_jobs=-1
)

results_3class = {}

for model_name, pipeline in MODELS.items():
    section(f"Training: {model_name}  (3-class PDI: Low/Medium/High)")
    t0 = time.time()

    fold_accs, fold_f1s, fold_tr_accs = [], [], []
    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_FINAL, y3), 1):
        X_tr, X_te = X_FINAL[tr_idx], X_FINAL[te_idx]
        y_tr, y_te = y3[tr_idx], y3[te_idx]

        pipeline.fit(X_tr, y_tr)
        tr_acc = accuracy_score(y_tr, pipeline.predict(X_tr))
        te_acc = accuracy_score(y_te, pipeline.predict(X_te))
        te_f1  = f1_score(y_te, pipeline.predict(X_te), average="macro")

        fold_accs.append(te_acc)
        fold_f1s.append(te_f1)
        fold_tr_accs.append(tr_acc)

        bar = "█" * fold + "░" * (10 - fold)
        sys.stdout.write(
            f"\r  [{bar}] Fold {fold:2d}/10 │ "
            f"Train={tr_acc*100:5.1f}%  Test={te_acc*100:5.1f}%  F1={te_f1:.4f}"
        )
        sys.stdout.flush()

    print()  # newline after progress bar

    mean_tr  = np.mean(fold_tr_accs)
    mean_te  = np.mean(fold_accs)
    std_te   = np.std(fold_accs)
    mean_f1  = np.mean(fold_f1s)
    gap      = mean_tr - mean_te

    results_3class[model_name] = {
        "train_acc": mean_tr, "test_acc": mean_te,
        "test_std": std_te,   "f1": mean_f1, "gap": gap,
        "per_fold": fold_accs
    }

    elapsed = time.time() - t0
    print(f"  {'─'*55}")
    print(f"  Train Acc   : {mean_tr*100:.2f}%")
    print(f"  Test Acc    : {mean_te*100:.2f}% ± {std_te*100:.2f}%")
    print(f"  Macro F1    : {mean_f1:.4f}")
    print(f"  Overfit Gap : {gap*100:.2f}%")
    print(f"  Time        : {elapsed:.1f}s")

# ── Save best classical model
best_cls_name = max(results_3class, key=lambda k: results_3class[k]["f1"])
best_cls_pipe = MODELS[best_cls_name]
best_cls_pipe.fit(X_FINAL, y3)
joblib.dump(best_cls_pipe, "models/classical/best_3class_pdi.joblib")
print(f"\n{tick()} Best classical model: {best_cls_name} → saved to models/classical/best_3class_pdi.joblib")

# ═══════════════════════════════════════════════════════════
# PHASE 2 — QUANTUM VQC
# ═══════════════════════════════════════════════════════════
banner("PHASE 2 — QUANTUM VQC TRAINING (6-Qubit PennyLane)")

# ── Prepare binary Low/High subset
df_bin = df_c[df_c["pdi_label"].isin(["Low","High"])].copy()
n_per  = min(df_bin["pdi_label"].value_counts())
balanced = pd.concat([
    df_bin[df_bin["pdi_label"]==lbl].sample(n=min(n_per,110), random_state=SEED)
    for lbl in ["Low","High"]
]).reset_index(drop=True)

X_b = np.nan_to_num(balanced[FINAL_FEATS].values)
y_b = (balanced["pdi_label"]=="High").astype(int).values

X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_b, y_b, test_size=0.30, random_state=SEED, stratify=y_b)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)

print(f"{tick()} Binary subset: N={len(y_b)}  │  Train={len(y_tr)}  Val={len(y_val)}  Test={len(y_te)}")
print(f"  High-PDI: {y_b.sum()}   Low-PDI: {(1-y_b).sum()}")

# ── StandardScaler + PCA (fit on train only)
sc2  = StandardScaler().fit(X_tr)
pca2 = PCA(n_components=6, random_state=SEED).fit(sc2.transform(X_tr))
exp_var = pca2.explained_variance_ratio_.sum()
print(f"  PCA 6-D explained variance: {exp_var*100:.2f}%")

Xa_tr  = pca2.transform(sc2.transform(X_tr))
Xa_val = pca2.transform(sc2.transform(X_val))
Xa_te  = pca2.transform(sc2.transform(X_te))

# ── Angle scaling to [-π, π]
pmax = np.abs(Xa_tr).max(axis=0) + 1e-8
def angle_scale(Xp): return np.pi * np.clip(Xp / pmax, -1, 1)
Xa_tr, Xa_val, Xa_te = angle_scale(Xa_tr), angle_scale(Xa_val), angle_scale(Xa_te)

# ── Classical comparator (SVM on same 6-D PCA space)
section("Classical Comparator: SVM on identical 6-D PCA input")
t0 = time.time()
svm_comp = SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)
svm_comp.fit(Xa_tr, y_tr)
svm_tr_acc = accuracy_score(y_tr, svm_comp.predict(Xa_tr))
svm_te_acc = accuracy_score(y_te, svm_comp.predict(Xa_te))
svm_f1     = f1_score(y_te, svm_comp.predict(Xa_te), average="macro")
svm_auc    = roc_auc_score(y_te, svm_comp.predict_proba(Xa_te)[:,1])
print(f"  Train Acc   : {svm_tr_acc*100:.2f}%")
print(f"  Test  Acc   : {svm_te_acc*100:.2f}%")
print(f"  Macro F1    : {svm_f1:.4f}")
print(f"  ROC-AUC     : {svm_auc:.4f}")
print(f"  Time        : {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────────────────
# QUANTUM CIRCUIT DEFINITION
# ─────────────────────────────────────────────────────────
section("VQC Architecture: 6 qubits, 3 layers, ring CNOT entanglement")
N_QUBITS = 6
N_LAYERS = 3
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="autograd")
def circuit(x, weights):
    # State preparation: RY angle encoding
    for i in range(N_QUBITS):
        qml.RY(x[i], wires=i)
    # Variational layers: RY + RZ rotations + ring CNOT entanglement
    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[l, i, 0], wires=i)
            qml.RZ(weights[l, i, 1], wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
    return qml.expval(qml.PauliZ(0))

def qmodel(x, weights, bias, scale):
    z = circuit(x, weights)
    return 1.0 / (1.0 + pnp.exp(-(scale * z + bias)))  # sigmoid → P(High)

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

print(f"  Qubits  : {N_QUBITS}")
print(f"  Layers  : {N_LAYERS}")
print(f"  Params  : {N_LAYERS * N_QUBITS * 2} variational + 2 classical (bias, scale)")
print(f"  Encoding: RY angle encoding on 6-D PCA features")
print(f"  Readout : ⟨Z₀⟩ → sigmoid head → P(High PDI)")

# ─────────────────────────────────────────────────────────
# VQC TRAINING — LIVE EPOCHS
# ─────────────────────────────────────────────────────────
section("VQC Training — Live Epoch Progress")

rng = np.random.default_rng(SEED)
weights = pnp.array(rng.normal(0, 0.3, (N_LAYERS, N_QUBITS, 2)), requires_grad=True)
bias    = pnp.array(0.0, requires_grad=True)
scale   = pnp.array(1.0, requires_grad=True)
opt     = qml.AdamOptimizer(stepsize=0.15)

N_EPOCHS = 15
BATCH    = 24
history  = {"epoch":[], "train_loss":[], "train_acc":[], "val_acc":[], "val_f1":[]}
best_val_acc, best_params = -1, None

print(f"\n  {'Epoch':>5} │ {'Train Loss':>10} │ {'Train Acc':>9} │ {'Val Acc':>8} │ {'Val F1':>7} │ {'Time':>7} │ {'Best':>5}")
print(f"  {'─'*5}─┼─{'─'*10}─┼─{'─'*9}─┼─{'─'*8}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*5}")

t_start = time.time()
for epoch in range(1, N_EPOCHS + 1):
    t_ep = time.time()
    perm = rng.permutation(len(Xa_tr))

    # Mini-batch gradient updates
    for start in range(0, len(Xa_tr), BATCH):
        idx = perm[start:start + BATCH]
        Xb  = Xa_tr[idx]
        yb  = y_tr[idx]
        (weights, bias, scale), _ = opt.step_and_cost(
            lambda w, b, s: bce_loss(w, b, s, Xb, yb), weights, bias, scale
        )

    # Metrics
    tr_loss = float(bce_loss(weights, bias, scale, Xa_tr[:80], y_tr[:80]))
    tr_pred, _ = predict(weights, bias, scale, Xa_tr)
    val_pred, _ = predict(weights, bias, scale, Xa_val)
    tr_acc  = accuracy_score(y_tr,  tr_pred)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1  = f1_score(y_val, val_pred, average="macro")
    ep_time = time.time() - t_ep

    history["epoch"].append(epoch)
    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc
        best_params  = (np.array(weights), float(bias), float(scale))

    star = " ★ " if is_best else "   "
    print(f"  {epoch:>5} │ {tr_loss:>10.4f} │ {tr_acc*100:>8.2f}% │ {val_acc*100:>7.2f}% │ {val_f1:>7.4f} │ {ep_time:>6.1f}s │{star}")
    sys.stdout.flush()

# ─────────────────────────────────────────────────────────
# VQC FINAL EVALUATION ON TEST SET
# ─────────────────────────────────────────────────────────
section("VQC Final Test Evaluation (best checkpoint)")
w_best, b_best, s_best = best_params
w_p = pnp.array(w_best, requires_grad=False)
te_pred, te_probs = predict(w_p, b_best, s_best, Xa_te)

te_acc  = accuracy_score(y_te,  te_pred)
te_f1   = f1_score(y_te,        te_pred, average="macro")
te_prec = precision_score(y_te, te_pred, average="macro", zero_division=0)
te_rec  = recall_score(y_te,    te_pred, average="macro", zero_division=0)
te_auc  = roc_auc_score(y_te,   te_probs)
te_mcc  = matthews_corrcoef(y_te, te_pred)
te_cm   = confusion_matrix(y_te, te_pred)

print(f"\n  Test Accuracy  : {te_acc*100:.2f}%")
print(f"  Macro F1       : {te_f1:.4f}")
print(f"  Precision      : {te_prec:.4f}")
print(f"  Recall         : {te_rec:.4f}")
print(f"  ROC-AUC        : {te_auc:.4f}")
print(f"  MCC            : {te_mcc:.4f}")
print(f"\n  Confusion Matrix  (rows=actual, cols=predicted)")
print(f"                 Pred Low  Pred High")
print(f"  Actual Low   :    {te_cm[0,0]:4d}       {te_cm[0,1]:4d}")
print(f"  Actual High  :    {te_cm[1,0]:4d}       {te_cm[1,1]:4d}")

# ── Noise robustness sweep
section("Noise Robustness: Depolarizing Channel (default.mixed)")
noise_probs = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
noise_results = []
print(f"  {'Noise p':>8} │ {'Test Acc':>9} │ {'MCC':>7} │ {'Status'}")
print(f"  {'─'*8}─┼─{'─'*9}─┼─{'─'*7}─┼─{'─'*15}")

for p in noise_probs:
    dev_noisy = qml.device("default.mixed", wires=N_QUBITS)

    @qml.qnode(dev_noisy, interface="autograd")
    def noisy_circuit(x, weights, noise_p=p):
        for i in range(N_QUBITS):
            qml.RY(x[i], wires=i)
        for l in range(N_LAYERS):
            for i in range(N_QUBITS):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)
            for i in range(N_QUBITS):
                qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
            if noise_p > 0:
                for i in range(N_QUBITS):
                    qml.DepolarizingChannel(noise_p, wires=i)
        return qml.expval(qml.PauliZ(0))

    n_preds = []
    for xi in Xa_te:
        z = float(noisy_circuit(xi, w_p))
        prob = 1.0 / (1.0 + np.exp(-(s_best * z + b_best)))
        n_preds.append(int(prob >= 0.5))

    n_acc = accuracy_score(y_te, n_preds)
    n_mcc = matthews_corrcoef(y_te, n_preds)
    status = "✅ Robust" if n_acc > 0.80 else ("⚠️  Degraded" if n_acc > 0.60 else "❌ Collapsed")
    noise_results.append({"p": p, "acc": n_acc, "mcc": n_mcc})
    print(f"  {p:>8.2f} │ {n_acc*100:>8.2f}% │ {n_mcc:>7.4f} │ {status}")

# ─────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────
banner("FINAL RESULTS SUMMARY")

print("\n📋  PHASE 1 — 3-Class PDI  (10-Fold Stratified CV, N=1400+)\n")
print(f"  {'Model':<30} {'Train Acc':>10} {'Test Acc':>12} {'Macro-F1':>9} {'Gap':>8}")
print(f"  {'─'*30}─┼─{'─'*10}─{'─'*12}─{'─'*9}─{'─'*8}")
for name, r in results_3class.items():
    print(f"  {name:<30} {r['train_acc']*100:>9.2f}%  "
          f"{r['test_acc']*100:>6.2f}±{r['test_std']*100:.1f}%  "
          f"{r['f1']:>9.4f}  {r['gap']*100:>7.2f}%")

print(f"\n⚛️   PHASE 2 — Binary PDI  (Low vs High, 6-D PCA, N=220)\n")
print(f"  {'Model':<35} {'Train Acc':>10} {'Test Acc':>10} {'F1':>7} {'AUC':>7} {'MCC':>7}")
print(f"  {'─'*35}─┼─{'─'*10}─{'─'*10}─{'─'*7}─{'─'*7}─{'─'*7}")
tr_vqc = np.mean(history["train_acc"][-5:])
print(f"  {'SVM (6-D PCA comparator)':<35} {svm_tr_acc*100:>9.2f}%  "
      f"{svm_te_acc*100:>9.2f}%  {svm_f1:>7.4f}  {svm_auc:>7.4f}       —")
print(f"  {'VQC (6-qubit, 3-layer)':<35} {tr_vqc*100:>9.2f}%  "
      f"{te_acc*100:>9.2f}%  {te_f1:>7.4f}  {te_auc:>7.4f}  {te_mcc:>7.4f}")

print(f"\n🔊  Noise Robustness (VQC under Depolarizing Noise):\n")
print(f"  {'p':>6} │ {'Acc':>8} │ {'MCC':>7}")
for nr in noise_results:
    print(f"  {nr['p']:>6.2f} │ {nr['acc']*100:>7.2f}% │ {nr['mcc']:>7.4f}")

# ─────────────────────────────────────────────────────────
# SAVE ALL RESULTS
# ─────────────────────────────────────────────────────────
section("Saving All Results & Trained Models")

os.makedirs("artifacts/metrics", exist_ok=True)
os.makedirs("models/classical", exist_ok=True)
os.makedirs("models/quantum", exist_ok=True)

vqc_results = {
    "architecture": {"n_qubits": N_QUBITS, "n_layers": N_LAYERS,
                     "n_params": N_LAYERS*N_QUBITS*2, "encoding": "RY angle"},
    "pca_variance": float(exp_var),
    "history": history,
    "test": {"accuracy": te_acc, "f1": te_f1, "precision": te_prec,
             "recall": te_rec, "roc_auc": te_auc, "mcc": te_mcc,
             "confusion_matrix": te_cm.tolist()},
    "svm_comparator": {"train_acc": svm_tr_acc, "test_acc": svm_te_acc,
                       "f1": svm_f1, "roc_auc": svm_auc},
    "noise_robustness": noise_results,
}
with open("artifacts/metrics/vqc_final_results.json", "w") as f:
    json.dump(vqc_results, f, indent=2)

with open("artifacts/metrics/classical_final_results.json", "w") as f:
    json.dump(results_3class, f, indent=2)

np.savez("models/quantum/vqc_parameters.npz",
         weights=w_best, bias=np.array([b_best]), scale=np.array([s_best]))
joblib.dump(sc2,  "models/quantum/scaler.joblib")
joblib.dump(pca2, "models/quantum/pca.joblib")

total_time = time.time() - t_start
print(f"\n{tick()} ✅  All results saved to artifacts/metrics/")
print(f"  Models saved to models/classical/ and models/quantum/")
print(f"  Total training time: {total_time/60:.1f} minutes")
print(f"\n{'═'*65}")
print("  TRAINING COMPLETE")
print(f"{'═'*65}\n")
