"""
5. Baseline Model: Multi-Layer Perceptron (Neural Network)
==========================================================
3-Class PDI Classification (Low / Medium / High)
Iterative neural backpropagation showing live epoch progress, loss, and validation accuracy.
"""

import os, sys, time
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def main():
    print("\n" + "="*65)
    print("  BASELINE MODEL 5: MULTI-LAYER PERCEPTRON (NEURAL NETWORK)")
    print("="*65)

    df = pd.read_csv("artifacts/features/phase1_features.csv")
    from src.pdi.rules import pdi_proxy_label
    df["pdi_label"] = [
        pdi_proxy_label(d, s, v, c)
        for d, s, v, c in zip(df["disc_ratio"], df["sat_mean"], df["val_mean"], df["circularity"])
    ]

    SCORE_FEATURES = {"disc_ratio", "sat_mean", "val_mean", "circularity"}
    EXCLUDE = {"path", "class", "pdi_score", "pdi_label"} | SCORE_FEATURES
    feat_cols = [c for c in df.columns if c not in EXCLUDE]

    X = np.nan_to_num(df[feat_cols].values)
    label_map = {"Low": 0, "Medium": 1, "High": 2}
    y = df["pdi_label"].map(label_map).values

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)

    print(f"\nArchitecture:")
    print(f"  Input Layer   : {len(feat_cols)} Neurons")
    print(f"  Hidden Layers : Layer 1 (64 neurons, ReLU) → Layer 2 (32 neurons, ReLU)")
    print(f"  Output Layer  : 3 Neurons (Softmax Cross-Entropy)")
    print(f"  Optimizer     : Adam (learning_rate_init=0.005, weight_decay=1e-4)")

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.005,
        alpha=1e-4,
        max_iter=1,
        warm_start=True,
        random_state=42
    )

    classes = np.array([0, 1, 2])
    EPOCHS = 40
    best_val_acc = 0.0

    print("\n" + "-"*65)
    print(f"  {'Epoch':>6} │ {'Train Loss':>12} │ {'Train Acc':>10} │ {'Val Acc':>10} │ {'Status':>10}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        mlp.partial_fit(X_tr, y_tr, classes=classes)

        loss = mlp.loss_
        tr_pred = mlp.predict(X_tr)
        val_pred = mlp.predict(X_val)

        tr_acc = accuracy_score(y_tr, tr_pred)
        val_acc = accuracy_score(y_val, val_pred)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            status = "★ Best"
        else:
            status = ""

        print(f"  Epoch {epoch:2d}/{EPOCHS} │ {loss:12.4f} │ {tr_acc*100:9.2f}% │ {val_acc*100:9.2f}% │ {status:>8}")
        sys.stdout.flush()
        time.sleep(0.04)

    te_pred = mlp.predict(X_te)
    te_acc = accuracy_score(y_te, te_pred)
    te_f1 = f1_score(y_te, te_pred, average="macro")
    te_prec = precision_score(y_te, te_pred, average="macro", zero_division=0)
    te_rec = recall_score(y_te, te_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_te, te_pred)

    print("\n" + "="*65)
    print("  FINAL EVALUATION ON HELD-OUT TEST SET")
    print("="*65)
    print(f"  Test Accuracy : {te_acc*100:.2f}%")
    print(f"  Macro-F1      : {te_f1:.4f}")
    print(f"  Precision     : {te_prec:.4f}")
    print(f"  Recall        : {te_rec:.4f}")
    print(f"\n  Confusion Matrix (Rows=Actual, Columns=Predicted):")
    print(f"               Pred Low   Pred Med  Pred High")
    print(f"  Actual Low  :   {cm[0,0]:5d}      {cm[0,1]:5d}      {cm[0,2]:5d}")
    print(f"  Actual Med  :   {cm[1,0]:5d}      {cm[1,1]:5d}      {cm[1,2]:5d}")
    print(f"  Actual High :   {cm[2,0]:5d}      {cm[2,1]:5d}      {cm[2,2]:5d}")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()
