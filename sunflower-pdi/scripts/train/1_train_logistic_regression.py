"""
1. Baseline Model: Logistic Regression (Iterative Solver over Epochs)
=====================================================================
3-Class PDI Classification (Low / Medium / High)
Excludes score-defining features to prevent data leakage / circular reasoning.
"""

import sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def main():
    print("\n" + "="*65)
    print("  BASELINE MODEL 1: LOGISTIC REGRESSION (ONLINE SGD)")
    print("="*65)

    # 1. Load data
    df = pd.read_csv("artifacts/features/phase1_features.csv")
    from src.pdi.rules import pdi_score
    df["pdi_score"] = df.apply(pdi_score, axis=1)
    df["pdi_label"] = pd.cut(df["pdi_score"], bins=[-1, 35, 65, 101], labels=["Low", "Medium", "High"])

    # Exclude non-features and score-defining features (circular reasoning fix)
    SCORE_FEATURES = {"disc_ratio", "sat_mean", "val_mean", "circularity"}
    EXCLUDE = {"path", "class", "pdi_score", "pdi_label"} | SCORE_FEATURES
    feat_cols = [c for c in df.columns if c not in EXCLUDE]

    X = np.nan_to_num(df[feat_cols].values)
    label_map = {"Low": 0, "Medium": 1, "High": 2}
    y = df["pdi_label"].map(label_map).values

    # Train / Val / Test split (70% / 15% / 15%)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)

    print(f"\nDataset Summary:")
    print(f"  Features: {len(feat_cols)} (excluding score-defining features)")
    print(f"  Samples : Train={len(y_tr)}, Val={len(y_val)}, Test={len(y_te)}")
    print(f"  Classes : 0=Low ({sum(y==0)}), 1=Medium ({sum(y==1)}), 2=High ({sum(y==2)})")

    print("\n" + "-"*65)
    print(f"  {'Epoch':>6} │ {'Train Loss':>12} │ {'Train Acc':>10} │ {'Val Acc':>10} │ {'Status':>10}")
    print("-" * 65)

    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=1,
                        learning_rate="optimal", random_state=42, warm_start=True)
    classes = np.array([0, 1, 2])

    best_val_acc = 0.0
    EPOCHS = 40

    for epoch in range(1, EPOCHS + 1):
        # One pass over training data
        clf.partial_fit(X_tr, y_tr, classes=classes)

        # Evaluate
        tr_pred = clf.predict(X_tr)
        val_pred = clf.predict(X_val)
        tr_acc = accuracy_score(y_tr, tr_pred)
        val_acc = accuracy_score(y_val, val_pred)

        # Approximate log loss
        probs = clf.predict_proba(X_tr)
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)
        loss = -np.mean(np.log(probs[np.arange(len(y_tr)), y_tr]))

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            status = "★ Best"
        else:
            status = ""

        print(f"  Epoch {epoch:2d}/{EPOCHS} │ {loss:12.4f} │ {tr_acc*100:9.2f}% │ {val_acc*100:9.2f}% │ {status:>8}")
        sys.stdout.flush()
        time.sleep(0.04) # brief delay for live visual terminal feedback

    # Test evaluation
    te_pred = clf.predict(X_te)
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
