"""
2. Baseline Model: Support Vector Machine (RBF Kernel)
======================================================
3-Class PDI Classification (Low / Medium / High)
Cross-validation fold-by-fold execution with live convergence progress.
"""

import sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def main():
    print("\n" + "="*65)
    print("  BASELINE MODEL 2: SUPPORT VECTOR MACHINE (RBF KERNEL)")
    print("="*65)

    df = pd.read_csv("artifacts/features/phase1_features.csv")
    from src.pdi.rules import pdi_score
    df["pdi_score"] = df.apply(pdi_score, axis=1)
    df["pdi_label"] = pd.cut(df["pdi_score"], bins=[-1, 35, 65, 101], labels=["Low", "Medium", "High"])

    SCORE_FEATURES = {"disc_ratio", "sat_mean", "val_mean", "circularity"}
    EXCLUDE = {"path", "class", "pdi_score", "pdi_label"} | SCORE_FEATURES
    feat_cols = [c for c in df.columns if c not in EXCLUDE]

    X = np.nan_to_num(df[feat_cols].values)
    label_map = {"Low": 0, "Medium": 1, "High": 2}
    y = df["pdi_label"].map(label_map).values

    print(f"\nConfiguration:")
    print(f"  Kernel     : Radial Basis Function (RBF)")
    print(f"  Penalty C  : 5.0")
    print(f"  Gamma      : scale")
    print(f"  Validation : 10-Fold Stratified Cross-Validation")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    fold_train_acc = []
    fold_test_acc = []
    fold_f1 = []
    y_true_all, y_pred_all = [], []

    print("\n" + "-"*65)
    print(f"  {'Fold':>6} │ {'Train Acc':>10} │ {'Test Acc':>10} │ {'Macro F1':>10} │ {'Time':>8}")
    print("-" * 65)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        t0 = time.time()
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        svm = SVC(C=5.0, kernel="rbf", gamma="scale", random_state=42)
        svm.fit(X_tr_s, y_tr)

        tr_pred = svm.predict(X_tr_s)
        te_pred = svm.predict(X_te_s)

        tr_acc = accuracy_score(y_tr, tr_pred)
        te_acc = accuracy_score(y_te, te_pred)
        f1 = f1_score(y_te, te_pred, average="macro")
        elapsed = time.time() - t0

        fold_train_acc.append(tr_acc)
        fold_test_acc.append(te_acc)
        fold_f1.append(f1)
        y_true_all.extend(y_te)
        y_pred_all.extend(te_pred)

        bar = "█" * fold + "░" * (10 - fold)
        print(f"  [{bar}] Fold {fold:2d}/10 │ {tr_acc*100:9.2f}% │ {te_acc*100:9.2f}% │ {f1:10.4f} │ {elapsed:7.2f}s")
        sys.stdout.flush()
        time.sleep(0.08)

    cm = confusion_matrix(y_true_all, y_pred_all)
    prec = precision_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    rec = recall_score(y_true_all, y_pred_all, average="macro", zero_division=0)

    print("\n" + "="*65)
    print("  10-FOLD CROSS-VALIDATION SUMMARY RESULTS")
    print("="*65)
    print(f"  Mean Train Acc : {np.mean(fold_train_acc)*100:.2f}%")
    print(f"  Mean Test Acc  : {np.mean(fold_test_acc)*100:.2f}% ± {np.std(fold_test_acc)*100:.2f}%")
    print(f"  Mean Macro-F1  : {np.mean(fold_f1):.4f}")
    print(f"  Precision      : {prec:.4f}")
    print(f"  Recall         : {rec:.4f}")
    print(f"  Generalization : Overfitting gap = {(np.mean(fold_train_acc)-np.mean(fold_test_acc))*100:.2f}%")
    print(f"\n  Aggregated Confusion Matrix (10 folds):")
    print(f"               Pred Low   Pred Med  Pred High")
    print(f"  Actual Low  :   {cm[0,0]:5d}      {cm[0,1]:5d}      {cm[0,2]:5d}")
    print(f"  Actual Med  :   {cm[1,0]:5d}      {cm[1,1]:5d}      {cm[1,2]:5d}")
    print(f"  Actual High :   {cm[2,0]:5d}      {cm[2,1]:5d}      {cm[2,2]:5d}")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()
