"""
3. Baseline Model: Random Forest Ensemble
==========================================
3-Class PDI Classification (Low / Medium / High)
Incremental tree building showing live convergence over 150 estimators.
"""

import os, sys, time
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def main():
    print("\n" + "="*65)
    print("  BASELINE MODEL 3: RANDOM FOREST ENSEMBLE")
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

    print(f"\nConfiguration:")
    print(f"  Target Estimators : 150 Decision Trees")
    print(f"  Max Depth         : 12")
    print(f"  Min Samples Split : 5")
    print(f"  Evaluation        : Progressive evaluation at every 10 trees")

    print("\n" + "-"*65)
    print(f"  {'Trees Built':>12} │ {'Train Acc':>10} │ {'Val Acc':>10} │ {'Val Macro F1':>12} │ {'Elapsed':>8}")
    print("-" * 65)

    rf = RandomForestClassifier(n_estimators=0, max_depth=12, min_samples_split=5,
                                warm_start=True, random_state=42, n_jobs=-1)

    t0 = time.time()
    for n_trees in range(10, 151, 10):
        rf.n_estimators = n_trees
        rf.fit(X_tr, y_tr)

        tr_pred = rf.predict(X_tr)
        val_pred = rf.predict(X_val)

        tr_acc = accuracy_score(y_tr, tr_pred)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        elapsed = time.time() - t0

        pct = int(n_trees / 150 * 10)
        bar = "█" * pct + "░" * (10 - pct)
        print(f"  [{bar}] {n_trees:3d} trees │ {tr_acc*100:9.2f}% │ {val_acc*100:9.2f}% │ {val_f1:12.4f} │ {elapsed:7.2f}s")
        sys.stdout.flush()
        time.sleep(0.05)

    te_pred = rf.predict(X_te)
    te_acc = accuracy_score(y_te, te_pred)
    te_f1 = f1_score(y_te, te_pred, average="macro")
    te_prec = precision_score(y_te, te_pred, average="macro", zero_division=0)
    te_rec = recall_score(y_te, te_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_te, te_pred)

    print("\n" + "="*65)
    print("  FINAL EVALUATION ON HELD-OUT TEST SET (150 TREES)")
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
