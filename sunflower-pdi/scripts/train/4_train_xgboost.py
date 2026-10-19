"""
4. Baseline Model: XGBoost Gradient Boosting
============================================
3-Class PDI Classification (Low / Medium / High)
Live boosting rounds (epochs) with training & validation multi-logloss tracking.
"""

import os, sys, time
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def main():
    print("\n" + "="*65)
    print("  BASELINE MODEL 4: XGBOOST GRADIENT BOOSTING")
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
    print(f"  Boosting Rounds   : 60 Iterations / Epochs")
    print(f"  Max Tree Depth    : 4")
    print(f"  Learning Rate (η) : 0.08")
    print(f"  Subsample Ratio   : 0.85")

    xgb = XGBClassifier(
        n_estimators=60,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss"
    )

    print("\n" + "-"*65)
    print(f"  {'Round (Epoch)':>14} │ {'Train Loss':>12} │ {'Val Loss':>10} │ {'Status':>12}")
    print("-" * 65)

    # Train with evaluation tracking
    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=False
    )

    eval_results = xgb.evals_result()
    tr_losses = eval_results["validation_0"]["mlogloss"]
    val_losses = eval_results["validation_1"]["mlogloss"]

    best_val = min(val_losses)
    for r in range(len(tr_losses)):
        tr_l = tr_losses[r]
        val_l = val_losses[r]
        is_best = (val_l == best_val)
        status = "★ Lowest Loss" if is_best else ""
        print(f"  Iteration {r+1:2d}/60 │ {tr_l:12.4f} │ {val_l:10.4f} │ {status:>12}")
        sys.stdout.flush()
        time.sleep(0.03)

    te_pred = xgb.predict(X_te)
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
