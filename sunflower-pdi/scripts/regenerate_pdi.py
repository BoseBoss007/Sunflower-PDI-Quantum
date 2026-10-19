"""
Regenerate all PDI labels, models, and metrics using the redesigned
continuous PDI scoring formula.

Run from the sunflower-pdi/ directory:
    source venv/bin/activate
    PYTHONPATH=. python scripts/regenerate_pdi.py
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)
import joblib
import sys

sys.path.insert(0, ".")
from src.pdi.rules import pdi_proxy_label, pdi_score

SEED = 42
np.random.seed(SEED)

FEAT_CSV  = "artifacts/features/phase1_features.csv"
OUT_FEAT  = "artifacts/features/phase1_features.csv"        # overwrite in-place
OUT_PHASE2 = "artifacts/features/phase2_pca_features.csv"
OUT_METRICS = "artifacts/metrics"
OUT_MODELS  = "models"

# ─────────────────────────────────────────────
# 1. Load existing extracted features
# ─────────────────────────────────────────────
print("Loading phase1_features.csv …")
df = pd.read_csv(FEAT_CSV)
print(f"  Loaded {len(df)} rows, {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. Recompute PDI labels with new formula
# ─────────────────────────────────────────────
print("\nRecomputing PDI labels (continuous scoring) …")
df["pdi_label"] = df.apply(
    lambda r: pdi_proxy_label(
        r["disc_ratio"], r["sat_mean"], r["val_mean"], r["circularity"]
    ), axis=1
)

print("  New label distribution:")
print(df["pdi_label"].value_counts())
print("\n  Label × class cross-tab:")
print(pd.crosstab(df["class"], df["pdi_label"]))

# Save updated features
df.to_csv(OUT_FEAT, index=False)
print(f"\n  Saved → {OUT_FEAT}")

# ─────────────────────────────────────────────
# 3. Phase 1 — Classical model training (full 35-D)
# ─────────────────────────────────────────────
feat_cols = [c for c in df.columns
             if c not in ("path", "class", "pdi_label")]
X = np.nan_to_num(df[feat_cols].values)
results = {}

def run_task(X, y_raw, task_name, class_order, tag):
    le = LabelEncoder()
    le.fit(class_order)
    y = le.transform(y_raw)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)

    sc = StandardScaler().fit(X_tr)
    X_tr_s  = sc.transform(X_tr)
    X_val_s = sc.transform(X_val)
    X_te_s  = sc.transform(X_te)

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"  Train/Val/Test: {len(y_tr)}/{len(y_val)}/{len(y_te)}")

    res = {"n_train": len(y_tr), "n_val": len(y_val), "n_test": len(y_te),
           "classes": list(class_order)}

    for name, clf in [
        ("random_forest", RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1)),
        ("svm", SVC(kernel="rbf", C=5.0, gamma="scale", random_state=SEED)),
    ]:
        clf.fit(X_tr_s, y_tr)
        tr_acc = accuracy_score(y_tr,  clf.predict(X_tr_s))
        te_acc = accuracy_score(y_te,  clf.predict(X_te_s))
        te_f1  = f1_score(y_te, clf.predict(X_te_s), average="macro")
        te_cm  = confusion_matrix(y_te, clf.predict(X_te_s)).tolist()

        print(f"\n  {name}:")
        print(f"    Train Accuracy : {tr_acc*100:.2f}%")
        print(f"    Test  Accuracy : {te_acc*100:.2f}%")
        print(f"    Test  Macro-F1 : {te_f1:.4f}")
        print(f"    Overfit gap    : {(tr_acc - te_acc)*100:.2f}%")
        print(classification_report(y_te, clf.predict(X_te_s),
                                    target_names=class_order, zero_division=0))

        res[name] = {"train_accuracy": tr_acc, "accuracy": te_acc,
                     "macro_f1": te_f1, "confusion_matrix": te_cm}
        joblib.dump(clf, os.path.join(OUT_MODELS, "classical",
                                      f"{name}_{tag}.joblib"))
    return res

# Task A: 5-class stage
results["growth_stage"] = run_task(
    X, df["class"].values, "5-class Growth Stage",
    ["YoungBud", "MatureBud", "EarlyBloom", "Healthy", "Wilted"], "stage")

# Task B: 3-class PDI
results["pdi"] = run_task(
    X, df["pdi_label"].values, "3-class PDI Proxy",
    ["Low", "Medium", "High"], "pdi")

os.makedirs(OUT_METRICS, exist_ok=True)
with open(os.path.join(OUT_METRICS, "classical_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved classical_results.json")

# ─────────────────────────────────────────────
# 4. Phase 2 — 5-fold CV on binary PDI (Low vs High)
# ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("Phase 2: Binary PDI (Low vs High) — 5-fold Cross-Validation")

sub = df[df["pdi_label"].isin(["Low", "High"])].copy()
n_per = min(sub["pdi_label"].value_counts())
balanced = pd.concat([
    sub[sub["pdi_label"] == lbl].sample(n=min(n_per, 110), random_state=SEED)
    for lbl in ["Low", "High"]
]).reset_index(drop=True)
print(f"  Binary subset: {len(balanced)} samples  "
      f"({balanced['pdi_label'].value_counts().to_dict()})")

X_b = np.nan_to_num(balanced[feat_cols].values)
y_b = (balanced["pdi_label"] == "High").astype(int).values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

cv_results = {}
for clf_name, clf in [
    ("random_forest", RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1)),
    ("svm", SVC(kernel="rbf", C=5.0, gamma="scale", random_state=SEED)),
    ("svm_6d_pca", Pipeline([
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=6, random_state=SEED)),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)),
    ])),
]:
    if clf_name == "svm_6d_pca":
        pipe = clf
    else:
        pipe = Pipeline([("sc", StandardScaler()), ("clf", clf)])

    acc_scores = cross_val_score(pipe, X_b, y_b, cv=skf, scoring="accuracy")
    f1_scores  = cross_val_score(pipe, X_b, y_b, cv=skf, scoring="f1_macro")
    print(f"\n  {clf_name}:")
    print(f"    5-fold Acc : {acc_scores.mean()*100:.2f}% ± {acc_scores.std()*100:.2f}%")
    print(f"    5-fold F1  : {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
    print(f"    Per-fold   : {[round(s*100,1) for s in acc_scores]}")
    cv_results[clf_name] = {
        "cv_accuracy_mean": float(acc_scores.mean()),
        "cv_accuracy_std":  float(acc_scores.std()),
        "cv_f1_mean":       float(f1_scores.mean()),
        "cv_f1_std":        float(f1_scores.std()),
        "per_fold_acc":     acc_scores.tolist(),
    }

with open(os.path.join(OUT_METRICS, "phase2_cv_results.json"), "w") as f:
    json.dump(cv_results, f, indent=2)
print(f"\nSaved phase2_cv_results.json")

# ─────────────────────────────────────────────
# 5. Rebuild phase2_pca_features.csv with new labels
# ─────────────────────────────────────────────
print("\nRebuilding phase2_pca_features.csv …")
sub_all = df[df["pdi_label"].isin(["Low", "High"])].copy()
X_sub = np.nan_to_num(sub_all[feat_cols].values)
y_sub = (sub_all["pdi_label"] == "High").astype(int).values

X_tr2, X_te2, y_tr2, y_te2, idx_tr, idx_te = \
    train_test_split(X_sub, y_sub, sub_all.index.values,
                     test_size=0.30, random_state=SEED, stratify=y_sub)

sc2  = StandardScaler().fit(X_tr2)
pca2 = PCA(n_components=6, random_state=SEED).fit(sc2.transform(X_tr2))
joblib.dump(sc2,  os.path.join(OUT_MODELS, "quantum", "scaler.joblib"))
joblib.dump(pca2, os.path.join(OUT_MODELS, "quantum", "pca.joblib"))

def make_pca_df(X_raw, indices, split_name):
    Xp = pca2.transform(sc2.transform(X_raw))
    d = pd.DataFrame(Xp, columns=[f"pca_{i}" for i in range(6)])
    d["path"]      = sub_all.loc[indices, "path"].values
    d["pdi_label"] = sub_all.loc[indices, "pdi_label"].values
    d["split"]     = split_name
    return d

pca_df = pd.concat([make_pca_df(X_tr2, idx_tr, "train"),
                    make_pca_df(X_te2, idx_te, "test")], ignore_index=True)
pca_df.to_csv(OUT_PHASE2, index=False)
print(f"  Saved → {OUT_PHASE2}")
print(f"  PCA explained variance: {pca2.explained_variance_ratio_.sum():.4f}")

print("\n✅  All done — PDI labels, models and metrics regenerated.")
