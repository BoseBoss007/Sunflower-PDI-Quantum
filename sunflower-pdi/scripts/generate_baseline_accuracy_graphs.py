"""
Publication-Quality Accuracy Graphs for All Baseline Models
=============================================================
Target: IEEE Access / Q1 Botanical/AI Research Papers.
Empirical Validation: 100% derived from real feature extraction dataset
(artifacts/features/phase1_features.csv). Zero synthetic or fabricated data.

Models Evaluated Separately:
  1. Logistic Regression (Online SGD Solver - 40 epochs)
  2. Support Vector Machine (RBF Kernel - 10-Fold CV & C Regularization Path)
  3. Random Forest Ensemble (10 to 150 Decision Trees)
  4. XGBoost Gradient Boosting (60 Boosting Iterations)
  5. Multi-Layer Perceptron (Neural Network - 40 Backpropagation Epochs)
  6. Multi-Model Baseline Synthesis (Train vs Test Accuracy & Overfitting Gap)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

# Ensure paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from src.pdi.rules import pdi_proxy_label

# Publication Plotting Configuration (Strict IEEE Access White Background Standards)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#334155",
    "axes.linewidth": 1.0,
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "savefig.facecolor": "#ffffff",
    "savefig.edgecolor": "#ffffff",
    "savefig.dpi": 300
})

OUT_DIR = "artifacts/figures/publication"
METRICS_DIR = "artifacts/metrics"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def load_and_preprocess_data():
    """Load empirical features and perform leakage-free stratified split."""
    df = pd.read_csv("artifacts/features/phase1_features.csv")
    df["pdi_label"] = [
        pdi_proxy_label(d, s, v, c)
        for d, s, v, c in zip(df["disc_ratio"], df["sat_mean"], df["val_mean"], df["circularity"])
    ]

    # Exclude score-defining features to avoid circular reasoning
    SCORE_FEATURES = {"disc_ratio", "sat_mean", "val_mean", "circularity"}
    EXCLUDE = {"path", "class", "pdi_score", "pdi_label"} | SCORE_FEATURES
    feat_cols = [c for c in df.columns if c not in EXCLUDE]

    X = np.nan_to_num(df[feat_cols].values)
    label_map = {"Low": 0, "Medium": 1, "High": 2}
    y = df["pdi_label"].map(label_map).values

    # Stratified Train (70%), Val (15%), Test (15%)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    return (X, y, feat_cols, X_tr_s, X_val_s, X_te_s, y_tr, y_val, y_te)


# =====================================================================
# 1. BASELINE 1: LOGISTIC REGRESSION (ONLINE SGD)
# =====================================================================
def run_and_plot_logistic_regression(X_tr, X_val, X_te, y_tr, y_val, y_te):
    print("\n" + "="*70)
    print("  [1/5] RUNNING REAL TRAINING: LOGISTIC REGRESSION (SGD)")
    print("="*70)

    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=1,
        learning_rate="optimal",
        random_state=42,
        warm_start=True
    )
    classes = np.array([0, 1, 2])
    EPOCHS = 40

    epochs = []
    tr_accs, val_accs, te_accs = [], [], []
    tr_losses, val_losses = [], []

    for ep in range(1, EPOCHS + 1):
        clf.partial_fit(X_tr, y_tr, classes=classes)

        tr_pred = clf.predict(X_tr)
        val_pred = clf.predict(X_val)
        te_pred = clf.predict(X_te)

        tr_prob = np.clip(clf.predict_proba(X_tr), 1e-15, 1 - 1e-15)
        val_prob = np.clip(clf.predict_proba(X_val), 1e-15, 1 - 1e-15)

        tr_loss = log_loss(y_tr, tr_prob, labels=classes)
        val_loss = log_loss(y_val, val_prob, labels=classes)

        epochs.append(ep)
        tr_accs.append(accuracy_score(y_tr, tr_pred) * 100.0)
        val_accs.append(accuracy_score(y_val, val_pred) * 100.0)
        te_accs.append(accuracy_score(y_te, te_pred) * 100.0)
        tr_losses.append(tr_loss)
        val_losses.append(val_loss)

        if ep % 5 == 0 or ep == EPOCHS:
            print(f"  Epoch {ep:2d}/{EPOCHS} │ Train Acc: {tr_accs[-1]:5.1f}% │ Val Acc: {val_accs[-1]:5.1f}% │ Test Acc: {te_accs[-1]:5.1f}% │ Loss: {tr_loss:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Accuracy Curves
    ax1.plot(epochs, tr_accs, label="Train Accuracy", color="#1e3a8a", linewidth=2.2, marker="o", markersize=3.5)
    ax1.plot(epochs, val_accs, label="Validation Accuracy", color="#059669", linewidth=2.0, linestyle="--", marker="s", markersize=3.5)
    ax1.plot(epochs, te_accs, label="Test Accuracy", color="#dc2626", linewidth=2.0, linestyle="-.", marker="^", markersize=3.5)
    ax1.axhline(te_accs[-1], color="#dc2626", linestyle=":", alpha=0.6, label=f"Final Test: {te_accs[-1]:.1f}%")

    ax1.set_title("(a) Logistic Regression Accuracy Progression", fontweight="bold", pad=10)
    ax1.set_xlabel("SGD Epoch")
    ax1.set_ylabel("Classification Accuracy (%)")
    ax1.set_ylim(40, 100)
    ax1.grid(True)
    ax1.legend(loc="lower right", framealpha=0.95)

    # Panel B: Loss Curves
    ax2.plot(epochs, tr_losses, label="Train Log-Loss", color="#1e3a8a", linewidth=2.0)
    ax2.plot(epochs, val_losses, label="Validation Log-Loss", color="#d97706", linewidth=2.0, linestyle="--")
    best_epoch = int(np.argmin(val_losses) + 1)
    ax2.scatter(best_epoch, min(val_losses), color="#dc2626", s=70, zorder=5, label=f"Min Val Loss (Ep {best_epoch})")

    ax2.set_title("(b) Empirical Cross-Entropy Loss Convergence", fontweight="bold", pad=10)
    ax2.set_xlabel("SGD Epoch")
    ax2.set_ylabel("Multi-class Cross-Entropy Loss")
    ax2.grid(True)
    ax2.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    fig_png = f"{OUT_DIR}/baseline_logistic_regression_accuracy.png"
    fig_pdf = f"{OUT_DIR}/baseline_logistic_regression_accuracy.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    print(f"  [Saved] {fig_png} and .pdf")

    return {
        "epochs": epochs,
        "train_acc": tr_accs,
        "val_acc": val_accs,
        "test_acc": te_accs,
        "train_loss": tr_losses,
        "val_loss": val_losses,
        "final_train_acc": tr_accs[-1],
        "final_test_acc": te_accs[-1]
    }


# =====================================================================
# 2. BASELINE 2: SUPPORT VECTOR MACHINE (RBF KERNEL)
# =====================================================================
def run_and_plot_svm(X, y):
    print("\n" + "="*70)
    print("  [2/5] RUNNING REAL TRAINING: SUPPORT VECTOR MACHINE (RBF)")
    print("="*70)

    # 1. 10-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_train_accs = []
    fold_test_accs = []
    folds = list(range(1, 11))

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        svm = SVC(C=5.0, kernel="rbf", gamma="scale", random_state=42)
        svm.fit(X_tr_s, y_tr)

        tr_acc = accuracy_score(y_tr, svm.predict(X_tr_s)) * 100.0
        te_acc = accuracy_score(y_te, svm.predict(X_te_s)) * 100.0

        fold_train_accs.append(tr_acc)
        fold_test_accs.append(te_acc)
        print(f"  Fold {fold:2d}/10 │ Train Acc: {tr_acc:5.1f}% │ Test Acc: {te_acc:5.1f}%")

    # 2. Regularization Parameter (C) Sensitivity Path
    C_values = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]
    c_tr_means, c_tr_stds = [], []
    c_te_means, c_te_stds = [], []

    print("\n  Evaluating Regularization Parameter (C) Sensitivity Path across 10 folds:")
    for c_val in C_values:
        cur_tr, cur_te = [], []
        for tr_idx, te_idx in skf.split(X, y):
            X_tr_s = StandardScaler().fit_transform(X[tr_idx])
            X_te_s = StandardScaler().fit(X[tr_idx]).transform(X[te_idx])
            s = SVC(C=c_val, kernel="rbf", gamma="scale", random_state=42)
            s.fit(X_tr_s, y[tr_idx])
            cur_tr.append(accuracy_score(y[tr_idx], s.predict(X_tr_s)) * 100.0)
            cur_te.append(accuracy_score(y[te_idx], s.predict(X_te_s)) * 100.0)
        c_tr_means.append(np.mean(cur_tr))
        c_tr_stds.append(np.std(cur_tr))
        c_te_means.append(np.mean(cur_te))
        c_te_stds.append(np.std(cur_te))
        print(f"    C={c_val:5.2f} │ Mean Train: {c_tr_means[-1]:5.1f}% ± {c_tr_stds[-1]:4.1f}% │ Mean Test: {c_te_means[-1]:5.1f}% ± {c_te_stds[-1]:4.1f}%")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Fold-by-Fold Stability
    x_indices = np.arange(len(folds))
    width = 0.38
    ax1.bar(x_indices - width/2, fold_train_accs, width, label="Train Accuracy", color="#1e3a8a", alpha=0.9, edgecolor="#0f172a")
    ax1.bar(x_indices + width/2, fold_test_accs, width, label="Test Accuracy (Holdout)", color="#059669", alpha=0.9, edgecolor="#064e3b")

    mean_te = np.mean(fold_test_accs)
    std_te = np.std(fold_test_accs)
    ax1.axhline(mean_te, color="#dc2626", linestyle="--", linewidth=1.8, label=f"Mean Test: {mean_te:.1f}% ± {std_te:.1f}%")

    ax1.set_title("(a) SVM 10-Fold Stratified Cross-Validation Accuracy", fontweight="bold", pad=10)
    ax1.set_xlabel("Cross-Validation Fold Index")
    ax1.set_ylabel("Classification Accuracy (%)")
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f"F{i}" for i in folds])
    ax1.set_ylim(60, 105)
    ax1.grid(True, axis="y")
    ax1.legend(loc="lower right", framealpha=0.95)

    # Panel B: C Regularization Sensitivity Curve
    c_tr_means = np.array(c_tr_means)
    c_tr_stds = np.array(c_tr_stds)
    c_te_means = np.array(c_te_means)
    c_te_stds = np.array(c_te_stds)

    ax2.plot(C_values, c_tr_means, marker="o", color="#1e3a8a", linewidth=2.0, label="Train Acc (Mean)")
    ax2.fill_between(C_values, c_tr_means - c_tr_stds, c_tr_means + c_tr_stds, color="#1e3a8a", alpha=0.15)

    ax2.plot(C_values, c_te_means, marker="s", color="#059669", linewidth=2.2, label="Test Acc (Mean 10-Fold)")
    ax2.fill_between(C_values, c_te_means - c_te_stds, c_te_means + c_te_stds, color="#059669", alpha=0.18)

    opt_idx = np.argmax(c_te_means)
    ax2.scatter(C_values[opt_idx], c_te_means[opt_idx], color="#dc2626", s=80, zorder=5,
                label=f"Optimum C={C_values[opt_idx]} ({c_te_means[opt_idx]:.1f}%)")

    ax2.set_xscale("log")
    ax2.set_title("(b) SVM Regularization Parameter (C) Sensitivity", fontweight="bold", pad=10)
    ax2.set_xlabel("Penalty Parameter C (log scale)")
    ax2.set_ylabel("Classification Accuracy (%)")
    ax2.set_ylim(60, 102)
    ax2.grid(True)
    ax2.legend(loc="lower right", framealpha=0.95)

    plt.tight_layout()
    fig_png = f"{OUT_DIR}/baseline_svm_accuracy.png"
    fig_pdf = f"{OUT_DIR}/baseline_svm_accuracy.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    print(f"  [Saved] {fig_png} and .pdf")

    return {
        "folds": folds,
        "fold_train_acc": fold_train_accs,
        "fold_test_acc": fold_test_accs,
        "mean_train_acc": float(np.mean(fold_train_accs)),
        "mean_test_acc": float(mean_te),
        "std_test_acc": float(std_te),
        "C_values": C_values,
        "c_test_means": c_te_means.tolist()
    }


# =====================================================================
# 3. BASELINE 3: RANDOM FOREST ENSEMBLE
# =====================================================================
def run_and_plot_random_forest(X_tr, X_val, X_te, y_tr, y_val, y_te):
    print("\n" + "="*70)
    print("  [3/5] RUNNING REAL TRAINING: RANDOM FOREST ENSEMBLE")
    print("="*70)

    tree_steps = list(range(10, 151, 10))
    tr_accs, val_accs, te_accs = [], [], []

    rf = RandomForestClassifier(
        n_estimators=0,
        max_depth=12,
        min_samples_split=5,
        warm_start=True,
        random_state=42,
        n_jobs=-1
    )

    for n_trees in tree_steps:
        rf.n_estimators = n_trees
        rf.fit(X_tr, y_tr)

        tr_pred = rf.predict(X_tr)
        val_pred = rf.predict(X_val)
        te_pred = rf.predict(X_te)

        tr_acc = accuracy_score(y_tr, tr_pred) * 100.0
        val_acc = accuracy_score(y_val, val_pred) * 100.0
        te_acc = accuracy_score(y_te, te_pred) * 100.0

        tr_accs.append(tr_acc)
        val_accs.append(val_acc)
        te_accs.append(te_acc)
        print(f"  {n_trees:3d} Trees │ Train Acc: {tr_acc:5.1f}% │ Val Acc: {val_acc:5.1f}% │ Test Acc: {te_acc:5.1f}%")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Accuracy Progression vs Tree Count
    ax1.plot(tree_steps, tr_accs, label="Train Accuracy (100% Fit)", color="#1e3a8a", linewidth=2.2, marker="o", markersize=4)
    ax1.plot(tree_steps, val_accs, label="Validation Accuracy", color="#059669", linewidth=2.0, linestyle="--", marker="s", markersize=4)
    ax1.plot(tree_steps, te_accs, label="Test Accuracy", color="#dc2626", linewidth=2.0, linestyle="-.", marker="^", markersize=4)
    ax1.axhline(te_accs[-1], color="#dc2626", linestyle=":", alpha=0.7, label=f"Final Test: {te_accs[-1]:.1f}%")

    ax1.set_title("(a) Random Forest Accuracy vs Number of Trees", fontweight="bold", pad=10)
    ax1.set_xlabel("Number of Decision Trees ($N_{est}$)")
    ax1.set_ylabel("Classification Accuracy (%)")
    ax1.set_ylim(60, 105)
    ax1.grid(True)
    ax1.legend(loc="lower right", framealpha=0.95)

    # Panel B: Generalization Gap (Overfitting Analysis)
    gen_gap = np.array(tr_accs) - np.array(te_accs)
    ax2.plot(tree_steps, gen_gap, color="#7c3aed", linewidth=2.2, marker="d", markersize=4.5, label="Generalization Gap (Train - Test)")
    ax2.fill_between(tree_steps, 0, gen_gap, color="#7c3aed", alpha=0.15)
    ax2.axhline(gen_gap[-1], color="#7c3aed", linestyle="--", label=f"Final Gap: +{gen_gap[-1]:.1f}%")

    ax2.set_title("(b) Empirical Overfitting Gap ($Acc_{train} - Acc_{test}$)", fontweight="bold", pad=10)
    ax2.set_xlabel("Number of Decision Trees ($N_{est}$)")
    ax2.set_ylabel("Generalization Discrepancy (%)")
    ax2.set_ylim(0, 30)
    ax2.grid(True)
    ax2.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    fig_png = f"{OUT_DIR}/baseline_random_forest_accuracy.png"
    fig_pdf = f"{OUT_DIR}/baseline_random_forest_accuracy.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    print(f"  [Saved] {fig_png} and .pdf")

    return {
        "tree_steps": tree_steps,
        "train_acc": tr_accs,
        "val_acc": val_accs,
        "test_acc": te_accs,
        "generalization_gap": gen_gap.tolist(),
        "final_train_acc": tr_accs[-1],
        "final_test_acc": te_accs[-1]
    }


# =====================================================================
# 4. BASELINE 4: XGBOOST GRADIENT BOOSTING
# =====================================================================
def run_and_plot_xgboost(X_tr, X_val, X_te, y_tr, y_val, y_te):
    print("\n" + "="*70)
    print("  [4/5] RUNNING REAL TRAINING: XGBOOST GRADIENT BOOSTING")
    print("="*70)

    ROUNDS = 60
    xgb = XGBClassifier(
        n_estimators=ROUNDS,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss"
    )

    # Train model
    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val), (X_te, y_te)],
        verbose=False
    )

    eval_results = xgb.evals_result()
    tr_loss = eval_results["validation_0"]["mlogloss"]
    val_loss = eval_results["validation_1"]["mlogloss"]
    te_loss = eval_results["validation_2"]["mlogloss"]

    # Compute step-by-step accuracies across iterations
    rounds = list(range(1, ROUNDS + 1))
    tr_accs, val_accs, te_accs = [], [], []

    for r in rounds:
        # Predict using up to round r
        tr_p = xgb.predict(X_tr, iteration_range=(0, r))
        val_p = xgb.predict(X_val, iteration_range=(0, r))
        te_p = xgb.predict(X_te, iteration_range=(0, r))

        tr_accs.append(accuracy_score(y_tr, tr_p) * 100.0)
        val_accs.append(accuracy_score(y_val, val_p) * 100.0)
        te_accs.append(accuracy_score(y_te, te_p) * 100.0)

        if r % 10 == 0 or r == ROUNDS:
            print(f"  Round {r:2d}/{ROUNDS} │ Train Acc: {tr_accs[-1]:5.1f}% │ Val Acc: {val_accs[-1]:5.1f}% │ Test Acc: {te_accs[-1]:5.1f}% │ Val Loss: {val_loss[r-1]:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Accuracy vs Boosting Round
    ax1.plot(rounds, tr_accs, label="Train Accuracy", color="#1e3a8a", linewidth=2.2)
    ax1.plot(rounds, val_accs, label="Validation Accuracy", color="#059669", linewidth=2.0, linestyle="--")
    ax1.plot(rounds, te_accs, label="Test Accuracy", color="#dc2626", linewidth=2.0, linestyle="-.")
    ax1.axhline(te_accs[-1], color="#dc2626", linestyle=":", alpha=0.7, label=f"Final Test: {te_accs[-1]:.1f}%")

    ax1.set_title("(a) XGBoost Accuracy vs Boosting Iterations", fontweight="bold", pad=10)
    ax1.set_xlabel("Boosting Rounds / Iterations")
    ax1.set_ylabel("Classification Accuracy (%)")
    ax1.set_ylim(60, 105)
    ax1.grid(True)
    ax1.legend(loc="lower right", framealpha=0.95)

    # Panel B: Multi-Class Cross-Entropy Loss
    ax2.plot(rounds, tr_loss, label="Train Log-Loss", color="#1e3a8a", linewidth=2.0)
    ax2.plot(rounds, val_loss, label="Validation Log-Loss", color="#d97706", linewidth=2.0, linestyle="--")
    ax2.plot(rounds, te_loss, label="Test Log-Loss", color="#dc2626", linewidth=2.0, linestyle="-.")

    min_val_round = int(np.argmin(val_loss) + 1)
    ax2.scatter(min_val_round, min(val_loss), color="#dc2626", s=70, zorder=5, label=f"Min Val Loss (Round {min_val_round})")

    ax2.set_title("(b) Multi-Class Log-Loss Progression", fontweight="bold", pad=10)
    ax2.set_xlabel("Boosting Rounds / Iterations")
    ax2.set_ylabel("Multi-class Log-Loss")
    ax2.grid(True)
    ax2.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    fig_png = f"{OUT_DIR}/baseline_xgboost_accuracy.png"
    fig_pdf = f"{OUT_DIR}/baseline_xgboost_accuracy.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    print(f"  [Saved] {fig_png} and .pdf")

    return {
        "rounds": rounds,
        "train_acc": tr_accs,
        "val_acc": val_accs,
        "test_acc": te_accs,
        "train_loss": tr_loss,
        "val_loss": val_loss,
        "test_loss": te_loss,
        "final_train_acc": tr_accs[-1],
        "final_test_acc": te_accs[-1]
    }


# =====================================================================
# 5. BASELINE 5: MULTI-LAYER PERCEPTRON (NEURAL NETWORK)
# =====================================================================
def run_and_plot_neural_net(X_tr, X_val, X_te, y_tr, y_val, y_te):
    print("\n" + "="*70)
    print("  [5/5] RUNNING REAL TRAINING: MULTI-LAYER PERCEPTRON (MLP)")
    print("="*70)

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
    epochs = []
    tr_accs, val_accs, te_accs = [], [], []
    losses = []

    for ep in range(1, EPOCHS + 1):
        mlp.partial_fit(X_tr, y_tr, classes=classes)

        tr_p = mlp.predict(X_tr)
        val_p = mlp.predict(X_val)
        te_p = mlp.predict(X_te)

        tr_acc = accuracy_score(y_tr, tr_p) * 100.0
        val_acc = accuracy_score(y_val, val_p) * 100.0
        te_acc = accuracy_score(y_te, te_p) * 100.0

        epochs.append(ep)
        tr_accs.append(tr_acc)
        val_accs.append(val_acc)
        te_accs.append(te_acc)
        losses.append(mlp.loss_)

        if ep % 5 == 0 or ep == EPOCHS:
            print(f"  Epoch {ep:2d}/{EPOCHS} │ Train Acc: {tr_acc:5.1f}% │ Val Acc: {val_acc:5.1f}% │ Test Acc: {te_acc:5.1f}% │ Loss: {mlp.loss_:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Accuracy Convergence
    ax1.plot(epochs, tr_accs, label="Train Accuracy", color="#1e3a8a", linewidth=2.2, marker="o", markersize=3.5)
    ax1.plot(epochs, val_accs, label="Validation Accuracy", color="#059669", linewidth=2.0, linestyle="--", marker="s", markersize=3.5)
    ax1.plot(epochs, te_accs, label="Test Accuracy", color="#dc2626", linewidth=2.0, linestyle="-.", marker="^", markersize=3.5)
    ax1.axhline(te_accs[-1], color="#dc2626", linestyle=":", alpha=0.7, label=f"Final Test: {te_accs[-1]:.1f}%")

    ax1.set_title("(a) Neural Network (MLP 64-32-3) Accuracy Convergence", fontweight="bold", pad=10)
    ax1.set_xlabel("Backpropagation Epoch")
    ax1.set_ylabel("Classification Accuracy (%)")
    ax1.set_ylim(40, 102)
    ax1.grid(True)
    ax1.legend(loc="lower right", framealpha=0.95)

    # Panel B: Training Loss
    ax2.plot(epochs, losses, label="Adam Cross-Entropy Loss", color="#1e3a8a", linewidth=2.2)
    ax2.set_title("(b) Neural Network Objective Function Loss", fontweight="bold", pad=10)
    ax2.set_xlabel("Backpropagation Epoch")
    ax2.set_ylabel("Cross-Entropy Loss (Softmax + L2)")
    ax2.grid(True)
    ax2.legend(loc="upper right", framealpha=0.95)

    plt.tight_layout()
    fig_png = f"{OUT_DIR}/baseline_neural_net_accuracy.png"
    fig_pdf = f"{OUT_DIR}/baseline_neural_net_accuracy.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    print(f"  [Saved] {fig_png} and .pdf")

    return {
        "epochs": epochs,
        "train_acc": tr_accs,
        "val_acc": val_accs,
        "test_acc": te_accs,
        "losses": losses,
        "final_train_acc": tr_accs[-1],
        "final_test_acc": te_accs[-1]
    }


# =====================================================================
# 6. ALL BASELINES SYNTHESIS & GENERALIZATION GAP BENCHMARK
# =====================================================================
def plot_baseline_synthesis(metrics_dict):
    print("\n" + "="*70)
    print("  [6/6] GENERATING MULTI-MODEL BASELINE SYNTHESIS BENCHMARK")
    print("="*70)

    model_names = [
        "Logistic\nRegression",
        "SVM\n(RBF Kernel)",
        "Random\nForest",
        "XGBoost\nTrees",
        "Neural Net\n(MLP 64-32)"
    ]

    train_accs = [
        metrics_dict["lr"]["final_train_acc"],
        metrics_dict["svm"]["mean_train_acc"],
        metrics_dict["rf"]["final_train_acc"],
        metrics_dict["xgb"]["final_train_acc"],
        metrics_dict["mlp"]["final_train_acc"]
    ]

    test_accs = [
        metrics_dict["lr"]["final_test_acc"],
        metrics_dict["svm"]["mean_test_acc"],
        metrics_dict["rf"]["final_test_acc"],
        metrics_dict["xgb"]["final_test_acc"],
        metrics_dict["mlp"]["final_test_acc"]
    ]

    gaps = [tr - te for tr, te in zip(train_accs, test_accs)]

    # Print summary table
    print(f"  {'Model':<25} │ {'Train Acc':>11} │ {'Test Acc':>10} │ {'Generalization Gap':>20}")
    print("  " + "-"*70)
    for name, tr, te, gap in zip(model_names, train_accs, test_accs, gaps):
        clean_name = name.replace('\n', ' ')
        print(f"  {clean_name:<25} │ {tr:10.2f}% │ {te:9.2f}% │ {gap:+19.2f}%")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Side-by-side Train vs Test Accuracy
    x = np.arange(len(model_names))
    width = 0.36

    rects1 = ax1.bar(x - width/2, train_accs, width, label="Train Accuracy", color="#1e3a8a", alpha=0.9, edgecolor="#0f172a")
    rects2 = ax1.bar(x + width/2, test_accs, width, label="Test Accuracy (Holdout)", color="#059669", alpha=0.9, edgecolor="#064e3b")

    # Value labels
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#064e3b")

    ax1.set_title("(a) Baseline Models: Train vs Test Classification Accuracy", fontweight="bold", pad=12)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.set_ylim(40, 112)
    ax1.grid(True, axis="y")
    ax1.legend(loc="upper left", framealpha=0.95)

    # Panel B: Overfitting Discrepancy Gap
    colors = ["#3b82f6", "#10b981", "#ef4444", "#f59e0b", "#8b5cf6"]
    bars = ax2.bar(model_names, gaps, color=colors, width=0.55, edgecolor="#1e293b", linewidth=1.1)

    for bar, gap in zip(bars, gaps):
        h = bar.get_height()
        y_pos = h + 0.6 if h >= 0 else h - 1.2
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos, f"+{gap:.2f}%", ha="center", va="bottom" if h >= 0 else "top",
                 fontsize=9.5, fontweight="bold")

    ax2.axhline(0, color="#64748b", linewidth=1.0)
    ax2.set_title("(b) Empirical Overfitting Discrepancy ($Acc_{train} - Acc_{test}$)", fontweight="bold", pad=12)
    ax2.set_ylabel("Generalization Gap (Percentage Points)")
    ax2.set_ylim(-5, 26)
    ax2.grid(True, axis="y")

    plt.tight_layout()
    fig_png = f"{OUT_DIR}/baseline_all_models_accuracy_comparison.png"
    fig_pdf = f"{OUT_DIR}/baseline_all_models_accuracy_comparison.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()
    print(f"  [Saved] {fig_png} and .pdf")


def main():
    start_time = time.time()
    print("\n" + "="*70)
    print("  IEEE ACCESS BASELINE ACCURACY GRAPHS GENERATOR")
    print("  Dataset Source: artifacts/features/phase1_features.csv (Strict Empirical)")
    print("="*70)

    # 1. Load data
    X, y, feat_cols, X_tr, X_val, X_te, y_tr, y_val, y_te = load_and_preprocess_data()
    print(f"  Dataset Loaded: {len(X)} samples, {len(feat_cols)} features (excluding PDI score features)")
    print(f"  Splits        : Train={len(y_tr)}, Val={len(y_val)}, Test={len(y_te)}")

    # 2. Run each model and generate individual figure
    lr_res = run_and_plot_logistic_regression(X_tr, X_val, X_te, y_tr, y_val, y_te)
    svm_res = run_and_plot_svm(X, y)
    rf_res = run_and_plot_random_forest(X_tr, X_val, X_te, y_tr, y_val, y_te)
    xgb_res = run_and_plot_xgboost(X_tr, X_val, X_te, y_tr, y_val, y_te)
    mlp_res = run_and_plot_neural_net(X_tr, X_val, X_te, y_tr, y_val, y_te)

    # 3. Overall synthesis
    all_metrics = {
        "lr": lr_res,
        "svm": svm_res,
        "rf": rf_res,
        "xgb": xgb_res,
        "mlp": mlp_res
    }
    plot_baseline_synthesis(all_metrics)

    # 4. Save JSON trajectory log
    json_path = f"{METRICS_DIR}/baseline_models_accuracy_trajectories.json"
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  [Audit Log Saved] Complete empirical metrics trajectory stored in: {json_path}")

    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print(f"  [COMPLETED] All 6 Baseline Accuracy Figures Generated in {elapsed:.2f}s")
    print(f"  Output Directory: {OUT_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
