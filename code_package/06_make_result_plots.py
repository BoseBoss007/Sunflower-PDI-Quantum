import json
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

OUT = "/home/claude/pipeline/outputs"
NAVY = "#1E2761"; PURPLE = "#2b2c6b"; GOLD = "#F9E795"; GREY = "#5A5A6E"

with open(f"{OUT}/classical_results.json") as f:
    C = json.load(f)
with open(f"{OUT}/quantum_results.json") as f:
    Q = json.load(f)

# ---------- 1. Confusion matrices (growth stage + PDI), real ----------
def plot_cm(ax, cm, labels, title):
    cm = np.array(cm)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title, fontsize=11, fontweight='bold')
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black', fontsize=10, fontweight='bold')

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
plot_cm(axes[0,0], C['growth_stage']['random_forest']['confusion_matrix'], C['growth_stage']['classes'],
        f"Growth Stage \u2014 Random Forest\nAcc={C['growth_stage']['random_forest']['accuracy']:.3f}  MacroF1={C['growth_stage']['random_forest']['macro_f1']:.3f}")
plot_cm(axes[0,1], C['growth_stage']['svm']['confusion_matrix'], C['growth_stage']['classes'],
        f"Growth Stage \u2014 SVM (RBF)\nAcc={C['growth_stage']['svm']['accuracy']:.3f}  MacroF1={C['growth_stage']['svm']['macro_f1']:.3f}")
plot_cm(axes[1,0], C['pdi']['random_forest']['confusion_matrix'], C['pdi']['classes'],
        f"PDI Proxy \u2014 Random Forest\nAcc={C['pdi']['random_forest']['accuracy']:.3f}  MacroF1={C['pdi']['random_forest']['macro_f1']:.3f}")
plot_cm(axes[1,1], C['pdi']['svm']['confusion_matrix'], C['pdi']['classes'],
        f"PDI Proxy \u2014 SVM (RBF)\nAcc={C['pdi']['svm']['accuracy']:.3f}  MacroF1={C['pdi']['svm']['macro_f1']:.3f}")
plt.suptitle("Real Confusion Matrices \u2014 Classical Baselines (test set, n=225 each task)", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(f"{OUT}/03_confusion_matrices_classical.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 03_confusion_matrices_classical.png")

# ---------- 2. Classical model comparison bar chart ----------
fig, ax = plt.subplots(figsize=(9, 5.5))
tasks = ['Growth Stage\n(5-class)', 'PDI Proxy\n(3-class)']
rf_acc = [C['growth_stage']['random_forest']['accuracy'], C['pdi']['random_forest']['accuracy']]
svm_acc = [C['growth_stage']['svm']['accuracy'], C['pdi']['svm']['accuracy']]
x = np.arange(len(tasks)); w = 0.32
b1 = ax.bar(x - w/2, rf_acc, w, label='Random Forest', color=NAVY)
b2 = ax.bar(x + w/2, svm_acc, w, label='SVM (RBF)', color=GOLD, edgecolor=NAVY)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{b.get_height():.3f}", ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(tasks, fontsize=11)
ax.set_ylabel('Test Accuracy'); ax.set_ylim(0, 1.05)
ax.set_title('Real Classical Baseline Accuracy (test set, real dataset, n=1,500 sample)', fontsize=12, fontweight='bold')
ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUT}/04_classical_accuracy_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 04_classical_accuracy_comparison.png")

# ---------- 3. VQC training curve (real) ----------
h = Q['vqc_history']
fig, ax1 = plt.subplots(figsize=(9, 5.5))
ax1.plot(h['epoch'], h['train_loss'], color=NAVY, marker='o', label='Train BCE loss')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Train loss (BCE)', color=NAVY)
ax1.tick_params(axis='y', labelcolor=NAVY)
ax2 = ax1.twinx()
ax2.plot(h['epoch'], h['val_acc'], color='#C97A2B', marker='s', label='Val accuracy')
ax2.set_ylabel('Validation accuracy', color='#C97A2B')
ax2.tick_params(axis='y', labelcolor='#C97A2B')
ax2.set_ylim(0, 1.05)
ax1.set_title(f"Real VQC Training Curve \u2014 {Q['n_qubits']} qubits, {Q['n_layers']} layers, PennyLane default.qubit\n(n_train={Q['n_train']}, n_val={Q['n_val']})", fontsize=12, fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='center right')
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/05_vqc_training_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 05_vqc_training_curve.png")

# ---------- 4. Quantum vs classical comparison (SAME binary task, real) ----------
fig, ax = plt.subplots(figsize=(8.5, 5.5))
labels = ['VQC\n(6 qubits, real)', 'SVM on same\n6-D PCA features', 'SVM on full\n34-D features']
accs = [Q['vqc_test_accuracy'], Q['svm_pca6d_test_accuracy'], Q['svm_full_test_accuracy']]
f1s = [Q['vqc_test_macro_f1'], Q['svm_pca6d_test_macro_f1'], Q['svm_full_test_macro_f1']]
x = np.arange(len(labels)); w = 0.35
b1 = ax.bar(x - w/2, accs, w, label='Accuracy', color=PURPLE)
b2 = ax.bar(x + w/2, f1s, w, label='Macro-F1', color=GOLD, edgecolor=NAVY)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{b.get_height():.3f}", ha='center', fontsize=9.5, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 1.08)
ax.set_title(f"Real Test-Set Comparison \u2014 Binary PDI Task (Low vs High)\nn_test={Q['n_test']}, balanced, same train/test split for all three models", fontsize=11.5, fontweight='bold')
ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUT}/06_quantum_vs_classical.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 06_quantum_vs_classical.png")

# ---------- 5. Noise robustness (real) ----------
nr = Q['noise_robustness']
ps = [r['p'] for r in nr]; accs_n = [r['accuracy'] for r in nr]
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.plot(ps, accs_n, color=NAVY, marker='o', linewidth=2.2, markersize=7)
ax.axhline(0.5, color='grey', linestyle='--', linewidth=1, label='Chance level (50%)')
ax.set_xlabel('Depolarizing noise probability (per gate)')
ax.set_ylabel(f"VQC test accuracy (n={nr[0]['n_eval']} samples)")
ax.set_title('Real Noise-Robustness Sweep \u2014 PennyLane default.mixed density-matrix simulation', fontsize=12, fontweight='bold')
ax.set_ylim(0.3, 1.0)
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/07_noise_robustness.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 07_noise_robustness.png")

# ---------- 6. Ablation: dimensionality/qubits (real, classical proxy) ----------
ab = Q['ablation_qubits']
nqs = [a['n_components'] for a in ab]; accs_a = [a['classical_svm_acc_same_dims'] for a in ab]; var_a = [a['explained_var'] for a in ab]
fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
ax1.bar([str(n) for n in nqs], accs_a, color=NAVY, alpha=0.85, label='Classifier accuracy')
ax1.set_xlabel('Number of PCA components (= qubits used for angle encoding)')
ax1.set_ylabel('Test accuracy', color=NAVY)
ax1.set_ylim(0, 1.05)
for i, v in enumerate(accs_a):
    ax1.text(i, v+0.02, f"{v:.3f}", ha='center', fontweight='bold')
ax2 = ax1.twinx()
ax2.plot(range(len(nqs)), var_a, color='#C97A2B', marker='D', linewidth=2, label='PCA explained variance')
ax2.set_ylabel('Cumulative explained variance', color='#C97A2B')
ax2.set_ylim(0, 1.05)
ax1.set_title('Real Ablation \u2014 Effect of Encoding Dimensionality (PCA components / qubits)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/08_ablation_dimensionality.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved 08_ablation_dimensionality.png")

print("\nAll real result visualizations generated.")
