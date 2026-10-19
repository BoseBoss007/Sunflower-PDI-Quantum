import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Quantum VQC (6 Qubits)', 'Classical SVM (RBF Kernel)']
accuracy = [96.97, 100.00]

plt.figure(figsize=(8, 6))

# Create bar plot
bars = plt.bar(models, accuracy, color=['#9b59b6', '#34495e'], width=0.5)

# Add title and labels
plt.title('Phase 2 Evaluation: Quantum vs Classical Baseline', fontsize=14, pad=20)
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.ylim(85, 102)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('artifacts/figures/model_comparison.png', dpi=300)
print("Saved comparison chart")
