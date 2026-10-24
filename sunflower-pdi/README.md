# Sunflower PDI Quantum: Production Package (`sunflower-pdi`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PennyLane](https://img.shields.io/badge/quantum-PennyLane-9cf.svg)](https://pennylane.ai/)
[![OpenCV](https://img.shields.io/badge/vision-OpenCV-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

## Mission & Architecture Overview
This package provides a modular, reproducible implementation for computing a Pollen Deficiency Index (PDI) proxy for sunflowers using Digital Image Processing (DIP), 5 classical machine learning baselines, and dual Variational Quantum Circuit (VQC) architectures:
1. **Method 1:** PCA Dimensionality Reduction ($31\text{D} \to 6\text{D}$) + Quantum Angle Encoding ($81.82\%$ test accuracy).
2. **Method 2:** Zero-Dimensionality-Reduction Quantum Amplitude Encoding ($100\%$ botanical variance preserved into $\mathbb{C}^{64}$ Hilbert space, $84.85\%$ test accuracy).

---

## Dataset
- **5,351 real ground-level sunflower images** across 5 developmental stages: `YoungBud`, `MatureBud`, `EarlyBloom`, `Healthy`, `Wilted`.
- Deterministic stratified sampling (seed 42): 1,500 images for Phase 1 feature extraction (300 per class), and a balanced 220-image subset for Phase 2 binary classification (Low vs. High deficiency).

---

## Phase 1 Pipeline: Classical Vision & Digital Image Processing (DIP)
1. **Preprocessing:** RGB to CIE-$L^*a^*b^*$ conversion, CLAHE contrast enhancement on the $L^*$ channel, and Bilateral edge-preserving denoising. Resized to $200 \times 200$.
2. **Dual-Stage Segmentation:** GrabCut to extract the foreground blossom, followed by morphological opening/closing. $K$-means ($k=3$) clustering extracts the central floret disc (darkest cluster).
3. **Botanical Feature Extraction:** 31 non-circular handcrafted botanical features:
   - 3 Morphological Shape Descriptors (flower area, perimeter, solidity)
   - 2 Disc Geometry Descriptors (disc area, disc eccentricity)
   - 22 Color Histogram Descriptors (normalized across H, S, V channels)
   - 4 GLCM Texture Descriptors (contrast, dissimilarity, homogeneity, energy)
4. **PDI Formulation:** Rule-based transparent proxy categorizing each image into Low, Medium, or High pollen deficiency.
5. **Classical Baseline Models:** Logistic Regression, SVM (RBF, $C=5$), Random Forest (150 trees), XGBoost, and MLP Neural Net (64-32).

---

## Phase 2 Pipeline: Dual Quantum Architectures
1. **Method 1: Angle-Encoded VQC (PCA 6-D):**
   - Features standardized and projected to 6 principal components via PCA ($75.96\%$ variance).
   - Encoded onto 6 qubits using single-qubit $RY$ rotations: $|\psi\rangle = \bigotimes_{i=0}^5 RY(\theta_i)|0\rangle$.
   - 3-layer strongly entangling ansatz ($RY$, $RZ$, circular CNOT ring).
   - Pauli-$Z$ measurement on qubit 0 with classical sigmoid head ($81.82\%$ accuracy, $0.9338$ ROC-AUC).
2. **Method 2: Amplitude-Encoded VQC (Hilbert 64-D, Zero PCA):**
   - Zero PCA dimensionality reduction ($100\%$ feature variance preserved).
   - All 31 features zero-padded to 64 dimensions ($2^6$) and normalized under Euclidean $L_2$ norm.
   - Embedded directly as quantum state amplitudes: $|\psi\rangle = \sum_{i=0}^{30} c_i |i\rangle$ in $\mathcal{H} \cong \mathbb{C}^{64}$.
   - Identical 3-layer ansatz processing the superposed state ($84.85\%$ accuracy, $0.9522$ ROC-AUC).
3. **Quantum Noise Simulation:** Depolarizing noise channels modeled via PennyLane `default.mixed` density matrix simulator ($p \in [0.0, 0.20]$).

---

## Performance Summary

| Architecture | Model Family | Input Dim | Test Acc | F1 Macro | ROC-AUC | Parameters |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | Classical Linear | 31-D | **93.94%** | 0.939 | - | 32 |
| **SVM (RBF Kernel)** | Classical Kernel | 31-D | **90.91%** | 0.909 | - | 1,921 |
| **Random Forest** | Classical Ensemble | 31-D | **90.91%** | 0.909 | - | 3,328 |
| **XGBoost Trees** | Classical Boosting | 31-D | **93.94%** | 0.939 | - | 682 |
| **Neural Net (MLP)** | Classical Neural | 31-D | **93.94%** | 0.939 | - | 4,161 |
| **Method 1: Angle VQC** | Quantum Variational | 6-D PCA | **81.82%** | 0.818 | 0.9338 | **38** |
| **Method 2: Amplitude VQC**| Quantum Variational | **31-D Full** | **84.85%** | **0.849** | **0.9522** | **38** |

---

## Directory Organization

```
sunflower-pdi/
├── configs/            # YAML configs (base.yaml, phase1.yaml, phase2.yaml)
├── src/                # Core modules: preprocessing, classical, quantum, utils
├── scripts/
│   ├── train/          # Standalone live training scripts (1 to 7)
│   ├── compare_methods.py          # Method 1 vs Method 2 automated comparison
│   ├── generate_paper_figures.py   # Publication Figs 1-11 generator
│   └── run_all.py                  # End-to-end pipeline orchestrator
├── artifacts/
│   ├── figures/publication/        # 11 IEEE Access figures (PDF/PNG)
│   └── metrics/                    # JSON/CSV evaluation metrics
├── models/             # Serialized classical and quantum model weights
├── docs/               # Technical and comparative analysis reports
└── tests/              # Pytest automated test suite
```

---

## Quickstart & Execution

```bash
# 1. Activate environment
source ../venv/bin/activate  # or local venv
pip install -r requirements.txt

# 2. Run standalone training scripts
python scripts/train/1_train_logistic_regression.py
python scripts/train/2_train_svm.py
python scripts/train/3_train_random_forest.py
python scripts/train/4_train_xgboost.py
python scripts/train/5_train_neural_net.py
python scripts/train/6_train_quantum_vqc.py       # Method 1
python scripts/train/7_train_amplitude_vqc.py     # Method 2

# 3. Head-to-Head Quantum Comparison
PYTHONPATH=. python scripts/compare_methods.py

# 4. Generate all IEEE publication figures
PYTHONPATH=. python scripts/generate_paper_figures.py
```
