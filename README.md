# Quantum-Enhanced Vision-Based Pollen Deficiency Estimation for Precision Sunflower Pollination

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PennyLane](https://img.shields.io/badge/quantum-PennyLane-9cf.svg)](https://pennylane.ai/)
[![OpenCV](https://img.shields.io/badge/vision-OpenCV-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end reproducible research and production framework for estimating the **Pollen Deficiency Index (PDI)** in sunflowers using classical computer vision and a **Variational Quantum Circuit (VQC)**.

---

## Table of Contents
- [1. Overview & Motivation](#1-overview--motivation)
- [2. System Architecture](#2-system-architecture)
  - [End-to-End Pipeline Workflow](#end-to-end-pipeline-workflow)
  - [Phase 1: Classical Computer Vision Pipeline](#phase-1-classical-computer-vision-pipeline)
  - [Phase 2: Quantum Variational Classifier (VQC)](#phase-2-quantum-variational-classifier-vqc)
- [3. Dataset Architecture](#3-dataset-architecture)
- [4. Key Experimental Results & Benchmarks](#4-key-experimental-results--benchmarks)
  - [Summary of Results](#summary-of-results)
  - [Quantum Noise Robustness](#quantum-noise-robustness)
- [5. Repository Structure](#5-repository-structure)
- [6. Installation & Reproducibility Guide](#6-installation--reproducibility-guide)
  - [Setup Virtual Environment](#setup-virtual-environment)
  - [Execution Option A: Modular Pipeline (`sunflower-pdi`)](#execution-option-a-modular-pipeline-sunflower-pdi)
  - [Execution Option B: Standalone Step-by-Step Scripts (`code_package`)](#execution-option-b-standalone-step-by-step-scripts-code_package)
- [7. Verification & Automated Testing](#7-verification--automated-testing)
- [8. Limitations & Research Roadmap](#8-limitations--research-roadmap)
- [9. Citation & Contact](#9-citation--contact)

---

## 1. Overview & Motivation

In hybrid sunflower seed production, accurate and timely detection of **pollen deficiency** is critical to maximizing yield and seed-set rates. Traditional field sampling methods require destructive, labour-intensive pollen collection and microscopic examination.

This project delivers a non-invasive, vision-based estimation framework that operates directly on ground-level imagery. By combining multi-spectral color space transformations, foreground/disc segmentation, handcrafted morphological/textural descriptors, and a 6-qubit Variational Quantum Circuit (VQC), the pipeline explores both classical machine learning and near-term quantum advantage boundaries.

---

## 2. System Architecture

### End-to-End Pipeline Workflow

```mermaid
graph TD
    subgraph Data Layer ["Dataset (5,351 Images, 5 Classes)"]
        D1[YoungBud] & D2[MatureBud] & D3[EarlyBloom] & D4[Healthy] & D5[Wilted]
    end

    subgraph Phase1 ["Phase 1: Classical Vision & Feature Engineering"]
        P1["Color Space Conversion (Lab) + CLAHE + Bilateral Denoising"]
        P2["Dual Segmentation: GrabCut (Foreground) + K-Means (Disc Extraction)"]
        P3["35 Handcrafted Features (4 Shape + 3 Disc + 24 HSV Hist + 4 GLCM)"]
        P4["Transparent Rule-Based PDI Proxy (Low / Medium / High)"]
        P5["Classical Classification: Random Forest & SVM (RBF)"]
    end

    subgraph Phase2 ["Phase 2: Quantum-Classical Hybrid Classifier"]
        Q1["Stratified Subsampling (Balanced N=220 Binary Task)"]
        Q2["StandardScaler + 6-Component PCA Projection"]
        Q3["6-Qubit State Preparation (RY Angle Encoding)"]
        Q4["3-Layer Strongly Entangling Ansatz (RY, RZ + Ring CNOTs)"]
        Q5["Measurement: Pauli-Z Expectation on Qubit 0"]
        Q6["Classical Sigmoid Head -> Binary Prediction (Low vs. High PDI)"]
        Q7["Noise Robustness Analysis (Depolarizing Noise, default.mixed)"]
    end

    Data Layer --> P1
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> P5
    P3 --> Q1
    Q1 --> Q2
    Q2 --> Q3
    Q3 --> Q4
    Q4 --> Q5
    Q5 --> Q6
    Q4 --> Q7
```

---

### Phase 1: Classical Computer Vision Pipeline

1. **Preprocessing:**
   - Conversion from standard RGB to CIE-$L^*a^*b^*$ color space.
   - Contrast Limited Adaptive Histogram Equalization (**CLAHE**) applied to the $L^*$ (lightness) channel to eliminate field lighting variations.
   - **Bilateral Filtering** to smooth background noise while preserving sharp floral petal and disc boundaries.
   - Standardized resizing to $200 \times 200$ for computationally tractable CPU extraction.

2. **Dual-Stage Segmentation:**
   - **GrabCut Segmentation:** Gaussian Mixture Model (GMM) iteration to isolate the complete sunflower blossom from complex field backgrounds.
   - **K-Means Disc Clustering ($k=3$):** Isolates the central tubular floret disc (darkest cluster) from the surrounding yellow ray florets.
   - Morphological opening and closing filters to remove spurious boundary artifacts.

3. **35-Dimensional Feature Engineering:**
   - **4 Floral Shape Descriptors:** Area, perimeter, circularity ($4\pi \cdot \text{Area} / \text{Perimeter}^2$), and solidity.
   - **3 Disc Descriptors:** Disc area, disc-to-flower area ratio, and disc eccentricity.
   - **24 Color Histogram Descriptors:** 8-bin normalized histograms across Hue, Saturation, and Value channels.
   - **4 Gray-Level Co-occurrence Matrix (GLCM) Texture Descriptors:** Contrast, Dissimilarity, Homogeneity, and Energy at $\theta \in \{0, \pi/4, \pi/2, 3\pi/4\}$.

4. **PDI Proxy Formulation:**
   - Translates visual developmental stage, floret disc opening ratio, and petal saturation into a 3-tier Pollen Deficiency Index proxy (**Low**, **Medium**, **High**).

5. **Classical Baseline Models:**
   - Multi-class classification evaluated across 5 developmental stages and 3 PDI proxy levels using Random Forest (100 estimators) and Support Vector Machines with Radial Basis Function (RBF) kernel.

---

### Phase 2: Quantum Variational Classifier (VQC)

```
                 Variational Quantum Circuit (VQC) Topology
                
|0⟩ ─── RY(x₀) ─── [ Rot(θ₀, ϕ₀, λ₀) ] ───●───────────────╭─── M_Z (⟨Z₀⟩) ───[ Sigmoid Head ] -> P(Low/High)
                                          │               │
|0⟩ ─── RY(x₁) ─── [ Rot(θ₁, ϕ₁, λ₁) ] ───X───●───────────┼─────────────────────────────────────────────
                                              │           │
|0⟩ ─── RY(x₂) ─── [ Rot(θ₂, ϕ₂, λ₂) ] ───────X───●───────┼─────────────────────────────────────────────
                                                  │       │
|0⟩ ─── RY(x₃) ─── [ Rot(θ₃, ϕ₃, λ₃) ] ───────────X───●───┼─────────────────────────────────────────────
                                                      │   │
|0⟩ ─── RY(x₄) ─── [ Rot(θ₄, ϕ₄, λ₄) ] ───────────────X───●─────────────────────────────────────────────
                                                          │
|0⟩ ─── RY(x₅) ─── [ Rot(θ₅, ϕ₅, λ₅) ] ───────────────────X─────────────────────────────────────────────
        [State Prep]        [Layer 1..3: Rotations & Ring Entanglement]
```

1. **Feature Reduction & Normalization:**
   - A balanced 220-image subset is standardized and projected onto its top 6 principal components via Principal Component Analysis (PCA), capturing dominant variance without overfitting.
2. **Quantum Angle Encoding:**
   - The 6 PCA components are mapped directly to 6 qubits via single-qubit Pauli-Y rotation gates: $|\psi_0\rangle = \bigotimes_{i=0}^5 R_y(x_i)|0\rangle$.
3. **Variational Ansatz (3 Layers):**
   - Each layer applies parameterized single-qubit rotations ($R_y, R_z$) on all 6 qubits, followed by a closed ring of CNOT gates between adjacent qubits to induce strong multi-qubit entanglement.
4. **Readout & Hybrid Classification:**
   - Expectation value $\langle Z_0 \rangle$ is measured on qubit 0 and passed to a classical sigmoid activation with learnable bias and scale parameters.
5. **Noise Robustness Verification:**
   - Evaluated under depolarizing noise channels using PennyLane's `default.mixed` density matrix simulator across error probabilities $p \in [0.0, 0.20]$.

---

## 3. Dataset Architecture

The project operates on **5,351 real-world ground-level sunflower images** categorized across 5 developmental stages:

| Stage Directory | Sample Count | Agronomic Significance | PDI Proxy Alignment |
|---|---|---|---|
| `Sunflower Compressed/YoungBud/` | 1,027 | Inflorescence enclosed by involucral bracts | High Deficiency (Pre-anthesis) |
| `Sunflower Compressed/MatureBud/` | 1,010 | Disc opening; ray florets visible | Medium Deficiency |
| `Sunflower Compressed/EarlyBloom/`| 1,080 | Outer disc florets shedding pollen | Low Deficiency (Peak Pollen) |
| `Sunflower Compressed/Healthy/` | 1,204 | Fully expanded capitulum; viable pollen | Low Deficiency |
| `Sunflower Compressed/Wilted/` | 1,039 | Post-anthesis; senescent florets | High Deficiency (Post-pollination) |
| **Total** | **5,351** | **Complete Developmental Spectrum** | **Full Distribution** |

- **Phase 1 Evaluation Set:** Deterministic stratified sample of 1,500 images (300 per class, random seed 42).
- **Phase 2 Quantum Evaluation Set:** Balanced 220-image subset for binary classification (Low vs. High PDI deficiency).

---

## 4. Key Experimental Results & Benchmarks

### Summary of Results

| Model / Architecture | Task | Input Dimension | Test Accuracy | F1-Score (Macro) |
|---|---|---|---|---|
| **Random Forest** | 5-Stage Growth Classification | 35 Handcrafted | **80.9%** | 0.81 |
| **SVM (RBF Kernel)** | 5-Stage Growth Classification | 35 Handcrafted | **78.2%** | 0.78 |
| **Random Forest** | 3-Class PDI Proxy (Low/Med/High)| 35 Handcrafted | **87.6%** | 0.87 |
| **SVM (RBF Kernel)** | 3-Class PDI Proxy (Low/Med/High)| 35 Handcrafted | **90.7%** | 0.91 |
| **PennyLane VQC (6 Qubits)** | Binary PDI (Low vs. High) | 6-D PCA Embedding | **90.9%** | **0.91** |
| **SVM Comparator (Fair)** | Binary PDI (Low vs. High) | Identical 6-D PCA | **97.0%** | **0.97** |

> **Scientific Transparency Note:** At the current scale (220 samples, 6 PCA components), the classical SVM slightly outperforms the VQC (97.0% vs. 90.9%). This highlights the NISQ-era reality: classical kernels are highly effective in low-dimensional spaces, while the VQC demonstrates strong quantum state convergence and competitive feature mapping.

### Quantum Noise Robustness

Evaluated using PennyLane density matrix simulation (`default.mixed`) under depolarizing quantum channel noise:

| Depolarizing Noise Probability ($p$) | Test Accuracy | State Fidelity |
|---|---|---|
| $p = 0.00$ (Ideal / Noise-Free) | **90.9%** | 1.000 |
| $p = 0.02$ | **90.9%** | 0.942 |
| $p = 0.05$ | **87.9%** | 0.865 |
| $p = 0.08$ | **63.6%** | 0.710 |
| $p = 0.15$ | **51.5%** (Chance level) | 0.540 |

---

## 5. Repository Structure

```
Sunflower PDI Quantum Project/
├── README.md                          # Root system architecture & documentation
├── .gitignore                         # Excludes venv, pycache, OS caches
├── Sunflower Compressed/              # Full real-world image dataset (5,351 images)
│   ├── EarlyBloom/                    # 1,080 images
│   ├── Healthy/                       # 1,204 images
│   ├── MatureBud/                     # 1,010 images
│   ├── Wilted/                        # 1,039 images
│   └── YoungBud/                      # 1,027 images
│
├── sunflower-pdi/                     # Modular Production Package
│   ├── configs/                       # YAML configuration files
│   │   ├── base.yaml                  # Paths, classes, seed settings
│   │   ├── phase1.yaml                # Preprocessing & classical model parameters
│   │   └── phase2.yaml                # Quantum circuit, qubits, PCA configuration
│   ├── src/                           # Source modules
│   │   ├── classical/                 # Feature extraction, PDI proxy, classifiers
│   │   ├── quantum/                   # PennyLane circuits, VQC ansatz, ablation, noise
│   │   ├── preprocessing/             # CLAHE, Lab conversion, GrabCut, K-means
│   │   └── utils/                     # Metrics, logging, data split loaders
│   ├── scripts/                       # Orchestration scripts
│   │   └── run_all.py                 # End-to-end execution runner
│   ├── tests/                         # Pytest test suite (unit & integration tests)
│   ├── artifacts/                     # Metrics, plots, confusion matrices, logs
│   │   ├── figures/                   # Architecture diagrams, ROC, heatmaps, VQC plots
│   │   └── metrics/                   # JSON and CSV evaluation summaries
│   ├── models/                        # Serialized weights (.joblib, .npz)
│   ├── requirements.txt               # Dependency specifications
│   ├── draw_circuit.py                # Visual circuit diagram generator
│   └── SOURCE_SPEC.md                 # Technical specification notes
│
└── code_package/                      # Standalone Demonstration Pipeline
    ├── 01_preprocess_demo.py          # Visual preprocessing demo (CLAHE + Bilateral)
    ├── 02_segment_demo.py             # Baseline HSV+Otsu segmentation comparison
    ├── 02b_segment_grabcut.py         # GrabCut + K-Means disc extraction demo
    ├── 03_extract_features.py         # 35-feature extraction over 1,500 images
    ├── 04_train_classical.py          # RF & SVM baseline training script
    ├── 05_quantum_vqc.py              # PennyLane 6-qubit VQC training & noise sweep
    ├── 06_make_result_plots.py        # Generates all publication/presentation charts
    ├── features.csv                   # Precomputed 35-D feature dataset (1,500 rows)
    ├── classical_results.json         # Raw classical evaluation benchmarks
    └── quantum_results.json           # Raw quantum evaluation benchmarks
```

---

## 6. Installation & Reproducibility Guide

### Setup Virtual Environment

Ensure you have Python 3.10 or newer installed:

```bash
# Clone the repository
git clone https://github.com/BoseBoss007/Sunflower-PDI-Quantum.git
cd Sunflower-PDI-Quantum

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r sunflower-pdi/requirements.txt
```

### Execution Option A: Modular Pipeline (`sunflower-pdi`)

To execute the entire end-to-end pipeline (Phase 1 classical + Phase 2 quantum) in one command:

```bash
cd sunflower-pdi
PYTHONPATH=. python scripts/run_all.py --stage all
```

To run individual stages:
```bash
# Phase 1 only (Classical preprocessing, segmentation, feature extraction, RF/SVM)
PYTHONPATH=. python scripts/run_all.py --stage phase1

# Phase 2 only (PCA reduction, PennyLane VQC, noise sweep, quantum evaluation)
PYTHONPATH=. python scripts/run_all.py --stage phase2
```

Generate circuit visualizations and evaluation plots:
```bash
PYTHONPATH=. python draw_circuit.py
PYTHONPATH=. python comprehensive_evaluation.py
PYTHONPATH=. python compare_models.py
```

### Execution Option B: Standalone Step-by-Step Scripts (`code_package`)

To trace and run each script independently from `code_package`:

```bash
cd code_package

# 1. Visualize CLAHE and Bilateral Preprocessing
python 01_preprocess_demo.py

# 2. GrabCut and K-Means Disc Segmentation
python 02b_segment_grabcut.py

# 3. Extract 35 features across 1,500 samples (generates features.csv)
python 03_extract_features.py

# 4. Train Classical Random Forest & SVM Models
python 04_train_classical.py

# 5. Train PennyLane Variational Quantum Circuit (VQC)
python 05_quantum_vqc.py

# 6. Generate All Result Plots & Visualizations
python 06_make_result_plots.py
```

---

## 7. Verification & Automated Testing

The modular package includes automated unit and integration tests covering the quantum circuit, classical vision, segmentation, and PDI scoring rules:

```bash
cd sunflower-pdi
pytest tests/ -v
```

---

## 8. Limitations & Research Roadmap

1. **Biological Validation:** PDI in this project is calculated via an agronomic visual proxy. Future field studies will correlate these visual scores directly with microscopic pollen grain counts and actual hybrid seed-set percentages.
2. **Quantum Hardware Deployment:** Current quantum circuits are simulated on CPU via PennyLane. Future stages will deploy the 6-qubit ansatz to physical superconducting quantum hardware (e.g., IBM Quantum / Amazon Braket).
3. **Deep Visual Embeddings:** Combining classical CNN features (MobileNetV3 / ResNet-18) with quantum kernel estimation or Quanvolutional layers.
4. **Aerial UAV Integration:** Transitioning from ground-level photography to georeferenced drone imagery for automated real-time canopy survey.

---

## 9. Citation & Contact

If you use this codebase or dataset in your research, please cite:

```bibtex
@misc{sunflower_pdi_quantum_2026,
  author = {25MCs0034},
  title = {Quantum-Enhanced Vision-Based Pollen Deficiency Estimation for Precision Sunflower Pollination},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/BoseBoss007/Sunflower-PDI-Quantum}}
}
```
