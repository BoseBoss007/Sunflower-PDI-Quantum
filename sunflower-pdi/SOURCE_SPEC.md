# Source Specification & Inconsistencies

This document logs discrepancies between the project report, requirements, and the actual executed code/artifacts found in the `code_package` directory, along with how they are resolved for this implementation.

## Feature Count
- **Report/Architecture:** Mentions 34 handcrafted features, but diagrams show 35-D. `05_quantum_vqc.py` has a print statement claiming `34-D`.
- **Actual Execution:** The `03_extract_features.py` script extracts 4 shape features, 3 disc features, 24 histogram features (3 channels × 8 bins), and 4 GLCM features, totalling **35 features**. The resulting `features.csv` has exactly 35 feature columns + path + class + label.
- **Resolution:** We will use the canonical **35 features** as implemented by the original script, noting the 34-D mention as a typo in the print statements.

## Resize Resolution
- **Report/Requirements:** Mentions 512x512 requirements and 256x256 in architecture diagrams.
- **Actual Execution:** 
  - `01_preprocess_demo.py` resizes to 256x256 for visualization.
  - `03_extract_features.py` sets `SIZE = 200` to speed up the full 1500-image extraction on CPU.
- **Resolution:** The feature-extraction pipeline will use **200x200** to ensure strict reproducibility of the reported metrics. We will support 256x256 as a configuration option for visualizations.

## CNN Features
- **Report/Architecture:** The architecture block diagram mentions MobileNetV3/ResNet18.
- **Actual Execution:** There is no implementation of deep feature extraction in the `code_package` directory. The results (Phase 1 & 2) rely entirely on handcrafted features.
- **Resolution:** CNN embeddings will be treated as an **optional future extension**. They were not used to generate the reported metrics.

## Quantum Scope
- **Report/Architecture:** General architecture mentions 4-10 qubits.
- **Actual Execution:** `05_quantum_vqc.py` implements exactly a 6-qubit (N_QUBITS=6) VQC acting on 6 PCA components of the features.
- **Resolution:** The core Phase 2 reproduction will strictly use **6 qubits**. Ablation on 2, 4, 6 PCA components using the SVM proxy will be implemented. A true VQC qubit ablation is an optional extension.
