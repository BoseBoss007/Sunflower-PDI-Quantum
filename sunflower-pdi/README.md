# Quantum-Enhanced Vision-Based Pollen Deficiency Estimation for Precision Sunflower Pollination

## Mission
This repository provides a reproducible, CPU/laptop-feasible implementation for computing a Pollen Deficiency Index (PDI) proxy for sunflowers using both classical vision pipelines and a Quantum Variational Circuit (VQC).

## Dataset
- 5,350 real ground-level sunflower images across 5 development stages: YoungBud, MatureBud, EarlyBloom, Healthy, Wilted.
- The pipeline samples from this dataset deterministically (seed 42). 1,500 images are used for Phase 1 (300 per class), and a balanced 220-image subset is used for the Phase 2 binary task (Low vs High deficiency).

## Phase 1 Pipeline: Classical Vision
1. **Preprocessing:** RGB to Lab color space conversion, CLAHE contrast enhancement on the L channel, and Bilateral denoising. Resized to 200x200 for feature extraction tractability.
2. **Segmentation:** GrabCut to extract the foreground flower region, followed by morphological operations. K-means (k=3) clustering extracts the central disc (darkest cluster) from the foreground.
3. **Feature Extraction:** 35 handcrafted features are computed, including HSV histograms, GLCM texture descriptors, and geometric shape properties (area, extent, solidity, circularity) on both foreground and disc masks.
4. **PDI Proxy:** A rule-based transparent proxy categorises each image into Low, Medium, or High pollen deficiency based on its growth stage, disc ratio, and saturation. Note: This is a **proxy**, as there is no biological pollen ground truth in the current project.
5. **Classical Modelling:** Random Forest and SVM (RBF) models classify growth stage and PDI proxy.

## Phase 2 Pipeline: Quantum Refinement
1. **Standardization & PCA:** Features for the 220-image subset are standardized and reduced to 6 principal components (fitted on train only).
2. **Angle Encoding:** The 6 PCA components are encoded into 6 qubits using RY rotations.
3. **VQC:** A 3-layer Variational Quantum Circuit using RY and RZ rotations with a ring of CNOTs for entanglement. Output is read from the Z expectation of qubit 0.
4. **Hybrid Head:** A classical sigmoid head processes the expectation value into a binary probability (Low vs. High deficiency).
5. **Fair Comparator:** An SVM trained on the identical 6-D PCA feature subspace.
6. **Noise Robustness:** Depolarizing noise simulation using density matrices (`default.mixed`).
7. **Ablation:** Evaluating classical proxy performance with 2, 4, and 6 PCA components.

## Limitations
- **No biological pollen ground truth:** The PDI is a visual proxy; future biological field validation (pollen measurements, seed-set measurements, or agronomist labels) is required.
- **No physical quantum hardware:** Executed entirely via PennyLane CPU simulation.
- **Limited Scope for Quantum:** The quantum experiment uses a small balanced subset (220 images) and binary classification. 
- **Classical vs. Quantum Advantage:** The current VQC (90.9% reported) does not outperform the identical-feature SVM (97.0% reported) on this subset.
- **Domain gap:** The current dataset consists of ground-level imagery rather than full UAV deployment.

## Reproducing the Results
Ensure you have Python 3.10+ installed.

1. Set up the virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the entire pipeline end-to-end:
   ```bash
   PYTHONPATH=. python scripts/run_all.py --stage all
   ```
3. Artifacts, metrics (JSON/CSV), models, and figures will be saved in the `artifacts/` and `models/` directories.

## Future Extensions
- **Full-data classical training:** Extend beyond the 1,500-image CPU benchmark.
- **CNN embeddings:** Experiment with MobileNetV3/ResNet18.
- **True VQC qubit ablation:** Train independent VQCs at 2/4/6/8/10 qubits.
- **Quanvolution:** Add a small quantum local-patch feature extractor.
- **Alternative encoding:** Test amplitude encoding if computationally feasible.
- **Biological validation:** Collect pollen measurements, seed-set measurements or agronomist pollination-adequacy labels.
- **UAV generalisation:** Validate on georeferenced/UAV imagery.
