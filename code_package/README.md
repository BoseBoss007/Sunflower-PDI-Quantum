# Sunflower PDI — Real Executed Pipeline

All scripts here were actually run against the real 5,350-image dataset
(downloaded from github.com/BoseBoss007/sunflower-pdi-dataset) in the
order below. Re-run them yourself by pointing DATA_DIR at your unzipped
dataset folder.

Run order:
1. `01_preprocess_demo.py`   — preprocessing visual (Fig. 10 / Slide 11)
2. `02_segment_demo.py`      — first (over-segmenting) attempt, HSV+Otsu — kept for transparency
3. `02b_segment_grabcut.py`  — corrected segmentation, GrabCut + K-means (Fig. 11 / Slide 11)
4. `03_extract_features.py`  — full feature extraction on a 1,500-image real stratified sample -> features.csv
5. `04_train_classical.py`   — Random Forest + SVM baselines -> classical_results.json
6. `05_quantum_vqc.py`       — PennyLane VQC training + noise sweep + ablation -> quantum_results.json
7. `06_make_result_plots.py` — generates every chart used in the report/PPT from the JSON results

Requirements: opencv-python, scikit-image, scikit-learn, pandas, matplotlib, pennylane

Key real results (see quantum_results.json / classical_results.json for full detail):
- Growth-stage classification (5-class): RF 80.9% acc / SVM 78.2% acc (test n=225)
- PDI-proxy classification (3-class): RF 87.6% acc / SVM 90.7% acc (test n=225)
- VQC (6 qubits, real training): 90.9% acc on binary PDI task (test n=33)
- Classical SVM on identical 6-D features: 97.0% acc (classical wins at this scale)
- Noise robustness: stable to depolarizing p≈0.05, collapses to chance at p≥0.08
