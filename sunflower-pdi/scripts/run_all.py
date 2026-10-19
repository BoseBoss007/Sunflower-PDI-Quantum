import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, required=True, 
                        choices=['verify', 'phase1-preprocess', 'phase1-segmentation', 
                                 'phase1-features', 'phase1-pdi', 'phase1-classical',
                                 'phase2-vqc', 'noise', 'ablation', 'all'])
    args = parser.parse_help_text = parser.parse_args()
    
    stage = args.stage
    
    if stage in ['verify', 'all']:
        print("=== Running Dataset Verification ===")
        from src.data.verify_dataset import verify_dataset
        verify_dataset("configs/base.yaml")
        
        from src.data.split_dataset import create_split
        create_split("configs/base.yaml", "configs/phase1.yaml")
        
    if stage in ['phase1-preprocess', 'all']:
        print("\n=== Running Phase 1 Preprocessing ===")
        from src.preprocessing.pipeline import generate_examples
        generate_examples("configs/base.yaml", "configs/phase1.yaml")
        
    if stage in ['phase1-segmentation', 'all']:
        print("\n=== Running Phase 1 Segmentation (HSV+Otsu Baseline) ===")
        from src.segmentation.hsv_otsu import hsv_otsu_baseline
        hsv_otsu_baseline("configs/base.yaml", "configs/phase1.yaml")
        
    if stage in ['phase1-features', 'all']:
        print("\n=== Running Phase 1 Feature Extraction ===")
        from src.features.run_extraction import run_extraction
        run_extraction("configs/base.yaml", "configs/phase1.yaml")
        
    if stage in ['phase1-pdi', 'all']:
        print("\n=== Generating PDI Heatmaps ===")
        from src.pdi.heatmap import run_heatmaps
        run_heatmaps("configs/base.yaml", "configs/phase1.yaml")
        
    if stage in ['phase1-classical', 'all']:
        print("\n=== Running Phase 1 Classical Models ===")
        from src.classical.train import run_classical_training
        run_classical_training("configs/base.yaml", "configs/phase1.yaml")
        
    if stage in ['phase2-vqc', 'all']:
        print("\n=== Running Phase 2 Quantum Preprocessing ===")
        from src.quantum.preprocessing import run_quantum_preprocessing
        run_quantum_preprocessing("configs/base.yaml", "configs/phase2.yaml")
        
        print("\n=== Running Phase 2 VQC Training ===")
        from src.quantum.train_vqc import run_train_vqc
        run_train_vqc("configs/base.yaml", "configs/phase2.yaml")
        
    if stage in ['noise', 'all']:
        print("\n=== Running Phase 2 Noise Experiment ===")
        from src.quantum.noise import run_noise_experiment
        run_noise_experiment("configs/base.yaml", "configs/phase2.yaml")
        
    if stage in ['ablation', 'all']:
        print("\n=== Running Phase 2 PCA Ablation ===")
        from src.quantum.ablation import run_ablation
        run_ablation("configs/base.yaml", "configs/phase2.yaml")
        
    if stage == 'all':
        # Add metadata saving
        import json
        import datetime
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "python_version": sys.version,
            "seed": 42
        }
        with open("artifacts/run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
    print("\nDone.")

if __name__ == "__main__":
    main()
