import os
import yaml
import pandas as pd
import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_ablation(base_config_path, phase2_config_path):
    base_config = load_config(base_config_path)
    phase2_config = load_config(phase2_config_path)
    
    artifacts_dir = base_config['artifacts_dir']
    seed = base_config['random_seed']
    
    df_split = pd.read_csv(os.path.join(base_config['data_dir'], "splits", "phase2_220_binary.csv"))
    
    feature_cols = [c for c in df_split.columns if c not in ("path", "class", "pdi_label", "split")]
    
    train_idx = df_split['split'] == 'train'
    test_idx = df_split['split'] == 'test'
    
    X_train = np.nan_to_num(df_split[train_idx][feature_cols].values)
    X_test = np.nan_to_num(df_split[test_idx][feature_cols].values)
    
    y_train = (df_split[train_idx]['pdi_label'].values == "High").astype(int)
    y_test = (df_split[test_idx]['pdi_label'].values == "High").astype(int)
    
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    comps = phase2_config['ablation']['components']
    
    results = []
    
    for nq in comps:
        pca = PCA(n_components=nq, random_state=seed).fit(X_train_s)
        svm = SVC(kernel='rbf', C=5.0, gamma='scale', random_state=seed)
        
        Xp_train = pca.transform(X_train_s)
        Xp_test = pca.transform(X_test_s)
        
        svm.fit(Xp_train, y_train)
        pred = svm.predict(Xp_test)
        
        acc = accuracy_score(y_test, pred)
        var = float(pca.explained_variance_ratio_.sum())
        
        results.append({
            "n_components": nq,
            "explained_variance": var,
            "classical_svm_acc_same_dims": acc
        })
        print(f"PCA {nq} comps -> Var: {var:.3f}, SVM Acc: {acc:.4f}")
        
    with open(os.path.join(artifacts_dir, "metrics", "ablation_qubits.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    run_ablation("configs/base.yaml", "configs/phase2.yaml")
