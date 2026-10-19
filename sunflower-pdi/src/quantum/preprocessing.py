import os
import json
import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_quantum_preprocessing(base_config_path, phase2_config_path):
    base_config = load_config(base_config_path)
    phase2_config = load_config(phase2_config_path)
    
    data_dir = base_config['data_dir']
    artifacts_dir = base_config['artifacts_dir']
    models_dir = base_config['models_dir']
    seed = base_config['random_seed']
    
    os.makedirs(os.path.join(models_dir, "quantum"), exist_ok=True)
    
    features_csv = os.path.join(artifacts_dir, "features", "phase1_features.csv")
    if not os.path.exists(features_csv):
        print("Features file not found.")
        return
        
    df = pd.read_csv(features_csv)
    sub = df[df['pdi_label'].isin(["Low", "High"])].copy()
    
    n_per_class = phase2_config['sample_size_per_class']
    
    parts = []
    for lbl, g in sub.groupby("pdi_label"):
        parts.append(g.sample(n=n_per_class, random_state=seed))
    sub_balanced = pd.concat(parts, axis=0).reset_index(drop=True)
    
    # 70/15/15 split -> 154/33/33
    train_size = phase2_config['split']['train']
    val_size = phase2_config['split']['val']
    test_size = phase2_config['split']['test']
    
    train_df, temp_df = train_test_split(sub_balanced, test_size=(val_size + test_size), random_state=seed, stratify=sub_balanced["pdi_label"])
    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(temp_df, train_size=val_ratio, random_state=seed, stratify=temp_df["pdi_label"])
    
    train_df = train_df.copy(); train_df['split'] = 'train'
    val_df = val_df.copy(); val_df['split'] = 'val'
    test_df = test_df.copy(); test_df['split'] = 'test'
    
    df_split = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    df_split.to_csv(os.path.join(data_dir, "splits", "phase2_220_binary.csv"), index=False)
    
    feature_cols = [c for c in df_split.columns if c not in ("path", "class", "pdi_label", "split")]
    
    X_train = df_split[df_split['split'] == 'train'][feature_cols].values
    X_train = np.nan_to_num(X_train)
    
    scaler = StandardScaler().fit(X_train)
    joblib.dump(scaler, os.path.join(models_dir, "quantum", "scaler.joblib"))
    
    X_all = np.nan_to_num(df_split[feature_cols].values)
    X_all_s = scaler.transform(X_all)
    
    n_components = phase2_config['pca']['n_components']
    
    # Fit PCA on train
    X_train_s = scaler.transform(X_train)
    pca = PCA(n_components=n_components, random_state=seed).fit(X_train_s)
    joblib.dump(pca, os.path.join(models_dir, "quantum", "pca.joblib"))
    
    variance = float(pca.explained_variance_ratio_.sum())
    print(f"PCA explained variance ({n_components} components): {variance:.4f}")
    
    with open(os.path.join(artifacts_dir, "metrics", "pca_variance.json"), "w") as f:
        json.dump({"components": n_components, "explained_variance": variance}, f, indent=2)
        
    X_all_pca = pca.transform(X_all_s)
    
    df_pca = pd.DataFrame(X_all_pca, columns=[f"pca_{i}" for i in range(n_components)])
    df_pca['path'] = df_split['path']
    df_pca['pdi_label'] = df_split['pdi_label']
    df_pca['split'] = df_split['split']
    
    df_pca.to_csv(os.path.join(artifacts_dir, "features", "phase2_pca_features.csv"), index=False)
    print("Phase 2 preprocessing complete.")

if __name__ == "__main__":
    run_quantum_preprocessing("configs/base.yaml", "configs/phase2.yaml")
