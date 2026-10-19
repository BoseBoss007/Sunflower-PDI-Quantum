import os
import json
import yaml
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_classical_training(base_config_path, phase1_config_path):
    base_config = load_config(base_config_path)
    phase1_config = load_config(phase1_config_path)
    
    data_dir = base_config['data_dir']
    artifacts_dir = base_config['artifacts_dir']
    models_dir = base_config['models_dir']
    seed = base_config['random_seed']
    
    os.makedirs(os.path.join(models_dir, "classical"), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, "figures"), exist_ok=True)
    
    features_csv = os.path.join(artifacts_dir, "features", "phase1_features.csv")
    split_csv = os.path.join(data_dir, "splits", f"phase1_1500_seed{seed}.csv")
    
    if not os.path.exists(features_csv) or not os.path.exists(split_csv):
        print("Features or split missing.")
        return
        
    df_feat = pd.read_csv(features_csv)
    df_split = pd.read_csv(split_csv)
    
    # We assume 1-1 mapping since extractor iterated df_split
    df_feat['split'] = df_split['split']
    
    feature_cols = [c for c in df_feat.columns if c not in ["path", "class", "pdi_label", "split"]]
    
    def evaluate_task(task_name, target_col, class_order):
        print(f"\n--- {task_name} ---")
        le = LabelEncoder()
        le.fit(class_order)
        
        y = le.transform(df_feat[target_col])
        X = np.nan_to_num(df_feat[feature_cols].values)
        
        train_idx = df_feat['split'] == 'train'
        val_idx = df_feat['split'] == 'val'
        test_idx = df_feat['split'] == 'test'
        
        X_train, y_train = X[train_idx], y[train_idx]
        # For simplicity in evaluation, we merge val and test or just evaluate on test since the requirements say "70/15/15" 
        # and evaluate on test. 
        X_test, y_test = X[test_idx], y[test_idx]
        
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Train RF
        rf_config = phase1_config['random_forest']
        rf = RandomForestClassifier(n_estimators=rf_config['n_estimators'], random_state=seed, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        rf_pred = rf.predict(X_test_s)
        
        rf_acc = accuracy_score(y_test, rf_pred)
        rf_f1 = f1_score(y_test, rf_pred, average='macro')
        rf_cm = confusion_matrix(y_test, rf_pred)
        
        # Train SVM
        svm_config = phase1_config['svm']
        svm = SVC(kernel=svm_config['kernel'], C=svm_config['C'], gamma=svm_config['gamma'], random_state=seed)
        svm.fit(X_train_s, y_train)
        svm_pred = svm.predict(X_test_s)
        
        svm_acc = accuracy_score(y_test, svm_pred)
        svm_f1 = f1_score(y_test, svm_pred, average='macro')
        svm_cm = confusion_matrix(y_test, svm_pred)
        
        print(f"RF -> Acc: {rf_acc:.3f}, F1: {rf_f1:.3f}")
        print(f"SVM -> Acc: {svm_acc:.3f}, F1: {svm_f1:.3f}")
        
        # Save models
        joblib.dump(rf, os.path.join(models_dir, "classical", f"random_forest_{task_name}.joblib"))
        joblib.dump(svm, os.path.join(models_dir, "classical", f"svm_{task_name}.joblib"))
        
        # Plot Confusion Matrices
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=class_order)
        disp.plot(ax=ax, cmap='Blues')
        plt.title(f'Random Forest - {task_name.capitalize()}')
        plt.savefig(os.path.join(artifacts_dir, "figures", f"confusion_matrix_{task_name}_rf.png"))
        plt.close()
        
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=svm_cm, display_labels=class_order)
        disp.plot(ax=ax, cmap='Blues')
        plt.title(f'SVM - {task_name.capitalize()}')
        plt.savefig(os.path.join(artifacts_dir, "figures", f"confusion_matrix_{task_name}_svm.png"))
        plt.close()
        
        return {
            "random_forest": {"accuracy": rf_acc, "macro_f1": rf_f1, "confusion_matrix": rf_cm.tolist()},
            "svm": {"accuracy": svm_acc, "macro_f1": svm_f1, "confusion_matrix": svm_cm.tolist()}
        }
        
    results = {}
    results["stage"] = evaluate_task("stage", "class", ["YoungBud", "MatureBud", "EarlyBloom", "Healthy", "Wilted"])
    results["pdi"] = evaluate_task("pdi", "pdi_label", ["Low", "Medium", "High"])
    
    with open(os.path.join(artifacts_dir, "metrics", "classical_results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    # Also save a CSV
    rows = []
    for tsk, res in results.items():
        rows.append({"task": tsk, "model": "Random Forest", "accuracy": res["random_forest"]["accuracy"], "macro_f1": res["random_forest"]["macro_f1"]})
        rows.append({"task": tsk, "model": "SVM", "accuracy": res["svm"]["accuracy"], "macro_f1": res["svm"]["macro_f1"]})
    df_res = pd.DataFrame(rows)
    df_res.to_csv(os.path.join(artifacts_dir, "metrics", "classical_results.csv"), index=False)
    
if __name__ == "__main__":
    run_classical_training("configs/base.yaml", "configs/phase1.yaml")
