"""
Stage 4 - Classical baseline training & evaluation (REAL, measured results).
70/15/15 stratified split on the 1,500-image feature set from Stage 3.
Two tasks:
  A) 5-class growth-stage classification (YoungBud/MatureBud/EarlyBloom/Healthy/Wilted)
  B) 3-class PDI-proxy classification (Low/Medium/High)
Models: Random Forest and SVM (RBF kernel), both from scikit-learn.
All numbers below are computed by actually running these models on the
actual extracted features - nothing here is hand-typed or invented.
"""
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

OUT = "/home/claude/pipeline/outputs"
df = pd.read_csv(f"{OUT}/features.csv")

feature_cols = [c for c in df.columns if c not in ("path", "class", "pdi_label")]
X = df[feature_cols].values
X = np.nan_to_num(X)

results = {}

def run_task(y_raw, task_name, class_order):
    le = LabelEncoder()
    le.fit(class_order)
    y = le.transform(y_raw)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s, X_test_s = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    print(f"\n=== {task_name} ===")
    print(f"Train/Val/Test sizes: {len(y_train)}/{len(y_val)}/{len(y_test)}")

    task_result = {"n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test), "classes": list(class_order)}

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    rf_pred = rf.predict(X_test_s)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average='macro')
    rf_cm = confusion_matrix(y_test, rf_pred).tolist()
    print(f"Random Forest  -> Accuracy={rf_acc:.4f}  Macro-F1={rf_f1:.4f}")
    print(classification_report(y_test, rf_pred, target_names=class_order, zero_division=0))

    # --- SVM (RBF) ---
    svm = SVC(kernel='rbf', C=5.0, gamma='scale', random_state=42)
    svm.fit(X_train_s, y_train)
    svm_pred = svm.predict(X_test_s)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_f1 = f1_score(y_test, svm_pred, average='macro')
    svm_cm = confusion_matrix(y_test, svm_pred).tolist()
    print(f"SVM (RBF)      -> Accuracy={svm_acc:.4f}  Macro-F1={svm_f1:.4f}")
    print(classification_report(y_test, svm_pred, target_names=class_order, zero_division=0))

    task_result["random_forest"] = {"accuracy": rf_acc, "macro_f1": rf_f1, "confusion_matrix": rf_cm}
    task_result["svm"] = {"accuracy": svm_acc, "macro_f1": svm_f1, "confusion_matrix": svm_cm}
    task_result["y_test"] = y_test.tolist()
    task_result["rf_pred"] = rf_pred.tolist()
    task_result["svm_pred"] = svm_pred.tolist()
    return task_result

results["growth_stage"] = run_task(df["class"].values, "Task A: 5-class Growth-Stage Classification",
                                    ["YoungBud", "MatureBud", "EarlyBloom", "Healthy", "Wilted"])
results["pdi"] = run_task(df["pdi_label"].values, "Task B: 3-class PDI-Proxy Classification",
                           ["Low", "Medium", "High"])

with open(f"{OUT}/classical_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved classical_results.json")
