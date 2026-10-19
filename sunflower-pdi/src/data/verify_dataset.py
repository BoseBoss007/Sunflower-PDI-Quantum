import os
import glob
import cv2
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def verify_dataset(base_config_path):
    config = load_config(base_config_path)
    dataset_dir = config['dataset_dir']
    classes = config['classes']
    artifacts_dir = config['artifacts_dir']
    
    os.makedirs(artifacts_dir, exist_ok=True)
    
    report = {
        "dataset_root": dataset_dir,
        "classes_found": {},
        "total_images": 0,
        "unreadable_images": [],
        "extensions_found": {}
    }
    
    class_counts = []
    
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"Warning: Directory for class {cls} not found.")
            report["classes_found"][cls] = 0
            class_counts.append({"class": cls, "count": 0})
            continue
            
        files = glob.glob(os.path.join(cls_dir, "*"))
        count = 0
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            report["extensions_found"][ext] = report["extensions_found"].get(ext, 0) + 1
            
            # test readability
            try:
                img = cv2.imread(f)
                if img is None:
                    report["unreadable_images"].append(f)
                else:
                    count += 1
            except Exception as e:
                report["unreadable_images"].append(f)
                
        report["classes_found"][cls] = count
        report["total_images"] += count
        class_counts.append({"class": cls, "count": count})
        print(f"{cls}: {count} readable images")
        
    print(f"Total images: {report['total_images']}")
    
    # Save JSON
    with open(os.path.join(artifacts_dir, "dataset_verification.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    # Save CSV
    df = pd.DataFrame(class_counts)
    df.to_csv(os.path.join(artifacts_dir, "dataset_distribution.csv"), index=False)
    
    # Generate Figure
    plt.figure(figsize=(10, 6))
    plt.bar(df['class'], df['count'], color='skyblue')
    plt.title('Dataset Class Distribution')
    plt.xlabel('Growth Stage')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(os.path.join(artifacts_dir, "figures"), exist_ok=True)
    plt.savefig(os.path.join(artifacts_dir, "figures", "dataset_distribution.png"))
    plt.close()
    
    return report

if __name__ == "__main__":
    verify_dataset("configs/base.yaml")
