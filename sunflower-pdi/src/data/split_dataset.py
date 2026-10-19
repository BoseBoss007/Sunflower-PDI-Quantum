import os
import glob
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_split(base_config_path, phase1_config_path):
    base_config = load_config(base_config_path)
    phase1_config = load_config(phase1_config_path)
    
    dataset_dir = base_config['dataset_dir']
    classes = base_config['classes']
    data_dir = base_config['data_dir']
    seed = base_config['random_seed']
    
    n_per_class = phase1_config['sample_size_per_class']
    train_size = phase1_config['split']['train']
    val_size = phase1_config['split']['val']
    test_size = phase1_config['split']['test']
    
    os.makedirs(os.path.join(data_dir, "splits"), exist_ok=True)
    
    records = []
    
    for class_id, cls in enumerate(classes):
        cls_dir = os.path.join(dataset_dir, cls)
        files = sorted(glob.glob(os.path.join(cls_dir, "*.jpg"))) # Ensure sorted before sampling
        
        # Sample N_PER_CLASS
        if len(files) < n_per_class:
            print(f"Warning: Not enough images in {cls}. Found {len(files)}, expected {n_per_class}.")
            sampled = files
        else:
            # use pandas sample for reproducibility with seed
            df_files = pd.DataFrame({"image_path": files})
            sampled = df_files.sample(n=n_per_class, random_state=seed)["image_path"].tolist()
            
        for path in sampled:
            # We want an absolute path for convenience, but relative path is also fine.
            records.append({
                "image_path": os.path.abspath(path),
                "class_name": cls,
                "class_id": class_id
            })
            
    df = pd.DataFrame(records)
    
    # Split
    # train 70, val 15, test 15
    train_df, temp_df = train_test_split(df, test_size=(val_size + test_size), random_state=seed, stratify=df["class_name"])
    
    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(temp_df, train_size=val_ratio, random_state=seed, stratify=temp_df["class_name"])
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    
    final_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    
    out_path = os.path.join(data_dir, "splits", f"phase1_{len(final_df)}_seed{seed}.csv")
    final_df.to_csv(out_path, index=False)
    
    print(f"Created split with {len(final_df)} samples at {out_path}")
    print("Split distribution:")
    print(final_df.groupby(["split", "class_name"]).size())

if __name__ == "__main__":
    create_split("configs/base.yaml", "configs/phase1.yaml")
