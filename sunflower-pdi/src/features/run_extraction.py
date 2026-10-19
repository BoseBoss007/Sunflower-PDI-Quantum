import os
import time
import pandas as pd
import json
import sys
import yaml

from src.preprocessing.pipeline import Preprocessor
from src.segmentation.pipeline import grabcut_mask, disc_mask_kmeans
from src.features.extractor import color_hist_features, disc_color_features, glcm_features, shape_descriptors
from src.pdi.rules import pdi_proxy_label

def run_extraction(base_config_path, phase1_config_path):
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    data_dir = base_config['data_dir']
    artifacts_dir = base_config['artifacts_dir']
    seed = base_config['random_seed']
    
    split_file = os.path.join(data_dir, "splits", f"phase1_1500_seed{seed}.csv")
    if not os.path.exists(split_file):
        print("Split file not found.")
        return
        
    df_split = pd.read_csv(split_file)
    preprocessor = Preprocessor(phase1_config_path)
    
    os.makedirs(os.path.join(artifacts_dir, "features"), exist_ok=True)
    
    rows = []
    t_start = time.time()
    
    total = len(df_split)
    
    for idx, row in df_split.iterrows():
        path = row['image_path']
        cls = row['class_name']
        
        img = preprocessor.read_image(path)
        if img is None:
            continue
            
        proc = preprocessor.process(img)
        fg = grabcut_mask(proc)
        disc = disc_mask_kmeans(proc, fg)
        
        shape = shape_descriptors(fg)
        color_feats = color_hist_features(proc, fg)
        glcm_feats = glcm_features(proc, fg)
        
        disc_ratio, sat_mean, val_mean = disc_color_features(proc, fg, disc)
        
        label = pdi_proxy_label(disc_ratio, sat_mean, val_mean, shape['circularity'])
        
        feat_row = {
            "path": path, "class": cls,
            "area_ratio": shape['area_ratio'], "extent": shape['extent'],
            "solidity": shape['solidity'], "circularity": shape['circularity'],
            "disc_ratio": disc_ratio, "sat_mean": sat_mean, "val_mean": val_mean,
            **{f"h_hist_{i}": v for i, v in enumerate(color_feats)},
            **glcm_feats,
            "pdi_label": label,
        }
        
        rows.append(feat_row)
        
        if (idx+1) % 100 == 0:
            print(f"Extracted {idx+1}/{total}  elapsed={time.time()-t_start:.1f}s")
            
    df_features = pd.DataFrame(rows)
    out_csv = os.path.join(artifacts_dir, "features", "phase1_features.csv")
    df_features.to_csv(out_csv, index=False)
    
    schema = {
        "features": [c for c in df_features.columns if c not in ["path", "class", "pdi_label"]],
        "feature_count": len(df_features.columns) - 3
    }
    
    out_schema = os.path.join(artifacts_dir, "features", "feature_schema.json")
    with open(out_schema, "w") as f:
        json.dump(schema, f, indent=2)
        
    print(f"\nDone. {len(df_features)} features saved to {out_csv}")

if __name__ == "__main__":
    # add project root to pythonpath
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    run_extraction("configs/base.yaml", "configs/phase1.yaml")
