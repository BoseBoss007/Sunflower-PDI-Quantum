import cv2
import numpy as np
import pandas as pd
import os
import glob
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def hsv_otsu_baseline(base_config_path, phase1_config_path):
    base_config = load_config(base_config_path)
    phase1_config = load_config(phase1_config_path)
    
    dataset_dir = base_config['dataset_dir']
    classes = base_config['classes']
    artifacts_dir = base_config['artifacts_dir']
    size = phase1_config['preprocessing']['resize']
    
    os.makedirs(os.path.join(artifacts_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, "figures"), exist_ok=True)
    
    results = []
    
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        files = sorted(glob.glob(os.path.join(cls_dir, "*.jpg")))
        if not files:
            continue
            
        sample_path = files[len(files)//2]
        
        img = cv2.imread(sample_path)
        if img is None: continue
        
        # We need RGB for saving, HSV for processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)
        
        # Negative Baseline: HSV Hue 10-45 + Otsu on Value
        hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
        
        # Hue mask
        lower_hue = np.array([10, 0, 0])
        upper_hue = np.array([45, 255, 255])
        hue_mask = cv2.inRange(hsv, lower_hue, upper_hue)
        
        # Otsu on Value
        v_channel = hsv[:, :, 2]
        _, otsu_mask = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combine
        combined_mask = cv2.bitwise_and(hue_mask, otsu_mask)
        
        fg_ratio = (combined_mask == 255).sum() / (size * size)
        
        results.append({
            "class": cls,
            "foreground_ratio": fg_ratio
        })
        
        # Save visualization for one example
        out_path = os.path.join(artifacts_dir, "figures", f"hsv_otsu_failure_{cls}.png")
        # create a 1x2 image showing original and mask
        mask_colored = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2RGB)
        vis = np.hstack((resized, mask_colored))
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
    df = pd.DataFrame(results)
    print("HSV+Otsu Negative Baseline Results:")
    print(df)
    
    # Save the log
    df.to_csv(os.path.join(artifacts_dir, "logs", "hsv_otsu_baseline.csv"), index=False)
    
if __name__ == "__main__":
    hsv_otsu_baseline("configs/base.yaml", "configs/phase1.yaml")
