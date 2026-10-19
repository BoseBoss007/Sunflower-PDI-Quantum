import os
import glob
import cv2
import numpy as np
import yaml
from src.preprocessing.pipeline import Preprocessor
from src.segmentation.pipeline import grabcut_mask, disc_mask_kmeans
from src.features.extractor import shape_descriptors, disc_color_features
from src.pdi.rules import pdi_proxy_label

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_heatmaps(base_config_path, phase1_config_path):
    base_config = load_config(base_config_path)
    
    dataset_dir = base_config['dataset_dir']
    classes = base_config['classes']
    artifacts_dir = base_config['artifacts_dir']
    
    os.makedirs(os.path.join(artifacts_dir, "figures"), exist_ok=True)
    preprocessor = Preprocessor(phase1_config_path)
    
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        files = sorted(glob.glob(os.path.join(cls_dir, "*.jpg")))
        if not files:
            continue
            
        sample = files[len(files)//2]
        
        img = preprocessor.read_image(sample)
        proc = preprocessor.process(img)
        fg = grabcut_mask(proc)
        disc = disc_mask_kmeans(proc, fg)
        
        shape = shape_descriptors(fg)
        disc_ratio, sat_mean, _ = disc_color_features(proc, fg, disc)
        
        label = pdi_proxy_label(cls, disc_ratio, shape['area_ratio'], sat_mean)
        
        # Color mapping for heatmap overlay
        color_map = {
            "Low": (0, 255, 0),     # Green
            "Medium": (0, 255, 255),# Yellow
            "High": (255, 0, 0)     # Red
        }
        
        color = color_map.get(label, (255, 255, 255))
        
        # Create an overlay on the original resized image
        resized = cv2.resize(img, (preprocessor.size, preprocessor.size), interpolation=cv2.INTER_AREA)
        overlay = resized.copy()
        
        # Highlight disc region more strongly, and foreground lightly
        # Using a simple alpha blend where mask is present
        alpha_fg = 0.3
        alpha_disc = 0.6
        
        colored_mask = np.zeros_like(resized)
        colored_mask[fg == 255] = color
        
        cv2.addWeighted(colored_mask, alpha_fg, overlay, 1 - alpha_fg, 0, overlay)
        
        colored_disc = np.zeros_like(resized)
        colored_disc[disc == 255] = color
        cv2.addWeighted(colored_disc, alpha_disc, overlay, 1, 0, overlay)
        
        # Add text
        cv2.putText(overlay, f"PDI Proxy: {label}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out_path = os.path.join(artifacts_dir, "figures", f"heatmap_{cls}.png")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
    print("Heatmaps generated.")

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    run_heatmaps("configs/base.yaml", "configs/phase1.yaml")
