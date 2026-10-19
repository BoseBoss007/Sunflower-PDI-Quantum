import cv2
import yaml
import os
import glob
import matplotlib.pyplot as plt

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class Preprocessor:
    def __init__(self, config_path="configs/phase1.yaml"):
        config = load_config(config_path)
        p_cfg = config['preprocessing']
        
        self.size = p_cfg['resize']
        self.clahe_clip = p_cfg['clahe_clip']
        self.clahe_tile = p_cfg['clahe_tile']
        self.d = p_cfg['denoise_d']
        self.sigma_color = p_cfg['denoise_sigma_color']
        self.sigma_space = p_cfg['denoise_sigma_space']
        
        self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_tile, self.clahe_tile))

    def read_image(self, path):
        img = cv2.imread(path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def process(self, img):
        # Resize
        resized = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        
        # RGB to Lab
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE
        l_eq = self.clahe.apply(l)
        enhanced_lab = cv2.merge((l_eq, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Bilateral denoise
        denoised = cv2.bilateralFilter(enhanced, d=self.d, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space)
        
        return denoised

def generate_examples(base_config_path, phase1_config_path):
    base_config = load_config(base_config_path)
    classes = base_config['classes']
    dataset_dir = base_config['dataset_dir']
    artifacts_dir = base_config['artifacts_dir']
    
    os.makedirs(os.path.join(artifacts_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, "preprocessed"), exist_ok=True)
    
    preprocessor = Preprocessor(phase1_config_path)
    
    fig, axes = plt.subplots(5, 3, figsize=(10, 15))
    col_titles = ["Original", "CLAHE Enhanced", "Bilateral Denoised (Final)"]
    
    for row, cls in enumerate(classes):
        cls_dir = os.path.join(dataset_dir, cls)
        files = sorted(glob.glob(os.path.join(cls_dir, "*.jpg")))
        if not files:
            continue
            
        sample = files[len(files)//2]
        
        # We simulate the intermediate steps for the plot
        img = preprocessor.read_image(sample)
        resized = cv2.resize(img, (preprocessor.size, preprocessor.size), interpolation=cv2.INTER_AREA)
        
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_eq = preprocessor.clahe.apply(l)
        enhanced_lab = cv2.merge((l_eq, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        denoised = cv2.bilateralFilter(enhanced, d=preprocessor.d, sigmaColor=preprocessor.sigma_color, sigmaSpace=preprocessor.sigma_space)
        
        # Save a final output
        out_path = os.path.join(artifacts_dir, "preprocessed", f"{cls}_example.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR))
        
        for c, im in enumerate([resized, enhanced, denoised]):
            ax = axes[row, c]
            ax.imshow(im)
            ax.axis('off')
            if row == 0:
                ax.set_title(col_titles[c], fontsize=11, fontweight='bold')
        axes[row, 0].set_ylabel(cls, fontsize=12, fontweight='bold')
        axes[row, 0].axis('on'); axes[row,0].set_xticks([]); axes[row,0].set_yticks([])
        for spine in axes[row,0].spines.values():
            spine.set_visible(False)
            
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "figures", "preprocessing_examples.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Preprocessing examples generated.")

if __name__ == "__main__":
    generate_examples("configs/base.yaml", "configs/phase1.yaml")
