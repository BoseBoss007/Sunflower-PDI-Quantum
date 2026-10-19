"""
Stage 1 - Preprocessing
Real sunflower images from the downloaded dataset are:
  1. Resized to 256x256 (256 chosen instead of 512 to keep full-dataset
     feature extraction tractable on CPU; documented explicitly).
  2. Converted RGB -> HSV/Lab for color-based analysis.
  3. Contrast-enhanced with CLAHE on the L channel (Lab space).
  4. Denoised with a bilateral filter (edge-preserving).
No synthetic images are used anywhere in this script.
"""
import cv2
import numpy as np
import glob
import os
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

DATA_DIR = "/home/claude/dataset/extracted/Sunflower Compressed"
CLASSES = ["YoungBud", "MatureBud", "EarlyBloom", "Healthy", "Wilted"]
OUT = "/home/claude/pipeline/outputs"
os.makedirs(OUT, exist_ok=True)

def load_image(path):
    img = cv2.imread(path)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def preprocess(img, size=256):
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    hsv = cv2.cvtColor(denoised, cv2.COLOR_RGB2HSV)
    return resized, enhanced, denoised, hsv

# Pick one real sample per class for a before/after visualization
fig, axes = plt.subplots(5, 4, figsize=(14, 17))
col_titles = ["1. Original (resized 256\u00d7256)", "2. CLAHE Contrast Enhanced", "3. Bilateral Denoised", "4. HSV representation"]
for row, cls in enumerate(CLASSES):
    files = sorted(glob.glob(os.path.join(DATA_DIR, cls, "*.jpg")))
    sample = files[len(files)//2]  # a real, deterministic pick (middle of sorted list), not cherry-picked
    img = load_image(sample)
    resized, enhanced, denoised, hsv = preprocess(img)
    for c, im in enumerate([resized, enhanced, denoised, hsv]):
        ax = axes[row, c]
        if c == 3:
            ax.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        else:
            ax.imshow(im)
        ax.axis('off')
        if row == 0:
            ax.set_title(col_titles[c], fontsize=11, fontweight='bold')
    axes[row, 0].set_ylabel(cls, fontsize=12, fontweight='bold')
    axes[row, 0].axis('on'); axes[row,0].set_xticks([]); axes[row,0].set_yticks([])
    for spine in axes[row,0].spines.values():
        spine.set_visible(False)

plt.suptitle("Stage 1 \u2014 Preprocessing Pipeline on Real Dataset Samples (one image per class)", fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(f"{OUT}/01_preprocessing_demo.png", dpi=150, bbox_inches='tight')
print("Saved 01_preprocessing_demo.png")
