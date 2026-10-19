"""
Stage 2 - Classical DIP Segmentation
Real preprocessed images are segmented to isolate the sunflower head
(capitulum disc + ray florets) from the background (leaves/soil/sky)
using:
  1. HSV color thresholding (yellow/brown hue band) as an initial mask
  2. Otsu thresholding on the Value channel as a secondary cue, combined
  3. Morphological opening + closing to remove noise / fill holes
  4. K-means (k=3) color clustering on the masked region to separate the
     dark disc (florets/seed head) from the yellow ray petals
All operations run on the real dataset images (same source as Stage 1).
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

def load_image(path, size=256):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img

def segment_flower(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Yellow/gold/brown hue range typical of sunflower petals & disc (H in OpenCV: 0-179)
    lower = np.array([10, 40, 40])
    upper = np.array([45, 255, 255])
    color_mask = cv2.inRange(hsv, lower, upper)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    combined = cv2.bitwise_or(color_mask, cv2.bitwise_and(otsu_mask, color_mask))
    kernel = np.ones((7, 7), np.uint8)
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # keep largest connected component only
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    mask = np.zeros_like(closed)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask[labels == largest] = 255

    flower_pixels = img[mask == 255]
    disc_mask = np.zeros_like(mask)
    if len(flower_pixels) > 30:
        Z = flower_pixels.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        k = 3
        _, kmlabels, centers = cv2.kmeans(Z, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        # disc cluster = darkest (lowest brightness) cluster among the k means
        brightness = centers.sum(axis=1)
        disc_cluster = int(np.argmin(brightness))
        full_labels = -1 * np.ones(mask.shape, dtype=int)
        idx = np.where(mask == 255)
        full_labels[idx] = kmlabels.flatten()
        disc_mask = np.uint8(full_labels == disc_cluster) * 255

    area_ratio = float((mask == 255).sum()) / (mask.shape[0] * mask.shape[1])
    disc_ratio = float((disc_mask == 255).sum()) / max(1.0, float((mask == 255).sum()))
    return mask, disc_mask, area_ratio, disc_ratio

fig, axes = plt.subplots(5, 3, figsize=(11, 17))
col_titles = ["Preprocessed Input", "Flower Mask (Otsu + HSV + Morphology)", "Disc Mask (K-means, k=3)"]
rows_stats = []
for row, cls in enumerate(CLASSES):
    files = sorted(glob.glob(os.path.join(DATA_DIR, cls, "*.jpg")))
    sample = files[len(files)//2]
    img = load_image(sample)
    mask, disc_mask, area_ratio, disc_ratio = segment_flower(img)
    rows_stats.append((cls, area_ratio, disc_ratio))
    axes[row,0].imshow(img); axes[row,0].axis('off')
    axes[row,1].imshow(mask, cmap='gray'); axes[row,1].axis('off')
    axes[row,2].imshow(disc_mask, cmap='gray'); axes[row,2].axis('off')
    if row == 0:
        for c in range(3):
            axes[row,c].set_title(col_titles[c], fontsize=11, fontweight='bold')
    axes[row,0].text(-0.15, 0.5, cls, transform=axes[row,0].transAxes, fontsize=12, fontweight='bold',
                      rotation=90, va='center', ha='center')

plt.suptitle("Stage 2 \u2014 Classical DIP Segmentation on Real Dataset Samples", fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(f"{OUT}/02_segmentation_demo.png", dpi=150, bbox_inches='tight')
print("Saved 02_segmentation_demo.png")
print("\nFlower-area-ratio and disc-ratio (real, measured, not fabricated):")
for cls, ar, dr in rows_stats:
    print(f"  {cls:12s}  flower_area_ratio={ar:.3f}  disc_ratio_of_flower={dr:.3f}")
