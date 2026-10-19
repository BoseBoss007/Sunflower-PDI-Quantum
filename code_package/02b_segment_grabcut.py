"""
Stage 2 (revised) - Classical DIP Segmentation
The naive HSV hue-threshold approach over-segmented background foliage
(logged and discarded - see 02_segment_demo.py output). Replaced with a
more robust classical pipeline:
  1. GrabCut (iterative graph-cut / GMM foreground-background classical
     segmentation, seeded with a central rectangle - no deep learning,
     no training data) to isolate the flower/bud from background.
  2. Morphological cleanup (open+close) + largest-connected-component.
  3. K-means (k=3) color clustering *within* the foreground mask to
     separate the dark disc/floret region from bright ray petals /
     green bud tissue.
Run on the same real dataset images as before.
"""
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

DATA_DIR = "/home/claude/dataset/extracted/Sunflower Compressed"
CLASSES = ["YoungBud", "MatureBud", "EarlyBloom", "Healthy", "Wilted"]
OUT = "/home/claude/pipeline/outputs"

def load_image(path, size=256):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img

def grabcut_mask(img, margin=0.10, iters=5):
    h, w = img.shape[:2]
    rect = (int(w*margin), int(h*margin), int(w*(1-2*margin)), int(h*(1-2*margin)))
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    clean = np.zeros_like(fg)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean[labels == largest] = 255
    return clean

def disc_from_kmeans(img, fg_mask, k=3):
    ys, xs = np.where(fg_mask == 255)
    disc_mask = np.zeros_like(fg_mask)
    if len(ys) < 30:
        return disc_mask
    pix = img[ys, xs].astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    _, lbl, centers = cv2.kmeans(pix, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    brightness = centers.sum(axis=1)
    disc_cluster = int(np.argmin(brightness))
    sel = (lbl.flatten() == disc_cluster)
    disc_mask[ys[sel], xs[sel]] = 255
    return disc_mask

fig, axes = plt.subplots(5, 3, figsize=(11, 17))
col_titles = ["Preprocessed Input", "GrabCut Foreground Mask", "Disc Mask (K-means, k=3)"]
rows_stats = []
for row, cls in enumerate(CLASSES):
    files = sorted(glob.glob(os.path.join(DATA_DIR, cls, "*.jpg")))
    sample = files[len(files)//2]
    img = load_image(sample)
    fg = grabcut_mask(img)
    disc = disc_from_kmeans(img, fg)
    area_ratio = float((fg == 255).sum()) / (fg.shape[0]*fg.shape[1])
    disc_ratio = float((disc == 255).sum()) / max(1.0, float((fg == 255).sum()))
    rows_stats.append((cls, area_ratio, disc_ratio))

    axes[row,0].imshow(img); axes[row,0].axis('off')
    overlay = img.copy()
    overlay[fg == 0] = (overlay[fg == 0] * 0.25).astype(np.uint8)
    axes[row,1].imshow(overlay); axes[row,1].axis('off')
    axes[row,2].imshow(disc, cmap='gray'); axes[row,2].axis('off')
    if row == 0:
        for c in range(3):
            axes[row,c].set_title(col_titles[c], fontsize=11, fontweight='bold')
    axes[row,0].text(-0.15, 0.5, cls, transform=axes[row,0].transAxes, fontsize=12, fontweight='bold',
                      rotation=90, va='center', ha='center')

plt.suptitle("Stage 2 (revised) \u2014 GrabCut Segmentation on Real Dataset Samples", fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(f"{OUT}/02b_segmentation_grabcut.png", dpi=150, bbox_inches='tight')
print("Saved 02b_segmentation_grabcut.png")
print("\nReal measured ratios:")
for cls, ar, dr in rows_stats:
    print(f"  {cls:12s}  foreground_area_ratio={ar:.3f}  disc_ratio_of_foreground={dr:.3f}")
