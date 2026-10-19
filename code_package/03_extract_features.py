"""
Stage 3 - Full feature extraction on a stratified real sample.

Sample size: 300 images/class x 5 classes = 1,500 real images
(random seed fixed = 42, sampled without replacement from the actual
dataset folders). This subsampling is done purely for CPU tractability
on a laptop-class machine within a single run; every image used is a
real file from the downloaded dataset - nothing is synthesized.

For each image we compute, on the GrabCut-segmented foreground:
  - Color histogram features (HSV, 8 bins/channel = 24 features)
  - GLCM texture features (contrast, homogeneity, energy, correlation)
    computed on the grayscale foreground region
  - Shape descriptors of the foreground mask (area ratio, extent,
    solidity, circularity)
  - Disc-related features from the K-means disc sub-mask (disc ratio,
    disc mean HSV)
  - A rule-based Pollen-Deficiency-Index (PDI) proxy label

Output: /home/claude/pipeline/outputs/features.csv
"""
import cv2
import numpy as np
import glob
import os
import random
import time
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

DATA_DIR = "/home/claude/dataset/extracted/Sunflower Compressed"
CLASSES = ["YoungBud", "MatureBud", "EarlyBloom", "Healthy", "Wilted"]
OUT = "/home/claude/pipeline/outputs"
N_PER_CLASS = 300
SEED = 42
SIZE = 200  # slightly smaller than the 256 used for the visual demo, for speed

random.seed(SEED)
np.random.seed(SEED)

def load_image(path, size=SIZE):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img

def preprocess(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2RGB)
    denoised = cv2.bilateralFilter(enhanced, d=7, sigmaColor=60, sigmaSpace=60)
    return denoised

def grabcut_mask(img, margin=0.10, iters=4):
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

def disc_mask_kmeans(img, fg_mask, k=3):
    ys, xs = np.where(fg_mask == 255)
    disc_mask = np.zeros_like(fg_mask)
    if len(ys) < 30:
        return disc_mask
    pix = img[ys, xs].astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _, lbl, centers = cv2.kmeans(pix, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    brightness = centers.sum(axis=1)
    disc_cluster = int(np.argmin(brightness))
    sel = (lbl.flatten() == disc_cluster)
    disc_mask[ys[sel], xs[sel]] = 255
    return disc_mask

def shape_descriptors(mask):
    area = float((mask == 255).sum())
    h, w = mask.shape
    area_ratio = area / (h * w)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return dict(area_ratio=area_ratio, extent=0, solidity=0, circularity=0)
    c = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
    extent = float(cv2.contourArea(c)) / (bw * bh) if bw * bh > 0 else 0
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(cv2.contourArea(c)) / hull_area if hull_area > 0 else 0
    perimeter = cv2.arcLength(c, True)
    circularity = (4 * np.pi * cv2.contourArea(c)) / (perimeter ** 2) if perimeter > 0 else 0
    return dict(area_ratio=area_ratio, extent=extent, solidity=solidity, circularity=circularity)

def color_hist_features(img, mask, bins=8):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], mask, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        feats.extend(hist.tolist())
    return feats

def glcm_features(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    masked = gray.copy()
    masked[mask == 0] = 0
    ys, xs = np.where(mask == 255)
    if len(ys) < 10:
        return dict(contrast=0, homogeneity=0, energy=0, correlation=0)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = masked[y0:y1+1, x0:x1+1]
    crop = (crop / 32).astype(np.uint8)  # 8 gray levels for speed
    glcm = graycomatrix(crop, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         levels=8, symmetric=True, normed=True)
    return dict(
        contrast=float(graycoprops(glcm, 'contrast').mean()),
        homogeneity=float(graycoprops(glcm, 'homogeneity').mean()),
        energy=float(graycoprops(glcm, 'energy').mean()),
        correlation=float(np.nan_to_num(graycoprops(glcm, 'correlation')).mean()),
    )

def pdi_proxy_label(cls, disc_ratio, area_ratio, sat_mean):
    """
    Rule-based PDI proxy (documented, reproducible rule - not a ground-truth
    label, since no manual pollination annotations exist in the source
    dataset). Encodes domain assumptions stated in the project report:
      - Pre-bloom stages (bud, not yet receptive) -> High deficiency
      - Early bloom (partial floret exposure) -> Medium
      - Healthy full bloom with a well-formed, high-coverage disc -> Low
      - Wilted (post-pollination-window decline) -> Medium-High depending
        on how much structured disc area remains
    """
    if cls in ("YoungBud", "MatureBud"):
        return "High"
    if cls == "EarlyBloom":
        return "Medium" if disc_ratio < 0.30 else "Low"
    if cls == "Healthy":
        return "Low" if (disc_ratio > 0.12 and area_ratio > 0.15) else "Medium"
    if cls == "Wilted":
        return "Medium" if disc_ratio > 0.15 else "High"
    return "Medium"

rows = []
t_start = time.time()
for cls in CLASSES:
    files = sorted(glob.glob(os.path.join(DATA_DIR, cls, "*.jpg")))
    rng = random.Random(SEED)
    sample_files = rng.sample(files, min(N_PER_CLASS, len(files)))
    print(f"{cls}: sampling {len(sample_files)} of {len(files)} real images")
    for i, path in enumerate(sample_files):
        img = load_image(path)
        if img is None:
            continue
        proc = preprocess(img)
        fg = grabcut_mask(proc)
        if (fg == 255).sum() < 50:
            fg = np.ones_like(fg) * 255  # fallback: use whole frame if grabcut collapses
        disc = disc_mask_kmeans(proc, fg)

        shape = shape_descriptors(fg)
        color_feats = color_hist_features(proc, fg)
        glcm_feats = glcm_features(proc, fg)

        hsv = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)
        disc_ratio = float((disc == 255).sum()) / max(1.0, float((fg == 255).sum()))
        sat_mean = float(hsv[:, :, 1][fg == 255].mean()) if (fg == 255).sum() > 0 else 0.0
        val_mean = float(hsv[:, :, 2][fg == 255].mean()) if (fg == 255).sum() > 0 else 0.0

        label = pdi_proxy_label(cls, disc_ratio, shape['area_ratio'], sat_mean)

        row = {
            "path": path, "class": cls,
            "area_ratio": shape['area_ratio'], "extent": shape['extent'],
            "solidity": shape['solidity'], "circularity": shape['circularity'],
            "disc_ratio": disc_ratio, "sat_mean": sat_mean, "val_mean": val_mean,
            **{f"h_hist_{i}": v for i, v in enumerate(color_feats)},
            **glcm_feats,
            "pdi_label": label,
        }
        rows.append(row)
        if (i+1) % 100 == 0:
            print(f"  {cls}: {i+1}/{len(sample_files)}  elapsed={time.time()-t_start:.1f}s")

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/features.csv", index=False)
print(f"\nDone. {len(df)} real feature rows written to features.csv in {time.time()-t_start:.1f}s")
print(df['class'].value_counts())
print(df['pdi_label'].value_counts())
