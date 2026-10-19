import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def color_hist_features(img, mask, bins=8):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], mask, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        feats.extend(hist.tolist())
    return feats

def disc_color_features(img, fg_mask, disc_mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    fg_area = (fg_mask == 255).sum()
    disc_area = (disc_mask == 255).sum()
    
    disc_ratio = float(disc_area) / max(1.0, float(fg_area))
    sat_mean = float(hsv[:, :, 1][fg_mask == 255].mean()) if fg_area > 0 else 0.0
    val_mean = float(hsv[:, :, 2][fg_mask == 255].mean()) if fg_area > 0 else 0.0
    
    return disc_ratio, sat_mean, val_mean

def glcm_features(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    masked = gray.copy()
    masked[mask == 0] = 0
    ys, xs = np.where(mask == 255)
    if len(ys) < 10:
        return dict(contrast=0, homogeneity=0, energy=0, correlation=0)
    
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = masked[y0:y1+1, x0:x1+1]
    crop = (crop / 32).astype(np.uint8)  # 8 gray levels
    
    glcm = graycomatrix(crop, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         levels=8, symmetric=True, normed=True)
                         
    return dict(
        contrast=float(graycoprops(glcm, 'contrast').mean()),
        homogeneity=float(graycoprops(glcm, 'homogeneity').mean()),
        energy=float(graycoprops(glcm, 'energy').mean()),
        correlation=float(np.nan_to_num(graycoprops(glcm, 'correlation')).mean()),
    )

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
