import cv2
import numpy as np

def grabcut_mask(img, margin=0.10, iters=4):
    """
    Perform GrabCut segmentation.
    img is expected to be in RGB or BGR, but grabCut takes BGR.
    Assuming img is RGB here as per our pipeline, we convert to BGR.
    """
    h, w = img.shape[:2]
    rect = (int(w*margin), int(h*margin), int(w*(1-2*margin)), int(h*(1-2*margin)))
    
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    try:
        cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        # Fallback if grabcut fails entirely
        return np.ones((h, w), np.uint8) * 255
        
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    
    # Morphology
    kernel = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Largest connected component
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    clean = np.zeros_like(fg)
    if n > 1:
        # Background is label 0, so find largest among 1 to n-1
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean[labels == largest] = 255
        
    # If the component is too small, fallback
    if (clean == 255).sum() < 50:
        return np.ones((h, w), np.uint8) * 255
        
    return clean

def disc_mask_kmeans(img, fg_mask, k=3):
    """
    Extract the central disc using K-means clustering on the foreground.
    """
    ys, xs = np.where(fg_mask == 255)
    disc_mask = np.zeros_like(fg_mask)
    
    if len(ys) < 30:
        return disc_mask
        
    pix = img[ys, xs].astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    
    _, lbl, centers = cv2.kmeans(pix, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    
    # Identify darkest cluster
    brightness = centers.sum(axis=1)
    disc_cluster = int(np.argmin(brightness))
    
    sel = (lbl.flatten() == disc_cluster)
    disc_mask[ys[sel], xs[sel]] = 255
    
    return disc_mask
