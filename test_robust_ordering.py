import numpy as np
import matplotlib.pyplot as plt
from src.feature_extractor import load_image_gray, harris_response, detect_corners, subpixel_refine
from src.math_core import dlt_homography
import os

IMG_DIR = "data/raw_checkerboards"
GRID_SHAPE = (9, 6)

def test_robust_ordering(corners, grid_shape):
    cols, rows = grid_shape
    expected = cols * rows
    if len(corners) < expected: return None
    
    pts = np.array(corners, dtype=np.float64)[:expected]
    
    centered = pts - pts.mean(axis=0)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    ax_long = eigvecs[:, -1]
    ax_short = eigvecs[:, 0]
    
    if ax_long[0] < 0: ax_long = -ax_long
    if ax_short[1] < 0: ax_short = -ax_short
        
    u = centered @ ax_long
    v = centered @ ax_short
    
    idx_tl = np.argmin(u + v)
    idx_br = np.argmax(u + v)
    idx_tr = np.argmax(u - v)
    idx_bl = np.argmin(u - v)
    
    if len(set([idx_tl, idx_br, idx_tr, idx_bl])) != 4:
        print("Not 4 distinct corners")
        return None
        
    img_4 = np.array([
        pts[idx_tl], pts[idx_tr], pts[idx_bl], pts[idx_br]
    ], dtype=np.float64)
    
    ideal_4 = np.array([
        [0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]
    ], dtype=np.float64)
    
    H = dlt_homography(ideal_4, img_4)
    
    ordered = []
    taken = set()
    for r in range(rows):
        for c in range(cols):
            p_ideal = np.array([c, r, 1.0])
            p_img = H @ p_ideal
            p_img /= p_img[2]
            
            # find closest point in pts
            dists = np.linalg.norm(pts - p_img[:2], axis=1)
            # ignore taken
            for i in taken:
                dists[i] = np.inf
            
            best_idx = np.argmin(dists)
            if dists[best_idx] > 30.0: # if it's too far, NMS failed poorly
                print(f"Distance too far: {dists[best_idx]:.1f} px at r={r}, c={c}")
                return None
            
            taken.add(best_idx)
            ordered.append(tuple(pts[best_idx]))
            
    return ordered

paths = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR)][:5]
for p in paths:
    img = load_image_gray(p)
    resp = harris_response(img)
    raw = detect_corners(resp)
    ref = subpixel_refine(img, raw)
    ordered = test_robust_ordering(ref, GRID_SHAPE)
    if ordered is None:
        print(f"{os.path.basename(p)}: FAILED")
    else:
        pts = np.array(ordered)
        H = dlt_homography(np.array([[c, r] for r in range(6) for c in range(9)]), pts)
        errs = []
        for c in range(9):
            for r in range(6):
                p_id = np.array([c,r,1.0])
                p_im = H @ p_id
                p_im /= p_im[2]
                errs.append(np.linalg.norm(p_im[:2] - pts[r*9+c]))
        print(f"{os.path.basename(p)}: OK. Max err = {np.max(errs):.2f}")
