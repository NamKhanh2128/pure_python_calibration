import numpy as np
import itertools
from src.math_core import dlt_homography

def find_4_corners_pure_numpy(pts):
    hull_candidates = set()
    angles = np.linspace(0, np.pi/2, 10)
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        proj_x = pts[:,0]*c + pts[:,1]*s
        proj_y = -pts[:,0]*s + pts[:,1]*c
        hull_candidates.add(np.argmin(proj_x))
        hull_candidates.add(np.argmax(proj_x))
        hull_candidates.add(np.argmin(proj_y))
        hull_candidates.add(np.argmax(proj_y))
        
    candidates = list(hull_candidates)
    max_area = 0
    best_quad = None
    
    for quad_idx in itertools.combinations(candidates, 4):
        quad = pts[list(quad_idx)]
        centroid = quad.mean(axis=0)
        angles_c = np.arctan2(quad[:,1] - centroid[1], quad[:,0] - centroid[0])
        quad = quad[np.argsort(angles_c)]
        
        x, y = quad[:,0], quad[:,1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        if area > max_area:
            max_area = area
            best_quad = quad
            
    return best_quad

def order_checkerboard_corners_robust(corners, grid_shape):
    cols, rows = grid_shape
    expected = cols * rows
    if len(corners) < expected: return None
    
    pts = np.array(corners, dtype=np.float64)[:expected]
    
    # 1. Find 4 extremal corners (unordered but cyclic)
    img_4 = find_4_corners_pure_numpy(pts)
    if img_4 is None: return None
    
    # 2. Define ideal 4 corners
    ideal_4 = np.array([
        [0, 0], 
        [cols - 1, 0], 
        [cols - 1, rows - 1], 
        [0, rows - 1]
    ], dtype=np.float64)
    
    # 3. Create full ideal grid
    ideal_grid = np.array([[c, r] for r in range(rows) for c in range(cols)], dtype=np.float64)
    ones = np.ones((len(ideal_grid), 1))
    ideal_grid_h = np.hstack([ideal_grid, ones])
    
    best_err = np.inf
    best_ordered = None
    
    # Try all 8 cyclic permutations/reflections of the 4 image corners
    for i in range(4):
        for flip in [False, True]:
            permuted_img_4 = np.roll(img_4, i, axis=0)
            if flip:
                permuted_img_4 = permuted_img_4[::-1]
                
            try:
                H = dlt_homography(ideal_4, permuted_img_4)
            except np.linalg.LinAlgError:
                continue
                
            # Project all ideal points
            proj = (H @ ideal_grid_h.T).T
            proj = proj[:, :2] / proj[:, 2:]
            
            # For each projected point, find distance to nearest detected point
            err = 0
            ordered = []
            used = set()
            
            for p in proj:
                dists = np.linalg.norm(pts - p, axis=1)
                # mask used
                for u in used:
                    dists[u] = np.inf
                best_idx = np.argmin(dists)
                err += dists[best_idx]
                used.add(best_idx)
                ordered.append(tuple(pts[best_idx]))
                
            if err < best_err:
                best_err = err
                best_ordered = ordered

    # Since subpixel refined points should match perfectly, mean err should be < 5px
    if best_err / expected > 10.0:
        print(f"Failed. Best mean error was {best_err / expected:.2f} px")
        return None
        
    return best_ordered

import os
from src.feature_extractor import load_image_gray, harris_response, detect_corners, subpixel_refine

IMG_DIR = "data/raw_checkerboards"

print("Testing all images...")
paths = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR)]
for p in paths:
    img = load_image_gray(p)
    resp = harris_response(img)
    raw = detect_corners(resp)
    ref = subpixel_refine(img, raw)
    ordered = order_checkerboard_corners_robust(ref, (9, 6))
    if ordered is None:
        print(f"{os.path.basename(p)}: FAILED ordering")
    else:
        print(f"{os.path.basename(p)}: OK")

