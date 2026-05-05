import os
import numpy as np
from src.feature_extractor import load_image_gray, harris_response, detect_corners, subpixel_refine
from src.math_core import dlt_homography

def global_grid_order(corners, grid_shape):
    cols, rows = grid_shape
    expected = cols * rows
    if len(corners) < expected:
        return None
        
    pts = np.array(corners, dtype=np.float64)
    
    # 1. Find 4 extremal points (candidates for corners of the checkerboard)
    # Using projections along multiple angles to find the boundary
    angles = np.linspace(0, 2*np.pi, 36)
    extremal = []
    for a in angles:
        v = np.array([np.cos(a), np.sin(a)])
        projs = pts @ v
        idx = np.argmax(projs)
        extremal.append(pts[idx])
    
    # Keep unique points
    extremal = np.unique(extremal, axis=0)
    
    # If we don't have at least 4, fallback
    if len(extremal) < 4:
        return None
        
    # Find 4 points that maximize the quadrilateral area
    import itertools
    best_quad = None
    best_area = 0
    
    def quad_area(q):
        # q is 4x2
        x = q[:, 0]
        y = q[:, 1]
        # Shoelace formula
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    for quad in itertools.combinations(extremal, 4):
        quad = np.array(quad)
        
        # Sort quad cyclically to form a polygon
        center = np.mean(quad, axis=0)
        angles = np.arctan2(quad[:,1]-center[1], quad[:,0]-center[0])
        quad = quad[np.argsort(angles)]
        
        area = quad_area(quad)
        if area > best_area:
            best_area = area
            best_quad = quad
            
    if best_quad is None:
        return None
        
    # best_quad is sorted cyclically (e.g. top-left, top-right, bottom-right, bottom-left)
    # We don't know which is which. Let's try the 4 cyclic shifts and optional reverse (8 permutations)
    ideal_corners = np.array([
        [0, 0],
        [cols - 1, 0],
        [cols - 1, rows - 1],
        [0, rows - 1]
    ], dtype=np.float64)
    
    ideal_grid = np.array([[c, r] for r in range(rows) for c in range(cols)], dtype=np.float64)
    ideal_grid_h = np.hstack([ideal_grid, np.ones((len(ideal_grid), 1))])
    
    best_ordered = None
    best_err = np.inf
    
    # Test permutations
    perms = [
        best_quad,
        np.roll(best_quad, 1, axis=0),
        np.roll(best_quad, 2, axis=0),
        np.roll(best_quad, 3, axis=0),
        best_quad[::-1],
        np.roll(best_quad[::-1], 1, axis=0),
        np.roll(best_quad[::-1], 2, axis=0),
        np.roll(best_quad[::-1], 3, axis=0),
    ]
    
    for q in perms:
        try:
            H = dlt_homography(ideal_corners, q)
        except:
            continue
            
        proj = (H @ ideal_grid_h.T).T
        z = proj[:, 2:]
        z[np.abs(z) < 1e-8] = 1e-8
        proj = proj[:, :2] / z
        
        # Snap with uniqueness
        err_sum = 0
        ordered = []
        used = set()
        
        # Fast vectorized distance matrix
        diff = proj[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.linalg.norm(diff, axis=2)
        
        valid = True
        for i in range(len(proj)):
            # Ignore used points
            dist_mat[i, list(used)] = np.inf
            b_match = np.argmin(dist_mat[i])
            min_dist = dist_mat[i, b_match]
            
            # If the closest point is too far, this is a bad homography
            # Using a dynamic threshold based on quad area
            thresh = np.sqrt(best_area) / max(cols, rows)
            if min_dist > thresh:
                valid = False
                break
                
            used.add(b_match)
            ordered.append((float(pts[b_match, 0]), float(pts[b_match, 1])))
            err_sum += min_dist
            
        if valid and err_sum < best_err:
            best_err = err_sum
            best_ordered = ordered

    if best_ordered is not None:
        # Enforce Canonical Orientation
        p_first = best_ordered[0]
        p_last = best_ordered[-1]
        
        dy = p_last[1] - p_first[1]
        dx = p_last[0] - p_first[0]
        
        if dy < -10 or (abs(dy) <= 10 and dx < 0):
            best_ordered = best_ordered[::-1]
            
        return best_ordered
    return None

import glob
paths = glob.glob("data/raw_checkerboards/*.jpg")
if paths:
    img = load_image_gray(paths[0])
    resp = harris_response(img)
    raw = detect_corners(resp)
    refined = subpixel_refine(img, raw)
    print("Found", len(refined), "corners")
    ordered = global_grid_order(refined, (11, 17))
    if ordered:
        print("Ordered size:", len(ordered))
        print("First:", ordered[0])
        print("Last:", ordered[-1])
    else:
        print("Failed to order")
