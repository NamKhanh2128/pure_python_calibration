import os
import numpy as np
from src.feature_extractor import *
from src.math_core import dlt_homography

IMG_DIR = "data/raw_checkerboards"
GRID_SHAPE = (9, 6)
SQUARE_SIZE = 25.0

print("Checking homographies...")
img_pts_list, obj_pts_list, valid_paths, size = extract_all_views(
    IMG_DIR, GRID_SHAPE, SQUARE_SIZE
)

for img_path, img_pts, obj_pts in zip(valid_paths, img_pts_list, obj_pts_list):
    # compute homography
    H = dlt_homography(obj_pts[:, :2], img_pts)
    
    # Check projection error: H @ obj_pts -> should be close to img_pts
    errs = []
    for i in range(len(obj_pts)):
        p_w = np.array([obj_pts[i,0], obj_pts[i,1], 1.0])
        p_i = H @ p_w
        p_i /= p_i[2]
        
        err = np.linalg.norm(p_i[:2] - img_pts[i])
        errs.append(err)
    
    mean_err = np.mean(errs)
    max_err = np.max(errs)
    print(f"{os.path.basename(img_path)}: Mean DLT err = {mean_err:.3f} px, Max = {max_err:.3f} px")
    if max_err > 5.0:
        print(f"   >>> DLT failed or ordering is completely broken in this image!!")
