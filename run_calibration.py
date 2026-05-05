"""
run_calibration.py — Orchestrate the full F1 -> L1 -> O1 calibration pipeline.

Usage
-----
    python run_calibration.py

Configuration
-------------
Edit the CONFIGURATION block below to match your checkerboard and images.
"""

import os
import json

from PIL import Image, ImageDraw
import numpy as np

from src.feature_extractor import extract_all_views
from src.math_core import (
    dlt_homography,
    solve_intrinsics_from_homographies,
    compute_extrinsics,
    init_distortion,
)
from src.optimizer import (
    pack_params,
    unpack_params,
    lm_optimize,
    reprojection_rmse,
)
from src.undistorter import undistort_image_file


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — edit these to match your setup
# ─────────────────────────────────────────────────────────────────────────────

# Directory that contains the checkerboard calibration images
CHECKERBOARD_DIR = "data/raw_checkerboards"

# Where to write the calibration result
MODEL_OUT = "models/camera_params.json"

# Where to save the output images with detected corners
DEBUG_OUT_DIR = "data/output_checkerboards"

# Inner-corner grid: (number of cols, number of rows)
# Example: a 10×7 board has (9, 6) inner corners
GRID_SHAPE  = (11, 17)

# Physical size of one checkerboard square (mm, or any consistent unit)
SQUARE_SIZE = 9.0

# Harris corner detector settings
HARRIS_K      = 0.04    # sensitivity — range [0.04, 0.06]
HARRIS_SIGMA  = 2.0     # Gaussian smoothing for structure tensor
THRESH_REL    = 0.05    # NMS threshold as fraction of peak response
MIN_DIST      = 30      # minimum separation (px) between corners

# Adam optimiser settings
ADAM_LR       = 1e-3    # learning rate
ADAM_ITER     = 500     # maximum iterations
ADAM_LOG      = 50      # print every N iterations


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Camera Calibration  :  F1 -> L1 -> O1")
    print("=" * 60)

    # ── F1 : Harris corner detection ─────────────────────────────────────────
    print(f"\n[F1] Harris Corner Detection  ({CHECKERBOARD_DIR})")
    img_pts_list, obj_pts_list, valid_paths, image_size = extract_all_views(
        image_dir    = CHECKERBOARD_DIR,
        grid_shape   = GRID_SHAPE,
        square_size  = SQUARE_SIZE,
        harris_k     = HARRIS_K,
        harris_sigma = HARRIS_SIGMA,
        threshold_rel= THRESH_REL,
        min_dist     = MIN_DIST,
    )
    n_views = len(img_pts_list)
    print(f"\n  -> {n_views} valid view(s) accepted")

    if n_views < 3:
        print("\n  ERROR: Need at least 3 valid checkerboard views — aborting.")
        return

    # ── F1.5: Draw and save detected corners ─────────────────────────────────
    print(f"\n[F1.5] Saving visualized corners to {DEBUG_OUT_DIR}")
    os.makedirs(DEBUG_OUT_DIR, exist_ok=True)
    for pts, path in zip(img_pts_list, valid_paths):
        out_name = "detected_" + os.path.basename(path)
        out_path = os.path.join(DEBUG_OUT_DIR, out_name)
        
        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        cols, rows = GRID_SHAPE
        
        # Draw lines connecting the rows
        for r in range(rows):
            for c in range(cols - 1):
                idx1 = r * cols + c
                idx2 = r * cols + c + 1
                pt1 = tuple(pts[idx1])
                pt2 = tuple(pts[idx2])
                draw.line([pt1, pt2], fill=(255, 0, 0), width=3)
                
        # Draw lines connecting the columns
        for c in range(cols):
            for r in range(rows - 1):
                idx1 = r * cols + c
                idx2 = (r + 1) * cols + c
                pt1 = tuple(pts[idx1])
                pt2 = tuple(pts[idx2])
                draw.line([pt1, pt2], fill=(0, 255, 0), width=3)
                
        # Draw circles at each corner
        for idx, pt in enumerate(pts):
            x, y = pt
            ratio = idx / (cols * rows)
            color = (int(255 * ratio), int(255 * (1 - ratio)), 255)
            r_circle = 4
            draw.ellipse([x - r_circle, y - r_circle, x + r_circle, y + r_circle], fill=color, outline=(0, 0, 0))
            
        img.save(out_path)
    print("  -> Done saving visualisations")

    # ── L1 : SVD — per-view homographies ─────────────────────────────────────
    print("\n[L1] SVD — Homographies (DLT, normalised)")
    Hs = []
    for img_pts, obj_pts in zip(img_pts_list, obj_pts_list):
        H = dlt_homography(obj_pts[:, :2], img_pts)   # src = world XY (z=0)
        Hs.append(H)
    print(f"  -> {len(Hs)} homograph(ies) computed")

    # ── L1 : SVD — intrinsic matrix (Zhang's V·b = 0) ────────────────────────
    print("\n[L1] SVD — Intrinsics  (Zhang's method)")
    K = solve_intrinsics_from_homographies(Hs)
    print(f"  -> K =\n{K}")

    # ── L1 : initial extrinsics ───────────────────────────────────────────────
    rvecs, ts = [], []
    for H in Hs:
        rv, t = compute_extrinsics(K, H)
        rvecs.append(rv)
        ts.append(t)

    # ── L1 : linear distortion init ───────────────────────────────────────────
    print("\n[L1] Linear init — Distortion  (least-squares)")
    dist_init = init_distortion(K, Hs, obj_pts_list, img_pts_list)
    print(f"  -> k1={dist_init[0]:.6f},  k2={dist_init[1]:.6f}")

    rmse_init = reprojection_rmse(
        pack_params(K, dist_init, rvecs, ts),
        obj_pts_list, img_pts_list,
    )
    print(f"  -> Initial RMSE = {rmse_init:.4f} px")

    # ── O1 : Levenberg-Marquardt optimization ─────────────────────────────────
    print("\n[O1] Levenberg-Marquardt — Refinement")
    p0      = pack_params(K, dist_init, rvecs, ts)
    p_opt, loss_hist = lm_optimize(
        p0, obj_pts_list, img_pts_list,
        max_iter  = 50,
        log_every = 10,
    )
    K_opt, dist_opt, rvecs_opt, ts_opt = unpack_params(p_opt, n_views)
    rmse_final = reprojection_rmse(p_opt, obj_pts_list, img_pts_list)

    print(f"\n  -> Final RMSE    = {rmse_final:.4f} px")
    print(f"  -> K_opt =\n{K_opt}")
    print(f"  -> dist_opt = k1={dist_opt[0]:.6f},  k2={dist_opt[1]:.6f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model = {
        "pipeline"      : "F1(Harris) -> L1(SVD) -> O1(Adam GD)",
        "grid_shape"    : list(GRID_SHAPE),
        "square_size_mm": SQUARE_SIZE,
        "image_size_wh" : list(image_size) if image_size else None,
        "n_views"       : n_views,
        "rmse_init_px"  : round(float(rmse_init),  6),
        "rmse_final_px" : round(float(rmse_final), 6),
        "K"             : K_opt.tolist(),
        "dist"          : [float(d) for d in dist_opt],
    }
    with open(MODEL_OUT, "w") as f:
        json.dump(model, f, indent=2)

    print(f"\n  -> Model saved -> {MODEL_OUT}")
    print("=" * 60)

    # ── Automatic Undistortion ────────────────────────────────────────────────
    INPUT_DIR  = "data/raw_scenes"
    OUTPUT_DIR = "data/output_flat"
    
    SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    if os.path.isdir(INPUT_DIR):
        print("\n" + "=" * 60)
        print("  Image Undistortion  :  R2 (Inverse + Nearest-Neighbour)")
        print("=" * 60)
        
        files = sorted(f for f in os.listdir(INPUT_DIR) if os.path.splitext(f)[1].lower() in SUPPORTED)
        if files:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            for fname in files:
                in_path  = os.path.join(INPUT_DIR,  fname)
                out_path = os.path.join(OUTPUT_DIR, fname)
                undistort_image_file(in_path, out_path, K_opt, dist_opt)
                print(f"  [R2] {fname}  →  {out_path}")
            print(f"\n  → {len(files)} image(s) written to '{OUTPUT_DIR}/'")
            print("=" * 60)
        else:
            print(f"  No supported images found in '{INPUT_DIR}'. Skipping undistortion.")
    else:
        print(f"\n  [INFO] Input directory '{INPUT_DIR}' does not exist. Skipping undistortion.")


if __name__ == "__main__":
    main()
