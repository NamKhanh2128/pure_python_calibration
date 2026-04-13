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
    adam_optimize,
    reprojection_rmse,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — edit these to match your setup
# ─────────────────────────────────────────────────────────────────────────────

# Directory that contains the checkerboard calibration images
CHECKERBOARD_DIR = "data/raw_checkerboards"

# Where to write the calibration result
MODEL_OUT = "models/camera_params.json"

# Inner-corner grid: (number of cols, number of rows)
# Example: a 10×7 board has (9, 6) inner corners
GRID_SHAPE  = (8, 8)

# Physical size of one checkerboard square (mm, or any consistent unit)
SQUARE_SIZE = 18.0

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

    # ── O1 : Adam gradient descent ────────────────────────────────────────────
    print("\n[O1] Adam Gradient Descent — Refinement")
    p0      = pack_params(K, dist_init, rvecs, ts)
    p_opt, loss_hist = adam_optimize(
        p0, obj_pts_list, img_pts_list,
        lr        = ADAM_LR,
        max_iter  = ADAM_ITER,
        log_every = ADAM_LOG,
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


if __name__ == "__main__":
    main()
