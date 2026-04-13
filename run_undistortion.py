"""
run_undistortion.py — Apply R2 (inverse mapping + nearest-neighbour) to every
image in data/raw_scenes/ and write results to data/output_flat/.

Usage
-----
    python run_undistortion.py

Prerequisites
-------------
    Run run_calibration.py first to generate models/camera_params.json.
"""

import os
import json

import numpy as np

from src.undistorter import undistort_image_file


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = "models/camera_params.json"
INPUT_DIR  = "data/raw_scenes"
OUTPUT_DIR = "data/output_flat"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Image Undistortion  :  R2 (Inverse + Nearest-Neighbour)")
    print("=" * 60)

    # ── Load calibration model ────────────────────────────────────────────────
    if not os.path.isfile(MODEL_PATH):
        print(f"\n  ERROR: model not found at '{MODEL_PATH}'.")
        print("  Run  python run_calibration.py  first.")
        return

    with open(MODEL_PATH, "r") as f:
        model = json.load(f)

    K    = np.array(model["K"],    dtype=np.float64)
    dist =          model["dist"]                       # [k1, k2]

    print(f"\n  Model  : {MODEL_PATH}")
    print(f"  Pipeline : {model.get('pipeline', 'n/a')}")
    print(f"  RMSE   : {model.get('rmse_final_px', 'n/a')} px")
    print(f"  K =\n{K}")
    print(f"  dist   = k1={dist[0]:.6f},  k2={dist[1]:.6f}\n")

    # ── Collect input images ──────────────────────────────────────────────────
    SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    if not os.path.isdir(INPUT_DIR):
        print(f"  ERROR: input directory '{INPUT_DIR}' does not exist.")
        return

    files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED
    )
    if not files:
        print(f"  No supported images found in '{INPUT_DIR}'.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── R2 : undistort each image ─────────────────────────────────────────────
    for fname in files:
        in_path  = os.path.join(INPUT_DIR,  fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        undistort_image_file(in_path, out_path, K, dist)
        print(f"  [R2] {fname}  →  {out_path}")

    print(f"\n  → {len(files)} image(s) written to '{OUTPUT_DIR}/'")
    print("=" * 60)


if __name__ == "__main__":
    main()
