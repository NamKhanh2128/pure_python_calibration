# Pure-Python Camera Calibration

> **Stack:** F1 (Harris) → L1 (SVD) → O1 (Gradient Descent) → R2 (Inverse + Nearest)
>
> A complete Zhang's camera calibration pipeline implemented from scratch using **NumPy only** — no OpenCV calibration functions, no SciPy optimizers.

## Overview

This project implements **Zhang's planar camera calibration method** (Zhang 2000) entirely in pure Python + NumPy. Given photographs of a planar checkerboard, it computes:
- **Intrinsics:** Focal lengths (`fx`, `fy`) and Principal point (`cx`, `cy`)
- **Distortion:** Radial distortion coefficients (`k1`, `k2`)
- **Extrinsics:** Rotation and translation vectors per image

The calibrated parameters are saved to a JSON model and can be used to undistort arbitrary scene images.

## Structure

```text
pure_python_calibration/
├── data/
│   ├── raw_checkerboards/     ← Put calibration images here
│   ├── raw_scenes/            ← Put images to undistort here
│   └── output_flat/           ← Corrected images show up here
├── models/
│   └── camera_params.json     ← Saved calibration result
├── src/
│   ├── feature_extractor.py   ← F1: Harris corner detector
│   ├── math_core.py           ← L1: SVD math primitives
│   ├── optimizer.py           ← O1: Adam gradient descent
│   └── undistorter.py         ← R2: Inverse mapping undistortion
├── run_calibration.py         ← Run this to calibrate
└── run_undistortion.py        ← Run this to undistort images
```

## Quick Start

### 1. Requirements
Requires **Python 3.9+**. Install dependencies:
```bash
pip install -r requirements.txt
```
*(Dependencies: `numpy>=1.24`, `Pillow>=10.0`, `matplotlib>=3.7`, `tqdm>=4.0`)*

### 2. Configure Checkerboard
Edit `run_calibration.py` to match your physical checkerboard:
```python
GRID_SHAPE  = (8, 8)    # (cols, rows) of INNER corners
SQUARE_SIZE = 18.0      # physical side length of one square in mm
```

### 3. Calibrate
Put at least 3 (recommended 10-20) images in `data/raw_checkerboards/`, then run:
```bash
python run_calibration.py
```
This will detect corners, compute the initial estimation, refine via gradient descent, and save `models/camera_params.json`.

### 4. Undistort
Place scene images in `data/raw_scenes/` and run:
```bash
python run_undistortion.py
```
The undistorted results will be written to `data/output_flat/`.

## Pipeline Architecture

1. **[F1] Harris Corner Detection** (`feature_extractor.py`): Detects and refines checkerboard corners. Uses spatial Sobel gradients, structure tensors, Non-Maximum Suppression, sub-pixel refinement, and RANSAC topology ordering.
2. **[L1] SVD Linear Solver** (`math_core.py`): Computes homographies via normalized DLT. Solves for Zhang's intrinsic matrix (V·b = 0) and initializes distortion via linear least-squares.
3. **[O1] Adam Gradient Descent** (`optimizer.py`): Jointly refines all parameters by minimizing reprojection RMSE using central finite-difference gradients and Adam.
4. **[R2] Image Undistortion** (`undistorter.py`): Uses an inverse mapping technique with nearest-neighbour interpolation to remove radial distortion without leaving "holes".

## References
1. **Z. Zhang** — *"A Flexible New Technique for Camera Calibration"*, IEEE TPAMI, 2000.
2. **D. Kingma & J. Ba** — *"Adam: A Method for Stochastic Optimization"*, ICLR, 2015.
