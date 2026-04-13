# Pure-Python Camera Calibration

> **Stack:** F1 (Harris) → L1 (SVD) → O1 (Gradient Descent) → R2 (Inverse + Nearest)
>
> A complete Zhang's camera calibration pipeline implemented from scratch using **NumPy only** — no OpenCV calibration functions, no SciPy optimizers.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Requirements & Installation](#3-requirements--installation)
4. [How to Run](#4-how-to-run)
5. [Pipeline Architecture](#5-pipeline-architecture)
6. [Stage F1 — Harris Corner Detector](#6-stage-f1--harris-corner-detector)
7. [Stage L1 — SVD Linear Solver](#7-stage-l1--svd-linear-solver)
8. [Stage O1 — Adam Gradient Descent](#8-stage-o1--adam-gradient-descent)
9. [Stage R2 — Inverse Mapping + Nearest Neighbour](#9-stage-r2--inverse-mapping--nearest-neighbour)
10. [Output Files](#10-output-files)
11. [Configuration Reference](#11-configuration-reference)
12. [Method & Formula Reference](#12-method--formula-reference)

---

## 1. Project Overview

This project implements **Zhang's planar camera calibration method** (Zhang 2000) entirely in pure Python + NumPy. Given a set of photographs of a planar checkerboard pattern taken from different angles, it computes:

| Output | Symbol | Description |
|---|---|---|
| Focal lengths | `fx`, `fy` | in pixels |
| Principal point | `cx`, `cy` | optical centre in pixels |
| Radial distortion | `k1`, `k2` | Brown-Conrady model |
| Per-view rotation | `R_i` | Rodrigues vector per image |
| Per-view translation | `t_i` | in calibration-target units (mm) |

The calibrated parameters are saved to `models/camera_params.json` and then used to undistort arbitrary scene images.

---

## 2. Repository Structure

```
pure_python_calibration/
│
├── data/
│   ├── raw_checkerboards/     ← checkerboard calibration images (input)
│   ├── raw_scenes/            ← scene images to undistort (input)
│   └── output_flat/           ← undistorted output images (generated)
│
├── models/
│   └── camera_params.json     ← calibration result (generated)
│
├── src/
│   ├── __init__.py            ← package public API
│   ├── feature_extractor.py   ← F1: Harris corner detector
│   ├── math_core.py           ← L1: SVD math primitives
│   ├── optimizer.py           ← O1: Adam gradient descent
│   └── undistorter.py         ← R2: inverse mapping + nearest neighbour
│
├── run_calibration.py         ← entry point: F1 → L1 → O1 → save model
├── run_undistortion.py        ← entry point: load model → R2 → save images
├── requirements.txt
└── README.md
```

---

## 3. Requirements & Installation

**Python 3.9+** is required.

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
numpy>=1.24
Pillow>=10.0
matplotlib>=3.7
tqdm>=4.0
```

> No OpenCV, no SciPy, no PyTorch. All algorithms are implemented from scratch.

---

## 4. How to Run

### Step 1 — Prepare your data

```
data/raw_checkerboards/    ← put checkerboard images here (JPG, PNG, BMP…)
data/raw_scenes/           ← put scene images to undistort here
```

> You need **at least 3** checkerboard images taken from different angles and distances. 10–20 images from varied viewpoints are recommended for good accuracy.

### Step 2 — Configure checkerboard parameters

Open `run_calibration.py` and edit the `CONFIGURATION` block:

```python
GRID_SHAPE  = (9, 6)    # (cols, rows) of INNER corners
                         # e.g. a 10×7 square board has (9, 6) inner corners
SQUARE_SIZE = 25.0      # physical side length of one square in mm
```

> **Important:** `GRID_SHAPE` counts only *inner* corners (where 4 squares meet), not the outer border. A board with 10 columns and 7 rows of squares has `(9, 6)` inner corners.

### Step 3 — Run calibration

```bash
python run_calibration.py
```

**What happens:**
```
[F1] Harris Corner Detection
  → detects and refines checkerboard corners in every image

[L1] SVD — Homographies
  → computes one 3×3 homography per calibration view

[L1] SVD — Intrinsics
  → solves Zhang's V·b=0 system → K matrix

[L1] Linear init — Distortion
  → least-squares init for k1, k2

[O1] Adam Gradient Descent — Refinement
  → jointly refines all parameters to minimize reprojection RMSE

  → Model saved to models/camera_params.json
```

Expected console output:
```
============================================================
  Camera Calibration  :  F1 → L1 → O1
============================================================

[F1] Harris Corner Detection  (data/raw_checkerboards)
  [OK]   img01.jpg — 54 corners
  [OK]   img02.jpg — 54 corners
  ...

[L1] SVD — Homographies (DLT, normalised)
  → 12 homograph(ies) computed

[L1] SVD — Intrinsics  (Zhang's method)
  → K =
  [[1234.5    0.  640.2]
   [   0.  1231.8 359.7]
   [   0.     0.    1. ]]

[L1] Linear init — Distortion  (least-squares)
  → k1=-0.123456,  k2=0.012345
  → Initial RMSE = 2.3410 px

[O1] Adam Gradient Descent — Refinement
    [Adam] start  RMSE = 2.341000 px  (params=78, views=12)
    [Adam iter   50] RMSE = 0.812345 px
    [Adam iter  100] RMSE = 0.645231 px
    [Adam] converged at iter 183,  RMSE = 0.412567 px

  → Final RMSE    = 0.4126 px
  → Model saved → models/camera_params.json
```

### Step 4 — Undistort scene images

```bash
python run_undistortion.py
```

**What happens:**
```
[R2] Loads camera_params.json
  → undistorts every image in data/raw_scenes/
  → writes corrected images to data/output_flat/
```

---

## 5. Pipeline Architecture

```
data/raw_checkerboards/
        │
        ▼
╔══════════════════════════════════════════╗
║  F1 — Harris Corner Detection            ║  feature_extractor.py
║                                          ║
║  load_image_gray                         ║
║    → Sobel gradients  Ix, Iy             ║
║    → Harris response  R = det(M)−k·tr²   ║
║    → NMS + threshold  → raw corners      ║
║    → Sub-pixel refine → (x, y) ± 0.1px  ║
║    → PCA grid order   → row-major list   ║
╚══════════════════════════════════════════╝
        │  img_pts_list, obj_pts_list
        ▼
╔══════════════════════════════════════════╗
║  L1 — SVD Linear Solve                   ║  math_core.py
║                                          ║
║  dlt_homography  (per view)              ║
║    → normalized DLT → SVD → H (3×3)     ║
║                                          ║
║  solve_intrinsics_from_homographies      ║
║    → Zhang's V·b = 0 → SVD → K          ║
║                                          ║
║  init_distortion                         ║
║    → linear least-squares → k1, k2      ║
╚══════════════════════════════════════════╝
        │  K_init, dist_init, rvecs, ts
        ▼
╔══════════════════════════════════════════╗
║  O1 — Adam Gradient Descent              ║  optimizer.py
║                                          ║
║  Minimise RMSE = √(mean(‖proj−obs‖²))   ║
║  over [fx,fy,cx,cy,k1,k2, R_i, t_i…]   ║
║                                          ║
║  Gradient via central finite differences ║
║  Adam: m,v moments + bias correction     ║
╚══════════════════════════════════════════╝
        │  K_opt, dist_opt
        ▼
  models/camera_params.json
        │
        ▼
╔══════════════════════════════════════════╗
║  R2 — Inverse Mapping + Nearest Neighbour║  undistorter.py
║                                          ║
║  For each dst pixel (u,v):               ║
║    normalize → distort → back-project    ║
║    round → nearest-neighbour copy        ║
╚══════════════════════════════════════════╝
        │
        ▼
  data/output_flat/  (undistorted images)
```

---

## 6. Stage F1 — Harris Corner Detector

**File:** `src/feature_extractor.py`

### 6.1 Sobel Gradients

Horizontal and vertical image derivatives computed via separable Sobel operators:

$$I_x = \begin{bmatrix}-1 & 0 & 1\end{bmatrix} \otimes \frac{1}{4}\begin{bmatrix}1 & 2 & 1\end{bmatrix}^T$$

$$I_y = \frac{1}{4}\begin{bmatrix}1 & 2 & 1\end{bmatrix} \otimes \begin{bmatrix}-1 & 0 & 1\end{bmatrix}^T$$

The `[1 2 1]/4` factor is a mild Gaussian smooth in the perpendicular direction, making the derivative more robust to noise than a plain finite difference.

### 6.2 Structure Tensor

For each pixel, the **2×2 structure tensor** (second-moment matrix) is:

$$M = \sum_{\text{window}} w(p) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$

where `w(p)` is a Gaussian window of width `σ`.  The eigenvalues λ₁, λ₂ of M describe the local gradient distribution:

| Case | λ₁, λ₂ | Interpretation |
|---|---|---|
| Flat region | both ≈ 0 | no feature |
| Edge | one large, one ≈ 0 | edge |
| **Corner** | **both large** | **Harris responds** |

### 6.3 Harris Response Score

Instead of computing eigenvalues, Harris (1988) uses:

$$R = \det(M) - k \cdot \text{tr}(M)^2 = \lambda_1\lambda_2 - k(\lambda_1+\lambda_2)^2$$

- `k ∈ [0.04, 0.06]` (sensitivity parameter, default `0.04`)
- Large positive `R` → corner. Negative `R` → edge. Small |R| → flat.

### 6.4 Non-Maximum Suppression (NMS)

1. Compute `threshold = threshold_rel × max(R)`
2. Keep pixels where `R > threshold`
3. Sort by descending score; greedily accept peaks, suppressing all pixels within `min_dist` pixels

### 6.5 Sub-Pixel Refinement

At a true corner, the gradient is perpendicular to the vector from the corner to any local point:

$$\nabla I(p)^T \cdot (p - q) = 0 \quad \forall p \text{ in window}$$

This leads to the 2×2 linear system (solved each iteration):

$$\underbrace{\begin{bmatrix} \sum w I_x^2 & \sum w I_x I_y \\ \sum w I_x I_y & \sum w I_y^2 \end{bmatrix}}_{A} \cdot \delta = \underbrace{\begin{bmatrix} \sum w I_x(I_x \Delta x + I_y \Delta y) \\ \sum w I_y(I_x \Delta x + I_y \Delta y) \end{bmatrix}}_{b}$$

Solved by **Cramer's rule:**

$$\delta_x = \frac{A_{11} b_0 - A_{01} b_1}{\det(A)}, \quad \delta_y = \frac{A_{00} b_1 - A_{01} b_0}{\det(A)}$$

Corner updated: `(x, y) ← (x + δx, y + δy)` until `‖δ‖ < ε`.

### 6.6 PCA-Based Corner Ordering

Corners from NMS are unordered but must match the world-point array row-by-row:

1. Compute 2×2 covariance matrix of the detected corner cloud
2. EVD gives the long axis (largest eigenvector) and short axis
3. Sort corners by short-axis projection → `rows` bands (checkerboard rows)
4. Within each band, sort by long-axis projection → left-to-right columns

### 6.7 World Points

$$X_j = \begin{bmatrix} c \cdot s \\ r \cdot s \\ 0 \end{bmatrix}, \quad (c, r) \in \{0,\ldots,\text{cols}-1\} \times \{0,\ldots,\text{rows}-1\}$$

where `s = square_size` (mm). Z=0 for all points (planar target).

---

## 7. Stage L1 — SVD Linear Solver

**File:** `src/math_core.py`

### 7.1 Point Normalization

Before DLT, normalize each point set isotropically (Hartley 1997):

$$T = \begin{bmatrix} s & 0 & -s \cdot c_x \\ 0 & s & -s \cdot c_y \\ 0 & 0 & 1 \end{bmatrix}, \quad s = \frac{\sqrt{2}}{\bar{d}}$$

where `c_x, c_y` = centroid and `d̄` = mean distance from centroid. After normalization, mean distance from origin = √2. This is applied to both source and destination points independently.

### 7.2 DLT Homography

Each world-to-image correspondence `(x,y) ↔ (u,v)` contributes 2 rows to the design matrix:

$$A_{2i} = \begin{bmatrix} -x & -y & -1 & 0 & 0 & 0 & ux & uy & u \end{bmatrix}$$

$$A_{2i+1} = \begin{bmatrix} 0 & 0 & 0 & -x & -y & -1 & vx & vy & v \end{bmatrix}$$

The homography `h = vec(H)` satisfies `A · h = 0`. Solution:

$$h = \text{last row of } V^T \quad \text{from SVD: } A = U \Sigma V^T$$

Denormalize: `H_real = T_dst^{-1} · H_norm · T_src`

### 7.3 Zhang's Intrinsic Recovery (V·b = 0)

The Image of the Absolute Conic (IAC) `B = K^{-T} K^{-1}` is symmetric with 6 unknowns:

$$b = [B_{11}, B_{12}, B_{22}, B_{13}, B_{23}, B_{33}]^T$$

Each homography provides the 6-element constraint vector:

$$v_{ij} = \begin{bmatrix} H_{0i}H_{0j} \\ H_{0i}H_{1j}+H_{1i}H_{0j} \\ H_{1i}H_{1j} \\ H_{2i}H_{0j}+H_{0i}H_{2j} \\ H_{2i}H_{1j}+H_{1i}H_{2j} \\ H_{2i}H_{2j} \end{bmatrix}$$

Two constraints per view: `v₀₁ · b = 0` and `(v₀₀ - v₁₁) · b = 0`.

Stack for all n views: `V · b = 0` (V is 2n×6).

**Solution:** `b = last column of V` from SVD.

**Extract K from b** (Zhang Appendix B):

$$c_y = \frac{B_{12}B_{13} - B_{11}B_{23}}{B_{11}B_{22} - B_{12}^2}$$

$$\lambda = B_{33} - \frac{B_{13}^2 + c_y(B_{12}B_{13} - B_{11}B_{23})}{B_{11}}$$

$$f_x = \sqrt{\lambda / B_{11}}, \quad f_y = \sqrt{\frac{\lambda B_{11}}{B_{11}B_{22}-B_{12}^2}}$$

$$\gamma = \frac{-B_{12} f_x^2 f_y}{\lambda}, \quad c_x = \frac{\gamma c_y}{f_y} - \frac{B_{13} f_x^2}{\lambda}$$

$$K = \begin{bmatrix} f_x & \gamma & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

### 7.4 Rodrigues Rotation

**rvec → R** (Rodrigues' formula):

$$\theta = \|\text{rvec}\|, \quad \hat{k} = \frac{\text{rvec}}{\theta}$$

$$[\hat{k}]_\times = \begin{bmatrix} 0 & -k_z & k_y \\ k_z & 0 & -k_x \\ -k_y & k_x & 0 \end{bmatrix}$$

$$R = I + \sin\theta \cdot [\hat{k}]_\times + (1 - \cos\theta) \cdot [\hat{k}]_\times^2$$

**R → rvec** (inverse):

$$\theta = \arccos\!\left(\frac{\text{tr}(R)-1}{2}\right), \quad \text{rvec} = \frac{\theta}{2\sin\theta}\begin{bmatrix}R_{32}-R_{23}\\R_{13}-R_{31}\\R_{21}-R_{12}\end{bmatrix}$$

### 7.5 Extrinsics Recovery

From `H ≈ K · [r₁ | r₂ | t]`:

$$\lambda = \frac{1}{\|K^{-1} h_1\|}, \quad r_1 = \lambda K^{-1} h_1, \quad r_2 = \lambda K^{-1} h_2$$

$$r_3 = r_1 \times r_2, \quad t = \lambda K^{-1} h_3$$

Project `[r₁|r₂|r₃]` onto SO(3) via SVD: `R = U·Vᵀ`

### 7.6 Forward Projection

$$P_c = R \cdot P_w + t$$

$$x_n = \frac{X_c}{Z_c}, \quad y_n = \frac{Y_c}{Z_c}$$

$$r^2 = x_n^2 + y_n^2, \quad \text{fac} = 1 + k_1 r^2 + k_2 r^4$$

$$x_d = x_n \cdot \text{fac}, \quad y_d = y_n \cdot \text{fac}$$

$$u = f_x x_d + c_x, \quad v = f_y y_d + c_y$$

### 7.7 Linear Distortion Initialization

Linearize distortion residual around `k1=k2=0`:

$$\Delta u \approx f_x \cdot x_n \cdot (k_1 r^2 + k_2 r^4)$$
$$\Delta v \approx f_y \cdot y_n \cdot (k_1 r^2 + k_2 r^4)$$

Design matrix row pair per point:

$$A_{\text{row}} = \begin{bmatrix} f_x x_n r^2 & f_x x_n r^4 \\ f_y y_n r^2 & f_y y_n r^4 \end{bmatrix}$$

Solve: `[k1, k2]ᵀ = argmin ‖A·x − b‖²` via `np.linalg.lstsq`.

---

## 8. Stage O1 — Adam Gradient Descent

**File:** `src/optimizer.py`

### 8.1 Parameter Vector

$$\theta = [\underbrace{f_x, f_y, c_x, c_y, k_1, k_2}_{\text{6 intrinsics}}, \underbrace{r_{x0}, r_{y0}, r_{z0}, t_{x0}, t_{y0}, t_{z0}}_{\text{view 0}}, \ldots]$$

Total length: `6 + 6·n_views`

### 8.2 Objective Function

$$\text{RMSE} = \sqrt{\frac{1}{2\sum N_i} \sum_{i,j} \left(\|\hat{p}_{ij}(\theta) - p_{ij}^{\text{obs}}\|^2\right)}$$

where `p̂_ij(θ)` is the projected position of world point j in view i, and `p_ij^obs` is the observed corner position.

### 8.3 Gradient via Central Finite Differences

$$\frac{\partial \text{RMSE}}{\partial \theta_i} \approx \frac{\text{RMSE}(\theta + \varepsilon e_i) - \text{RMSE}(\theta - \varepsilon e_i)}{2\varepsilon}$$

Second-order accurate: error is O(ε²). Default `ε = 1e-5`.

### 8.4 Adam Update Rule

At each iteration `t`:

| Step | Formula | Purpose |
|---|---|---|
| Gradient | `g = ∂RMSE/∂θ` | direction of steepest ascent |
| 1st moment | `m = β₁·m + (1−β₁)·g` | exponential moving average of g |
| 2nd moment | `v = β₂·v + (1−β₂)·g²` | exponential moving average of g² |
| Bias correct | `m̂ = m/(1−β₁ᵗ)` | unbias early iterations |
| Bias correct | `v̂ = v/(1−β₂ᵗ)` | unbias early iterations |
| **Update** | **`θ ← θ − α · m̂ / (√v̂ + ε)`** | **adaptive per-parameter step** |

**Default hyperparameters:**

| Parameter | Symbol | Default |
|---|---|---|
| Learning rate | α | `1e-3` |
| 1st-moment decay | β₁ | `0.9` |
| 2nd-moment decay | β₂ | `0.999` |
| Stability constant | ε | `1e-8` |
| Max iterations | — | `500` |
| Convergence tol | — | `1e-7` |

**Convergence:** Stop when `|RMSE[t] − RMSE[t-1]| < tol`.

---

## 9. Stage R2 — Inverse Mapping + Nearest Neighbour

**File:** `src/undistorter.py`

### 9.1 Why Inverse Mapping

**Forward mapping** (distorted src → corrected dst) leaves holes because distorted source pixels do not land on a regular grid. **Inverse (backward) mapping** iterates over every *destination* pixel and looks up where it came from in the source — every output pixel is always filled.

### 9.2 Algorithm (per destination pixel `(u, v)`)

**Step 1 — Normalize** (destination pixel → undistorted normalized plane):

$$x_n = \frac{u - c_x}{f_x}, \quad y_n = \frac{v - c_y}{f_y}$$

**Step 2 — Apply forward distortion** (find the corresponding *distorted* source location):

$$r^2 = x_n^2 + y_n^2$$
$$\text{fac} = 1 + k_1 r^2 + k_2 r^4$$
$$x_d = x_n \cdot \text{fac}, \quad y_d = y_n \cdot \text{fac}$$

**Step 3 — Back-project** to source pixel:

$$u_{\text{src}} = f_x \cdot x_d + c_x, \quad v_{\text{src}} = f_y \cdot y_d + c_y$$

**Step 4 — Nearest neighbour:**

$$u_s = \text{round}(u_{\text{src}}), \quad v_s = \text{round}(v_{\text{src}})$$

**Step 5 — Copy pixel:**

$$\text{dst}[v, u] = \begin{cases} \text{src}[v_s, u_s] & \text{if } 0 \le u_s < W \text{ and } 0 \le v_s < H \\ 0 \text{ (black)} & \text{otherwise} \end{cases}$$

### 9.3 Vectorized Implementation

All H×W destination pixels are processed simultaneously using NumPy meshgrids — no Python loops over pixels.

---

## 10. Output Files

### `models/camera_params.json`

```json
{
  "pipeline": "F1(Harris) -> L1(SVD) -> O1(Adam GD)",
  "grid_shape": [9, 6],
  "square_size_mm": 25.0,
  "image_size_wh": [1280, 720],
  "n_views": 12,
  "rmse_init_px": 2.341,
  "rmse_final_px": 0.413,
  "K": [
    [1234.5, 0.0, 640.2],
    [0.0, 1231.8, 359.7],
    [0.0, 0.0, 1.0]
  ],
  "dist": [-0.123456, 0.012345]
}
```

| Field | Description |
|---|---|
| `K` | 3×3 intrinsic matrix `[[fx, γ, cx],[0, fy, cy],[0, 0, 1]]` |
| `dist` | `[k1, k2]` radial distortion coefficients |
| `rmse_final_px` | Reprojection RMSE in pixels (target: < 1.0) |

### `data/output_flat/`

One corrected (undistorted) image per file in `data/raw_scenes/`, same filename.

---

## 11. Configuration Reference

### `run_calibration.py`

```python
CHECKERBOARD_DIR = "data/raw_checkerboards"  # input images
MODEL_OUT        = "models/camera_params.json"

GRID_SHAPE   = (9, 6)    # (cols, rows) INNER corners of your board
SQUARE_SIZE  = 25.0      # physical side length in mm

HARRIS_K      = 0.04     # Harris sensitivity [0.04 – 0.06]
HARRIS_SIGMA  = 2.0      # Gaussian σ for structure tensor
THRESH_REL    = 0.01     # NMS threshold = THRESH_REL × max(R)
MIN_DIST      = 10       # min distance (px) between accepted corners

ADAM_LR       = 1e-3     # Adam learning rate
ADAM_ITER     = 500      # max Adam iterations
ADAM_LOG      = 50       # print every N iterations
```

### `run_undistortion.py`

```python
MODEL_PATH = "models/camera_params.json"
INPUT_DIR  = "data/raw_scenes"
OUTPUT_DIR = "data/output_flat"
```

---

## 12. Method & Formula Reference

| Method | File | Formula / Operation |
|---|---|---|
| `load_image_gray` | `feature_extractor` | PIL "L" → float64 / 255 |
| `_gaussian_kernel_1d` | `feature_extractor` | g(x)=exp(−x²/2σ²) / Σg |
| `_conv2d_sep` | `feature_extractor` | kx along axis=1, ky along axis=0 |
| `_sobel_gradients` | `feature_extractor` | Ix=diff⊗smooth ; Iy=smooth⊗diff |
| `harris_response` | `feature_extractor` | R = det(M) − k·tr(M)² |
| `detect_corners` | `feature_extractor` | threshold + greedy NMS |
| `subpixel_refine` | `feature_extractor` | A·δ=b (2×2 Cramer solve) |
| `order_checkerboard_corners` | `feature_extractor` | PCA eigenvector → band sort |
| `make_object_points` | `feature_extractor` | (c·s, r·s, 0) grid |
| `extract_all_views` | `feature_extractor` | F1 driver (all images) |
| `normalize_points` | `math_core` | T: centroid→origin, mean_dist→√2 |
| `dlt_homography` | `math_core` | A·h=0 → SVD → H = T_dst⁻¹·Hn·T_src |
| `compute_v_ij` | `math_core` | v_ij = [h₀ᵢh₀ⱼ, h₀ᵢh₁ⱼ+h₁ᵢh₀ⱼ, …] |
| `solve_intrinsics_from_homographies` | `math_core` | V·b=0 → SVD → K (Zhang §B) |
| `rodrigues_to_R` | `math_core` | R = I+sin(θ)[k]×+(1−cos(θ))[k]×² |
| `R_to_rodrigues` | `math_core` | θ=arccos((tr−1)/2), axis from R−Rᵀ |
| `compute_extrinsics` | `math_core` | λ=1/‖K⁻¹h₁‖, r₃=r₁×r₂, SVD→SO(3) |
| `project_points` | `math_core` | Pc=R·Pw+t → divide by Zc → distort → K |
| `init_distortion` | `math_core` | Δu≈fx·xₙ·(k1r²+k2r⁴) → lstsq |
| `pack_params` | `optimizer` | [fx,fy,cx,cy,k1,k2, rvec₀,t₀, …] |
| `unpack_params` | `optimizer` | inverse of pack_params |
| `reprojection_error_vector` | `optimizer` | proj(θ) − obs → flat residuals |
| `reprojection_rmse` | `optimizer` | √(mean(r²)) in pixels |
| `_numerical_gradient` | `optimizer` | [f(p+ε)−f(p−ε)]/(2ε) per param |
| `adam_optimize` | `optimizer` | m,v EMA + bias correct + p−α·m̂/(√v̂+ε) |
| `_build_distortion_maps` | `undistorter` | meshgrid → normalize → distort → pixel |
| `undistort_image` | `undistorter` | inverse map + round + valid mask + gather |
| `undistort_image_file` | `undistorter` | load → undistort_image → save |

---

## References

1. **Z. Zhang** — *"A Flexible New Technique for Camera Calibration"*, IEEE TPAMI, 2000.
2. **R. Hartley & A. Zisserman** — *"Multiple View Geometry in Computer Vision"*, Cambridge, 2003.
   - §4.1: DLT Homography
   - §4.4: Point normalization
3. **C. Harris & M. Stephens** — *"A Combined Corner and Edge Detector"*, Alvey Vision Conf., 1988.
4. **J.-Y. Bouguet** — *"Pyramidal Implementation of the Lucas Kanade Feature Tracker"*, Intel, 2001. (Sub-pixel refinement)
5. **D. Kingma & J. Ba** — *"Adam: A Method for Stochastic Optimization"*, ICLR, 2015.
6. **D. Brown** — *"Decentering Distortion of Lenses"*, Photogrammetric Engineering, 1966. (Distortion model)
