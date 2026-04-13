"""
feature_extractor.py — F1 (Harris): Pure-NumPy Harris corner detector for
checkerboard calibration images.

Pipeline within this module:
  load_image_gray
      → harris_response        (structure tensor + Harris corner formula)
      → detect_corners         (threshold + greedy non-maximum suppression)
      → subpixel_refine        (iterative gradient-window least-squares)
      → order_checkerboard_corners  (PCA row/col sort → row-major grid)
  extract_all_views            (orchestrate all the above per directory)
"""

import os
import numpy as np
from PIL import Image


# ═════════════════════════════════════════════════════════════════════════════
# 1. IMAGE I/O
# ═════════════════════════════════════════════════════════════════════════════

def load_image_gray(path):
    """
    Load an image file and convert it to a float64 grayscale array in [0, 1].

    STEPS:
      1. PIL.Image.open reads the file (any format: JPEG, PNG, BMP, …).
      2. .convert("L") reduces to single-channel luminance using the ITU-R BT.601
         formula: L = 0.299·R + 0.587·G + 0.114·B.
      3. np.array casts to float64 in [0, 255], then divide by 255 → [0, 1].

    Returns
    -------
    img : (H, W) float64 numpy array, values in [0, 1]
    """
    # -- open file, force single-channel 8-bit grayscale ----------------------
    pil_img = Image.open(path).convert("L")      # "L" = luminance (grayscale)

    # -- cast to float64 then normalize pixel range from [0,255] to [0,1] -----
    # Dividing by 255.0 ensures gradients later are computed in a consistent scale.
    return np.array(pil_img, dtype=np.float64) / 255.0


def load_image_rgb(path):
    """
    Load an image as an uint8 RGB array  (H, W, 3).

    Used by undistort_image_file to load color images for R2 processing.

    Returns
    -------
    img : (H, W, 3) uint8 numpy array
    """
    # -- open file, force 3-channel RGB regardless of source format (RGBA, L, …)
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
# 2. SIGNAL-PROCESSING HELPERS  (internal)
# ═════════════════════════════════════════════════════════════════════════════

def _gaussian_kernel_1d(sigma, truncate=3.0):
    """
    Build a normalised 1-D Gaussian kernel.

    FORMULA:
      g(x) = exp(−x² / (2σ²))
      Then normalize so that Σ g(x) = 1  (no DC gain change after convolution).

    The kernel is truncated at ±(truncate·σ) pixels, which captures >99.7% of
    the Gaussian mass for truncate=3.

    Parameters
    ----------
    sigma    : standard deviation (controls smoothing width)
    truncate : how many sigma to keep on each side (default 3.0)

    Returns
    -------
    k : 1-D float64 array of length  2·round(truncate·sigma)+1
    """
    # -- radius: number of pixels to each side of center ----------------------
    radius = int(truncate * sigma + 0.5)     # round half-up

    # -- sample positions: −radius, −radius+1, …, 0, …, +radius ---------------
    x = np.arange(-radius, radius + 1, dtype=np.float64)

    # -- Gaussian values: g(x) = exp(−x²/(2σ²)) --------------------------------
    k = np.exp(-0.5 * (x / sigma) ** 2)

    # -- normalize so kernel sums to 1 (energy preserving) --------------------
    return k / k.sum()


def _conv2d_sep(img, kx, ky=None):
    """
    2-D separable convolution: apply kx along each row, then ky along each column.

    WHY SEPARABLE: A 2-D Gaussian G(x,y) = g(x)·g(y) can be applied as two
    sequential 1-D convolutions, reducing cost from O(k²N) to O(2kN).

    Parameters
    ----------
    img : (H, W) float64 image
    kx  : 1-D kernel applied along columns (axis=1 = horizontal direction)
    ky  : 1-D kernel applied along rows (axis=0 = vertical direction).
           If None, kx is reused for both directions (isotropic Gaussian).

    Returns
    -------
    out : (H, W) float64 filtered image
    """
    # -- default: same kernel in both directions (isotropic) ------------------
    if ky is None:
        ky = kx

    # -- apply kx horizontally: convolve each row independently ---------------
    # np.apply_along_axis calls the lambda for each row (axis=1).
    # mode="same" keeps the output the same length as the input.
    out = np.apply_along_axis(
        lambda row: np.convolve(row, kx, mode="same"), 1, img
    )

    # -- apply ky vertically: convolve each column of the intermediate result --
    out = np.apply_along_axis(
        lambda col: np.convolve(col, ky, mode="same"), 0, out
    )
    return out


def _sobel_gradients(img):
    """
    Compute horizontal (Ix) and vertical (Iy) image gradients via Sobel operators.

    SOBEL OPERATOR (separable form):
      Sobel-x  =  [−1  0  1] ⊗ [1  2  1]/4    (horizontal differences)
      Sobel-y  =  [1  2  1]/4 ⊗ [−1  0  1]    (vertical differences)

    The [1 2 1]/4 factor is a gentle Gaussian smoothing in the perpendicular
    direction, which reduces noise sensitivity compared to a plain finite-difference.

    FORMULA applied via _conv2d_sep:
      Ix[r,c] ≈ ∂I/∂x ≈ (−I[r,c−1] + I[r,c+1]) averaged over rows r−1,r,r+1
      Iy[r,c] ≈ ∂I/∂y ≈ (−I[r−1,c] + I[r+1,c]) averaged over cols c−1,c,c+1

    Parameters
    ----------
    img : (H, W) float64 grayscale image in [0,1]

    Returns
    -------
    Ix : (H, W) float64  horizontal gradient
    Iy : (H, W) float64  vertical gradient
    """
    # -- Sobel kernels in separable form ----------------------------------------
    diff = np.array([-1.0, 0.0, 1.0])          # finite-difference kernel
    smth = np.array([ 1.0, 2.0, 1.0]) / 4.0   # smoothing kernel (sums to 1)

    # Ix: differentiate horizontally (diff), smooth vertically (smth)
    Ix = _conv2d_sep(img, diff, smth)

    # Iy: smooth horizontally (smth), differentiate vertically (diff)
    Iy = _conv2d_sep(img, smth, diff)

    return Ix, Iy


# ═════════════════════════════════════════════════════════════════════════════
# 3. F1 — HARRIS RESPONSE  (structure tensor corner score)
#    Reference: Harris & Stephens (1988) "A combined corner and edge detector"
# ═════════════════════════════════════════════════════════════════════════════

def harris_response(img, k=0.04, sigma=2.0):
    """
    Compute the Harris corner response map R for the entire image.

    THEORY — Structure Tensor:
      For each pixel (r,c) build the local 2×2 second-moment (structure) matrix:

        M = Σ_{window} w(p) · | Ix²   Ix·Iy |
                               | Ix·Iy  Iy²  |

      where the sum is over a Gaussian-weighted neighbourhood of width σ.

      M captures gradient distribution in a local patch:
        • Flat region:   both eigenvalues small  → no edge, no corner
        • Edge:          one large, one small eigenvalue
        • Corner:        both eigenvalues large  → Harris responds strongly

    HARRIS SCORE:
      Instead of computing eigenvalues directly (expensive), Harris approximates:

        R = det(M) − k · trace(M)²

      • det(M)  = λ₁·λ₂
      • trace(M) = λ₁+λ₂
      • A large positive R indicates a corner (both λ large).
      • k ∈ [0.04, 0.06] is a free sensitivity parameter.

    Parameters
    ----------
    img   : (H, W) float64 grayscale image in [0,1]
    k     : Harris sensitivity constant  (default 0.04)
    sigma : Gaussian window width for structure tensor (default 2.0)

    Returns
    -------
    R : (H, W) float64 Harris response map (high values = likely corners)
    """
    # -- STEP 1: compute image gradients Ix, Iy via Sobel ----------------------
    Ix, Iy = _sobel_gradients(img)

    # -- STEP 2: build 1-D Gaussian kernel for structure tensor smoothing ------
    g = _gaussian_kernel_1d(sigma)

    # -- STEP 3: compute and smooth the three unique elements of M --------------
    # Ixx = Ix²  smoothed by g   → (Σ w·Ix²)
    Ixx = _conv2d_sep(Ix * Ix, g)

    # Iyy = Iy²  smoothed by g   → (Σ w·Iy²)
    Iyy = _conv2d_sep(Iy * Iy, g)

    # Ixy = Ix·Iy smoothed by g  → (Σ w·Ix·Iy)  [off-diagonal element]
    Ixy = _conv2d_sep(Ix * Iy, g)

    # -- STEP 4: Harris response: R = det(M) − k·trace(M)² -------------------
    # det(M)  = Ixx·Iyy − Ixy²        (product of eigenvalues)
    det   = Ixx * Iyy - Ixy ** 2

    # trace(M) = Ixx + Iyy            (sum of eigenvalues)
    trace = Ixx + Iyy

    R = det - k * trace ** 2
    
    # -- STEP 5: Suppress border artifacts ------------------------------------
    # np.convolve padding creates massive gradients at the image borders.
    border = 15
    R[:border, :] = 0
    R[-border:, :] = 0
    R[:, :border] = 0
    R[:, -border:] = 0
    
    return R


# ═════════════════════════════════════════════════════════════════════════════
# 4. NON-MAXIMUM SUPPRESSION + THRESHOLD
# ═════════════════════════════════════════════════════════════════════════════

def detect_corners(response, threshold_rel=0.01, min_dist=10):
    """
    Detect corner candidates from the Harris response map.

    ALGORITHM — Greedy Non-Maximum Suppression (NMS):
      1. Compute threshold = threshold_rel × max(R).
      2. Keep only pixels where R > threshold.
      3. Sort surviving pixels by decreasing R score.
      4. Iterate in score order:
           • If the pixel is not already suppressed → accept it as a corner.
           • Mark all pixels within a (min_dist × min_dist) square around it
             as suppressed (taken = True).
      This ensures no two accepted corners are within min_dist pixels of each other.

    Parameters
    ----------
    response      : (H, W) float64 Harris response map (from harris_response)
    threshold_rel : fraction of global maximum used as acceptance threshold
                    (default 0.01 → keep top 1% strongest responses)
    min_dist      : minimum Euclidean separation (pixels) between corners

    Returns
    -------
    corners : list of (x, y) float tuples, ordered by decreasing response score
              x = column index (horizontal), y = row index (vertical)
    """
    R = response

    # -- STEP 1: absolute threshold from relative fraction of global maximum ---
    threshold = threshold_rel * R.max()   # e.g. 0.01 × max_response

    # -- STEP 2: boolean mask of pixels that exceed the threshold -------------
    mask    = R > threshold               # (H, W) bool

    # -- STEP 3: gather (row, col) indices and their scores -------------------
    indices = np.argwhere(mask)           # (N_above_thresh, 2):  pairs [row, col]
    scores  = R[mask]                     # (N_above_thresh,):    response values

    # -- STEP 4: sort by DESCENDING score (strongest corner first) -------------
    order = np.argsort(-scores)           # argsort of −scores = descending order

    # -- STEP 5: greedy NMS ---------------------------------------------------
    taken   = np.zeros(R.shape, dtype=bool)  # suppression grid: False = available
    corners = []
    for idx in order:
        r, c = indices[idx]               # row, col of this candidate
        if taken[r, c]:
            continue                      # already suppressed → skip

        # Accept this corner (convert to (x=col, y=row) convention)
        corners.append((float(c), float(r)))

        # Suppress all pixels in the min_dist neighbourhood
        r0 = max(0, r - min_dist);  r1 = min(R.shape[0], r + min_dist + 1)
        c0 = max(0, c - min_dist);  c1 = min(R.shape[1], c + min_dist + 1)
        taken[r0:r1, c0:c1] = True       # mark square as suppressed

    return corners  # list of (x, y) already sorted strong-first


# ═════════════════════════════════════════════════════════════════════════════
# 5. SUB-PIXEL CORNER REFINEMENT
#    Reference: Bouguet J-Y "Pyramidal implementation of the Lucas Kanade
#               feature tracker" (Intel 2001), §2
# ═════════════════════════════════════════════════════════════════════════════

def subpixel_refine(img, corners, win=5, max_iter=20, eps=1e-3):
    """
    Refine each detected corner to sub-pixel accuracy using an iterative
    gradient-weighted least-squares method.

    THEORY:
      At a true corner, the image gradient ∇I(p) is perpendicular to the
      vector (p − q) where q is the exact corner location.  Therefore:

        ∇I(p)ᵀ · (p − q) = 0   for all pixels p in the window

      Expanding  p − q = (p − current_estimate) − δ  and grouping:

        Σ w(p)·∇I(p)·∇I(p)ᵀ · δ  =  Σ w(p)·∇I(p)·∇I(p)ᵀ·(p − x_current)

      where w(p) is a Gaussian weight.  Written as the 2×2 system:

        A = | Σ w·Ix²    Σ w·Ix·Iy |       b = | Σ w·Ix·(Ix·Δx + Iy·Δy) |
            | Σ w·Ix·Iy  Σ w·Iy²   |           | Σ w·Iy·(Ix·Δx + Iy·Δy) |

        δ = A⁻¹ · b     (Cramer's rule is used for the 2×2 inversion)

      The corner estimate is updated: x ← x + δ.  Repeat until ‖δ‖ < eps.

    Parameters
    ----------
    img      : (H, W) float64 grayscale
    corners  : list of (x, y) from detect_corners  (integer-level accuracy)
    win      : half-window size  (window is (2·win+1)×(2·win+1) pixels)
    max_iter : maximum iterations per corner before giving up
    eps      : convergence threshold in pixels (‖δ‖ < eps → stop)

    Returns
    -------
    refined : list of (x, y) float tuples, same order as input, sub-pixel accurate
    """
    # -- pre-compute image gradients once for all corners ----------------------
    Ix, Iy  = _sobel_gradients(img)
    H, W    = img.shape

    # -- Gaussian window sigma: make the window half-width the 2σ point --------
    sigma_w = win / 2.0                  # → Gaussian drops to e^{-2} at window edge
    refined = []

    for (cx, cy) in corners:
        # -- initialize estimate at the integer-level NMS position -------------
        x, y = float(cx), float(cy)

        for _ in range(max_iter):
            # -- round current estimate to nearest integer for window bounds ---
            xi = int(round(x));  yi = int(round(y))

            # -- window bounds centred on (xi, yi), clipped to image boundary --
            x0 = max(0, xi - win);  x1 = min(W, xi + win + 1)
            y0 = max(0, yi - win);  y1 = min(H, yi + win + 1)

            # -- skip if window is too small (corner near image border) --------
            if (x1 - x0) < 3 or (y1 - y0) < 3:
                break

            # -- offset grids: Δcol and Δrow relative to current (x, y) --------
            # cols = [x0-x, x0-x+1, …, x1-1-x]  (horizontal displacements from x)
            # rows = [y0-y, y0-y+1, …, y1-1-y]  (vertical   displacements from y)
            cols = np.arange(x0, x1, dtype=np.float64) - x   # (w_cols,)
            rows = np.arange(y0, y1, dtype=np.float64) - y   # (w_rows,)

            # cc[r,c] = col offset,  rr[r,c] = row offset  for each window pixel
            cc, rr = np.meshgrid(cols, rows)     # both shape: (w_rows, w_cols)

            # -- Gaussian weights centred at current (x, y) --------------------
            # w(p) = exp(−(Δx²+Δy²) / (2·σ_w²))
            w_g = np.exp(-(cc**2 + rr**2) / (2.0 * sigma_w**2))  # (w_rows, w_cols)

            # -- gradient patches in the current window ------------------------
            ix_p = Ix[y0:y1, x0:x1]   # (w_rows, w_cols) horizontal gradient
            iy_p = Iy[y0:y1, x0:x1]   # (w_rows, w_cols) vertical gradient

            # -- build the 2×2 system  A·δ = b --------------------------------
            # A = [ Σ w·Ix²    Σ w·Ix·Iy ]
            #     [ Σ w·Ix·Iy  Σ w·Iy²   ]
            A00 = (w_g * ix_p * ix_p).sum()     # (1,1) element of A
            A01 = (w_g * ix_p * iy_p).sum()     # (1,2) = (2,1) element of A
            A11 = (w_g * iy_p * iy_p).sum()     # (2,2) element of A

            # b = [ Σ w·Ix·(cc·Ix + rr·Iy) ]   (Ix component of gradient·displacement)
            #     [ Σ w·Iy·(cc·Ix + rr·Iy) ]   (Iy component)
            b0  = (w_g * ix_p * (cc * ix_p + rr * iy_p)).sum()
            b1  = (w_g * iy_p * (cc * ix_p + rr * iy_p)).sum()

            # -- solve 2×2 system via Cramer's rule ---------------------------
            # det(A) = A00·A11 − A01²
            det = A00 * A11 - A01 ** 2
            if abs(det) < 1e-12:
                break                   # singular matrix → cannot refine further

            # δx = (A11·b0 − A01·b1) / det
            dx = ( A11 * b0 - A01 * b1) / det
            # δy = (A00·b1 − A01·b0) / det
            dy = (-A01 * b0 + A00 * b1) / det

            # -- update corner position estimate --------------------------------
            x += dx;  y += dy

            # -- convergence check: stop if update is sub-eps ─────────────────
            if abs(dx) < eps and abs(dy) < eps:
                break

        refined.append((float(x), float(y)))

    return refined


# ═════════════════════════════════════════════════════════════════════════════
# 6. CHECKERBOARD CORNER ORDERING  (PCA-based grid sort)
# ═════════════════════════════════════════════════════════════════════════════

def order_checkerboard_corners(corners, grid_shape):
    """
    Robust combinatorial checkerboard ordering using Local Topology RANSAC.

    ALGORITHM:
      1. Local 1x1 Cell Discovery: Find any 3 points that form a right-angle
         L-shape. Extrapolate the 4th point of the 1x1 square.
      2. Local Homography: Estimate H mapping the ideal [0,1]x[0,1] cell to
         this localized square using DLT.
      3. Global Projection: Extrapolate the complete cols×rows grid using H.
      4. Inlier Matching: Snap projected points to the nearest Harris corners.
      5. Global Refinement: If >70% of the grid matches, recompute a Global H
         using all matched points, and re-project for pixel-perfect sorting.

    WHY THIS APPROACH: A pure-numpy substitute for OpenCV's topological connected 
    components. Strong against heavy background clutter since it leverages local
    grid regularities built symmetrically.

    Parameters
    ----------
    corners    : list of (x, y) — subpixel Harris corners
    grid_shape : (cols, rows) 

    Returns
    -------
    ordered : list of (cols×rows) coordinates in row-major sequence, or None if failed.
    """
    cols, rows = grid_shape
    expected = cols * rows
    
    # 1. Take top 150 points (limits combinatorial explosion but retains robust background tolerance)
    pts = np.array(corners, dtype=np.float64)[:min(len(corners), 150)]
    N = len(pts)
    
    if N < expected:
        return None
        
    from src.math_core import dlt_homography
    
    # Pairwise distances for extracting spatial nearest neighbors
    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    neighbors = np.argsort(dists, axis=1) 
    
    # Mathematical models for 1x1 square and complete grid
    ideal_1x1 = np.array([[0,0], [1,0], [1,1], [0,1]], dtype=np.float64)
    ideal_grid = np.array([[c, r] for r in range(rows) for c in range(cols)], dtype=np.float64)
    ideal_grid_h = np.hstack([ideal_grid, np.ones((len(ideal_grid), 1))])
    
    best_count = 0
    best_ordered = None
    best_err = np.inf
    
    # Scan through every point as the origin A (0,0)
    for i in range(N):
        A = pts[i]
        
        # Test nearest neighbors as B (1, 0)
        for b_idx in neighbors[i, 1:6]:
            B = pts[b_idx]
            vAB = B - A
            L = np.linalg.norm(vAB)
            
            # Test nearest neighbors as C (0, 1)
            for c_idx in neighbors[i, 1:6]:
                if c_idx == b_idx: continue
                C = pts[c_idx]
                vAC = C - A
                
                # Check for Orthogonality: dot(vAB, vAC) ≈ 0
                cos_angle = np.abs(np.dot(vAB, vAC)) / (L * np.linalg.norm(vAC))
                if cos_angle > 0.4: 
                    continue
                    
                # Ensure a strict right-handed coordinate frame avoiding mirror-flipped grids!
                if np.cross(vAB, vAC) < 0:
                    continue
                    
                # Theoretically expected position of the 4th corner D (1, 1)
                # D_theoretical = A + vAB + vAC = B + C - A
                D_ex = B + C - A
                
                # Search for an actual detected Harris point near D_ex
                d_dists = np.linalg.norm(pts - D_ex, axis=1)
                d_idx = np.argmin(d_dists)
                if d_dists[d_idx] > 0.6 * L:
                    continue
                    
                D = pts[d_idx]
                quad = np.array([A, B, D, C])
                
                # DLT Homography for the local 1x1 square
                try:
                    H_local = dlt_homography(ideal_1x1, quad)
                except np.linalg.LinAlgError:
                    continue
                    
                # Extrapolate entire cols×rows grid
                proj = (H_local @ ideal_grid_h.T).T
                z = proj[:, 2:]
                z[np.abs(z) < 1e-8] = 1e-8
                proj = proj[:, :2] / z
                
                inliers = 0
                used = set()
                curr_ordered = []
                err_sum = 0
                
                # Snap extrapolated coords to the nearest physical Harris corners
                for p in proj:
                    dsts = np.linalg.norm(pts - p, axis=1)
                    for u in used: dsts[u] = np.inf
                    best_match = np.argmin(dsts)
                    min_dist = dsts[best_match]
                    
                    if min_dist < max(15.0, 0.8 * L): 
                        inliers += 1
                        used.add(best_match)
                        curr_ordered.append(best_match)
                        err_sum += min_dist
                
                # Successful validation threshold
                if inliers >= expected * 0.7:
                    if inliers > best_count or (inliers == best_count and err_sum < best_err):
                        best_count = inliers
                        best_err = err_sum
                        
                        # Full geometric refinement: recompute H on ALL strictly matched inliers
                        src_matched = []
                        dst_matched = []
                        for m_idx, pt_idx in enumerate(curr_ordered):
                            src_matched.append(ideal_grid[m_idx])
                            dst_matched.append(pts[pt_idx])
                            
                        try:
                            H_global = dlt_homography(src_matched, dst_matched)
                            proj_full = (H_global @ ideal_grid_h.T).T
                            z_full = proj_full[:, 2:]
                            z_full[np.abs(z_full) < 1e-8] = 1e-8
                            proj_full = proj_full[:, :2] / z_full
                            
                            # Final assignment mapping
                            final_ordered = []
                            for pf in proj_full:
                                pdists = np.linalg.norm(pts - pf, axis=1)
                                b_match = np.argmin(pdists)
                                final_ordered.append((float(pts[b_match, 0]), float(pts[b_match, 1])))
                                
                            best_ordered = final_ordered
                            best_count = expected # force accept
                        except:
                            pass
                            
                if best_count == expected:
                    break
        if best_count == expected:
            break
            
    if best_count == expected and best_ordered is not None:
        return best_ordered
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 7. WORLD-POINT GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

def make_object_points(grid_shape, square_size):
    """
    Generate the ideal 3-D world coordinates for all inner corners of a
    planar checkerboard calibration target.

    CONVENTION:
      • World origin at the top-left inner corner.
      • X axis runs left-to-right (along columns of corners).
      • Y axis runs top-to-bottom (along rows of corners).
      • Z = 0 for all points (planar target assumption).

    FORMULA:
      Point at (col_index c, row_index r):
        X = c × square_size
        Y = r × square_size
        Z = 0

    Parameters
    ----------
    grid_shape  : (cols, rows)  number of inner corners in each direction
    square_size : physical side length of one square in any consistent unit (mm, cm, …)

    Returns
    -------
    obj_pts : (N, 3) float64,  N = cols × rows,  z-column is all zeros
    """
    cols, rows = grid_shape

    # -- list comprehension: iterate rows (outer) then cols (inner) ------------
    # This produces row-major order:  (0,0), (1,0), …, (C-1,0), (0,1), …
    pts = [
        [c * square_size,   # X coordinate: c squares from origin
         r * square_size,   # Y coordinate: r squares from origin
         0.0]               # Z coordinate: always 0 (planar target)
        for r in range(rows)
        for c in range(cols)
    ]
    return np.array(pts, dtype=np.float64)   # (cols×rows, 3)


# ═════════════════════════════════════════════════════════════════════════════
# 8. MAIN EXTRACTION DRIVER
# ═════════════════════════════════════════════════════════════════════════════

def extract_all_views(
    image_dir,
    grid_shape,
    square_size,
    harris_k     = 0.04,
    harris_sigma = 2.0,
    threshold_rel= 0.01,
    min_dist     = 10,
):
    """
    Run the complete F1 pipeline on every supported image in `image_dir`.

    For each image the pipeline is:
      load_image_gray
        → harris_response   (F1 step A: compute R for all pixels)
        → detect_corners    (F1 step B: NMS + threshold → candidate list)
        → subpixel_refine   (F1 step C: gradient-LS → sub-pixel positions)
        → order_checkerboard_corners  (F1 step D: PCA sort → row-major order)

    Images where the corner count does not match grid_shape are skipped.

    Parameters
    ----------
    image_dir     : directory path containing checkerboard images
    grid_shape    : (cols, rows) inner corners per image
    square_size   : physical side length of one square (same unit as desired t)
    harris_k      : Harris sensitivity  (default 0.04)
    harris_sigma  : Gaussian σ for structure tensor  (default 2.0)
    threshold_rel : NMS threshold as fraction of peak response  (default 0.01)
    min_dist      : minimum corner separation in pixels  (default 10)

    Returns
    -------
    img_pts_list : list of (N, 2) float64 arrays — observed corners per view
    obj_pts_list : list of (N, 3) float64 arrays — world points (same for all)
    valid_paths  : list of file paths that were successfully processed
    image_size   : (width, height) tuple from the first accepted image
    """
    # -- supported file extensions (case-insensitive) -------------------------
    SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    # -- collect and sort all supported image paths ---------------------------
    # Sorting ensures deterministic processing order across platforms.
    paths = sorted(
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED
    )

    # -- pre-compute the constant world-point template (same for every view) ---
    obj_template = make_object_points(grid_shape, square_size)   # (N, 3)

    img_pts_list, obj_pts_list, valid_paths = [], [], []
    image_size = None    # set from the first accepted image

    for path in paths:
        # -- F1 step 0: load as float64 grayscale ----------------------------
        img  = load_image_gray(path)
        H, W = img.shape
        if image_size is None:
            image_size = (W, H)      # (width, height) — set once

        # -- F1 step A: Harris response map -----------------------------------
        resp = harris_response(img, k=harris_k, sigma=harris_sigma)
        # resp[r,c] is large and positive at corner-like pixels

        # -- F1 step B: NMS + threshold to get integer-level corner list ------
        raw = detect_corners(resp, threshold_rel=threshold_rel, min_dist=min_dist)
        # raw = [(x0,y0), (x1,y1), …] ordered strong-to-weak

        # -- F1 step C: sub-pixel gradient refinement -------------------------
        refined = subpixel_refine(img, raw)
        # refined = [(x0',y0'), …] with sub-pixel accuracy

        # -- F1 step D: sort into row-major checkerboard order ----------------
        ordered = order_checkerboard_corners(refined, grid_shape)
        # ordered = None if corner count mismatch, else [(x,y)×N] in grid order

        name = os.path.basename(path)
        if ordered is None:
            n_found = len(refined)
            n_need  = grid_shape[0] * grid_shape[1]
            print(f"  [SKIP] {name} — found {n_found} corners, need {n_need}")
            continue

        # -- accept this view: store observed and world corners ---------------
        img_pts_list.append(np.array(ordered, dtype=np.float64))  # (N, 2)
        obj_pts_list.append(obj_template.copy())                   # (N, 3)
        valid_paths.append(path)
        print(f"  [OK]   {name} — {len(ordered)} corners")

    return img_pts_list, obj_pts_list, valid_paths, image_size
