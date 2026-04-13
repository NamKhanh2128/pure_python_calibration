"""
undistorter.py — R2 (Inverse + Nearest): Remove lens distortion from images
using inverse (backward) mapping and nearest-neighbour pixel interpolation.

WHY INVERSE MAPPING:
  Forward mapping (distorted src → corrected dst) leaves gaps because distorted
  source pixels do not map to a regular grid in the destination.
  Inverse mapping iterates over every *destination* pixel, computes where it
  came from in the distorted *source* image, and copies that pixel value.
  Every destination pixel is filled → no holes.

DISTORTION MODEL (Brown-Conrady, 2-parameter radial only):
  Given an undistorted normalized coordinate (x_n, y_n):
    r²  = x_n² + y_n²
    fac = 1 + k1·r² + k2·r⁴
    x_d = x_n · fac          (distorted normalized x)
    y_d = y_n · fac          (distorted normalized y)

  In pixel space:
    u_src = fx·x_d + cx      (source column in the captured/distorted image)
    v_src = fy·y_d + cy      (source row)

INTERPOLATION: Nearest-Neighbour
  Round (u_src, v_src) to the nearest integer → copy that source pixel.
  Fast and simple; bilinear would give smoother results but is not required
  for this project (R2 specification).
"""

import numpy as np
from PIL import Image


# ═════════════════════════════════════════════════════════════════════════════
# 1. DISTORTION MAP BUILDER  (internal helper)
# ═════════════════════════════════════════════════════════════════════════════

def _build_distortion_maps(K, dist, H, W):
    """
    Pre-compute (src_col, src_row) for every destination pixel (u, v).

    This function embodies steps 1–3 of the R2 algorithm:
      1. Normalize destination pixel to undistorted normalized plane.
      2. Apply the forward distortion model to get distorted normalized coords.
      3. Back-project to source pixel coordinates.

    Computing the maps once and reusing them for lookup is efficient because it
    separates the geometry computation (NumPy vectorized) from the pixel fetch.

    Parameters
    ----------
    K    : (3,3) intrinsic matrix  [[fx,0,cx],[0,fy,cy],[0,0,1]]
    dist : [k1, k2] radial distortion coefficients
    H    : image height (destination grid height in pixels)
    W    : image width  (destination grid width  in pixels)

    Returns
    -------
    src_col : (H, W) float64 — source column (u_src) for each dst pixel (u,v)
    src_row : (H, W) float64 — source row    (v_src) for each dst pixel (u,v)
    """
    # -- unpack intrinsics and distortion coefficients -------------------------
    fx, fy = float(K[0, 0]), float(K[1, 1])   # focal lengths (pixels)
    cx, cy = float(K[0, 2]), float(K[1, 2])   # principal point (pixels)
    k1, k2 = float(dist[0]), float(dist[1])   # radial distortion coefficients

    # -- STEP 1A: build destination pixel coordinate grids (vectorized) --------
    # u_d[v, u] = u   (column index, i.e. horizontal pixel position)
    # v_d[v, u] = v   (row index,    i.e. vertical   pixel position)
    # meshgrid(cols, rows): cols varies along axis=1, rows along axis=0.
    u_d, v_d = np.meshgrid(
        np.arange(W, dtype=np.float64),   # column indices: 0, 1, …, W-1
        np.arange(H, dtype=np.float64),   # row    indices: 0, 1, …, H-1
    )
    # Both u_d, v_d have shape (H, W)

    # -- STEP 1B: convert destination pixel → undistorted normalized plane -----
    # This is the inverse of the final pixel-to-normalized mapping in projection:
    #   u = fx·x_n + cx  →  x_n = (u − cx) / fx
    #   v = fy·y_n + cy  →  y_n = (v − cy) / fy
    # x_n, y_n are the normalized image plane coordinates (z=1) for an ideal
    # (undistorted) camera.
    x_n = (u_d - cx) / fx     # (H, W): normalized undistorted x
    y_n = (v_d - cy) / fy     # (H, W): normalized undistorted y

    # -- STEP 2: apply the forward radial distortion to get source coords ------
    # The forward model says: given a normalized UNdistorted point (x_n, y_n),
    # the distorted point (x_d, y_d) is obtained by:
    #   r²  = x_n² + y_n²           (squared radius in normalized plane)
    #   fac = 1 + k1·r² + k2·r⁴     (polynomial distortion factor)
    #   x_d = x_n·fac               (x is pushed outward for k1 > 0: barrel dist.)
    #   y_d = y_n·fac               (y similarly)
    r2  = x_n ** 2 + y_n ** 2          # (H, W): r² at each destination pixel
    fac = 1.0 + k1 * r2 + k2 * r2**2  # (H, W): distortion factor

    x_d_norm = x_n * fac               # (H, W): distorted normalized x
    y_d_norm = y_n * fac               # (H, W): distorted normalized y

    # -- STEP 3: back-project distorted normalized coords → source pixel space --
    # Apply the intrinsic mapping (same K as the undistorted camera):
    #   u_src = fx·x_d + cx
    #   v_src = fy·y_d + cy
    src_col = fx * x_d_norm + cx       # (H, W): source column (float)
    src_row = fy * y_d_norm + cy       # (H, W): source row    (float)

    return src_col, src_row


# ═════════════════════════════════════════════════════════════════════════════
# 2. R2 — CORE UNDISTORTION  (nearest-neighbour gather)
# ═════════════════════════════════════════════════════════════════════════════

def undistort_image(img_array, K, dist):
    """
    Remove radial lens distortion from a single image.

    FULL R2 ALGORITHM (per destination pixel (u, v)):
      1. x_n = (u − cx)/fx,  y_n = (v − cy)/fy      [normalize]
      2. r² = x_n²+y_n²;  fac = 1+k1·r²+k2·r⁴       [distortion factor]
         x_d = x_n·fac,   y_d = y_n·fac              [distorted coords]
      3. u_src = fx·x_d+cx,  v_src = fy·y_d+cy       [source pixel (float)]
      4. u_s   = round(u_src),  v_s = round(v_src)   [nearest-neighbour]
      5. dst[v, u] = src[v_s, u_s]  (if in bounds, else 0 = black)

    Steps 1–3 are vectorized over all (H×W) destination pixels at once
    via _build_distortion_maps.  Steps 4–5 are then performed in bulk
    using NumPy advanced indexing.

    Parameters
    ----------
    img_array : (H, W, C) uint8 RGB image  OR  (H, W) uint8 grayscale
    K         : (3,3) intrinsic matrix
    dist      : [k1, k2] radial distortion coefficients

    Returns
    -------
    dst : same shape and dtype as img_array, with distortion removed
          (out-of-bounds source regions appear black)
    """
    # -- determine if image is colour or grayscale ----------------------------
    is_color = img_array.ndim == 3        # True for (H,W,3), False for (H,W)
    H, W     = img_array.shape[:2]

    # -- STEPS 1-3: compute source coordinates for every destination pixel ----
    # Both return (H, W) float64 arrays.
    src_col_f, src_row_f = _build_distortion_maps(K, dist, H, W)

    # -- STEP 4: nearest-neighbour — round float source coords to integers -----
    # np.round uses "round half to even" (banker's rounding), which is fine here.
    src_col = np.round(src_col_f).astype(np.int32)   # (H, W)  integer column
    src_row = np.round(src_row_f).astype(np.int32)   # (H, W)  integer row

    # -- STEP 5A: build validity mask — True where source index is in-bounds ---
    # A source pixel is valid if 0 ≤ col < W  AND  0 ≤ row < H.
    # Invalid pixels (outside the sensor) will remain 0 (black) in dst.
    valid = (
        (src_col >= 0) & (src_col < W) &   # column in [0, W-1]
        (src_row >= 0) & (src_row < H)     # row    in [0, H-1]
    )                                        # (H, W) bool

    # -- STEP 5B: clamp indices for safe NumPy indexing (out-of-bound → dummy) -
    # We first clamp to [0, W-1] / [0, H-1] so that fancy indexing never raises
    # an IndexError.  The validity mask then ensures clamped-but-invalid pixels
    # are written as 0 anyway.
    sc = np.clip(src_col, 0, W - 1)   # (H, W) clamped column indices
    sr = np.clip(src_row, 0, H - 1)   # (H, W) clamped row indices

    # -- STEP 5C: gather pixels from source into destination ------------------
    if is_color:
        # dst shape: (H, W, C), initialized to black (0, 0, 0)
        dst = np.zeros((H, W, img_array.shape[2]), dtype=img_array.dtype)

        # For each valid destination pixel (v, u):
        #   dst[v, u, :] = src[sr[v,u], sc[v,u], :]
        # valid is (H,W) bool; fancy indexing extracts the valid rows.
        dst[valid] = img_array[sr[valid], sc[valid]]
    else:
        # dst shape: (H, W), grayscale, initialized to 0
        dst = np.zeros((H, W), dtype=img_array.dtype)
        dst[valid] = img_array[sr[valid], sc[valid]]

    return dst


# ═════════════════════════════════════════════════════════════════════════════
# 3. FILE-LEVEL WRAPPER
# ═════════════════════════════════════════════════════════════════════════════

def undistort_image_file(input_path, output_path, K, dist):
    """
    Load an image from disk, apply R2 undistortion, and save the result.

    This is a thin convenience wrapper around undistort_image that:
      1. Opens the file with PIL and forces RGB (3-channel) format.
      2. Converts to uint8 NumPy array.
      3. Calls undistort_image (the core R2 algorithm).
      4. Converts back to PIL Image and saves (format inferred from extension).

    Parameters
    ----------
    input_path  : path to the distorted source image (any PIL-supported format)
    output_path : path to write the corrected image (directory must exist)
    K           : (3,3) intrinsic matrix
    dist        : [k1, k2] radial distortion coefficients
    """
    # -- STEP 1: load source image as RGB uint8 array -------------------------
    # .convert("RGB") handles grayscale, RGBA, palette images uniformly.
    img = np.array(Image.open(input_path).convert("RGB"), dtype=np.uint8)
    # img shape: (H, W, 3),  values in [0, 255]

    # -- STEP 2: apply R2 inverse-mapping + nearest-neighbour undistortion ----
    out = undistort_image(img, K, dist)
    # out shape: (H, W, 3),  same dtype (uint8)

    # -- STEP 3: save the corrected image to disk -----------------------------
    # PIL.Image.fromarray wraps the uint8 array back into a PIL Image object.
    # .save() infers format from the file extension (JPEG, PNG, etc.).
    Image.fromarray(out).save(output_path)
