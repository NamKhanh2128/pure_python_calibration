"""
math_core.py — L1 (SVD): Pure-NumPy math primitives for Zhang's camera calibration.

Public API:
  normalize_points                   — isotropic DLT conditioning
  dlt_homography                     — 3×3 H via normalized SVD
  compute_v_ij                       — row of Zhang's V constraint matrix
  solve_intrinsics_from_homographies — K from ≥3 homographies via SVD
  rodrigues_to_R                     — rvec → 3×3 R (Rodrigues formula)
  R_to_rodrigues                     — 3×3 R → rvec
  compute_extrinsics                 — H, K → rvec, t  (per-view pose)
  project_points                     — forward projection with radial distortion
  init_distortion                    — linear least-squares init for k1, k2
"""

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# 1. POINT NORMALIZATION
#    Reference: Hartley & Zisserman "Multiple View Geometry" §4.4
# ═════════════════════════════════════════════════════════════════════════════

def normalize_points(pts):
    """
    Isotropic similarity normalization for DLT conditioning.

    WHY: Without normalization the DLT system is poorly conditioned because
    pixel coordinates can be O(1000) while homogeneous coordinates are O(1),
    causing severe numerical cancellation in the SVD.

    WHAT: Find a similarity transform T such that, after applying T:
      • the centroid of the point cloud is at the origin
      • the mean Euclidean distance from the origin is √2

    FORMULA:
      T = | s  0  -s·cx |      where  cx, cy = centroid
          | 0  s  -s·cy |             s  = √2 / mean_dist
          | 0  0    1   |

    Parameters
    ----------
    pts : (N, 2) float array   raw 2-D points (world XY or image UV)

    Returns
    -------
    T     : (3, 3) normalization matrix
    pts_n : (N, 2) normalized points satisfying the two conditions above
    """
    # -- cast to float64 so later arithmetic never silently truncates ----------
    pts = np.array(pts, dtype=np.float64)

    # -- centroid: mean position across all N points --------------------------
    # centroid[0] = (1/N)·Σ xᵢ,  centroid[1] = (1/N)·Σ yᵢ
    centroid = pts.mean(axis=0)          # shape (2,)

    # -- shift: subtract centroid so the new origin coincides with the centre -
    shifted = pts - centroid             # shape (N, 2)

    # -- mean distance from the new origin ------------------------------------
    # dist_i = √(shifted_x² + shifted_y²) for each point i
    # mean_dist = (1/N)·Σ dist_i
    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()

    # -- guard: if all points are identical mean_dist=0 → use scale=1 ---------
    if mean_dist < 1e-10:
        mean_dist = 1.0

    # -- scale factor: chosen so that after scaling mean_dist becomes √2 ------
    # s·mean_dist = √2  →  s = √2 / mean_dist
    scale = np.sqrt(2.0) / mean_dist

    # -- build T: 3×3 similarity transform  T·[x,y,1]ᵀ = [s(x-cx), s(y-cy), 1]ᵀ
    T = np.array([
        [scale,  0,     -scale * centroid[0]],   # row 0: scaled x
        [0,      scale, -scale * centroid[1]],   # row 1: scaled y
        [0,      0,      1.0               ],    # row 2: homogeneous
    ], dtype=np.float64)

    # -- apply T via matrix multiply in homogeneous coordinates ----------------
    # lift pts to (N,3) by appending a column of 1s:  [x, y, 1]
    ones  = np.ones((len(pts), 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])          # (N, 3)

    # T @ pts_h.T  gives (3, N);  transpose → (N, 3);  drop last column → (N, 2)
    pts_n = (T @ pts_h.T).T[:, :2]

    return T, pts_n


# ═════════════════════════════════════════════════════════════════════════════
# 2. DLT HOMOGRAPHY  (L1 — SVD)
#    Reference: Hartley & Zisserman §4.1
# ═════════════════════════════════════════════════════════════════════════════

def dlt_homography(src, dst):
    """
    Compute the 3×3 planar homography H such that  dst ~ H · src.

    THEORY:
      A homography maps one projective plane to another.  For calibration,
      src = (X, Y) world coordinates on the checkerboard plane (z=0),
      dst = (u, v) observed image corner positions.

      For each correspondence (x,y) ↔ (u,v), the constraint  dst ~ H·src
      expands to two linear equations in h = vec(H):

        [-x  -y  -1   0   0   0   ux  uy  u] · h = 0   (u equation)
        [ 0   0   0  -x  -y  -1   vx  vy  v] · h = 0   (v equation)

      Stacking all N pairs gives  A·h = 0  (A is 2N×9).
      Solution: h = right singular vector of A for the smallest singular value.

    Parameters
    ----------
    src : (N, 2)  source points  (world XY),  N ≥ 4
    dst : (N, 2)  destination points  (image UV corners)

    Returns
    -------
    H : (3, 3),  H[2,2] == 1
    """
    # -- ensure float64 --------------------------------------------------------
    src = np.array(src, dtype=np.float64)
    dst = np.array(dst, dtype=np.float64)
    assert len(src) >= 4, "DLT requires at least 4 point correspondences."

    # -- NORMALIZE both sets independently ------------------------------------
    # This is the "Normalized DLT" of Hartley (1997).
    # T_src maps src → src_n;  T_dst maps dst → dst_n.
    T_src, src_n = normalize_points(src)    # T_src: (3,3),  src_n: (N,2)
    T_dst, dst_n = normalize_points(dst)    # T_dst: (3,3),  dst_n: (N,2)

    # -- BUILD design matrix A of shape (2N, 9) --------------------------------
    # Each row pair encodes one point correspondence in the normalized frame.
    N = len(src_n)
    A = np.zeros((2 * N, 9), dtype=np.float64)
    for i in range(N):
        x, y = src_n[i]     # normalized source coords (world)
        u, v = dst_n[i]     # normalized destination coords (image)

        # u-equation (cross product constraint, row 0):
        #   [−x, −y, −1,  0,  0,  0,  u·x, u·y, u]
        A[2*i    ] = [-x, -y, -1,  0,  0,  0,  u*x,  u*y,  u]

        # v-equation (cross product constraint, row 1):
        #   [ 0,  0,  0, −x, −y, −1,  v·x, v·y, v]
        A[2*i + 1] = [ 0,  0,  0, -x, -y, -1,  v*x,  v*y,  v]

    # -- SVD: A = U·S·Vᵀ -------------------------------------------------------
    # Rows of Vt are right singular vectors, sorted by ascending singular value.
    # The LAST row of Vt → smallest singular value → least-squares solution to A·h=0.
    _, _, Vt = np.linalg.svd(A)
    h   = Vt[-1]             # shape (9,): flattened H in normalized coordinates
    H_n = h.reshape(3, 3)    # reshape to 3×3 matrix

    # -- DENORMALIZE: undo the two normalization transforms -------------------
    # If H_n maps src_n → dst_n, then:
    #   H_real = T_dst⁻¹ · H_n · T_src
    H = np.linalg.inv(T_dst) @ H_n @ T_src

    # -- Fix scale: convention H[2,2] = 1 (homogeneous, up to scale) -----------
    H /= H[2, 2]
    return H


# ═════════════════════════════════════════════════════════════════════════════
# 3. ZHANG'S V-MATRIX  (L1 — SVD)
#    Reference: Zhang "A flexible new technique for camera calibration" (2000)
# ═════════════════════════════════════════════════════════════════════════════

def compute_v_ij(H, i, j):
    """
    Build one 6-element row of Zhang's constraint matrix V.

    THEORY:
      Let B = K⁻ᵀ·K⁻¹ (the image of the Absolute Conic, IAC).
      B is symmetric, so it has 6 independent entries:
        b = [B₁₁, B₁₂, B₂₂, B₁₃, B₂₃, B₃₃]ᵀ

      For a homography H = [h₁ | h₂ | h₃], the IAC constraint
        hᵢᵀ · B · hⱼ = 0  (orthogonality / equal-norm conditions)
      expands to a 6-element dot product:
        v_ij · b = 0

      This function computes v_ij for a given pair of column indices i, j.

    FORMULA (indices 0-based, H has columns h₀, h₁, h₂):
      v_ij = [ H[0,i]·H[0,j],
               H[0,i]·H[1,j] + H[1,i]·H[0,j],
               H[1,i]·H[1,j],
               H[2,i]·H[0,j] + H[0,i]·H[2,j],
               H[2,i]·H[1,j] + H[1,i]·H[2,j],
               H[2,i]·H[2,j] ]

    Parameters
    ----------
    H : (3, 3) homography
    i : first column index  (0 or 1)
    j : second column index (0 or 1)

    Returns
    -------
    (6,) float64  constraint row
    """
    # Each element below is the coefficient of one B entry in hᵢᵀ·B·hⱼ.
    return np.array([
        H[0, i] * H[0, j],                              # coeff of B₁₁
        H[0, i] * H[1, j] + H[1, i] * H[0, j],         # coeff of B₁₂ (symmetric)
        H[1, i] * H[1, j],                              # coeff of B₂₂
        H[2, i] * H[0, j] + H[0, i] * H[2, j],         # coeff of B₁₃ (symmetric)
        H[2, i] * H[1, j] + H[1, i] * H[2, j],         # coeff of B₂₃ (symmetric)
        H[2, i] * H[2, j],                              # coeff of B₃₃
    ], dtype=np.float64)


def solve_intrinsics_from_homographies(Hs):
    """
    Recover the camera intrinsics K from ≥3 homographies (Zhang's method).

    THEORY:
      Every homography H provides two linear constraints on b = vec(B):

        Constraint 1 (orthogonality):  h₁ᵀ·B·h₂ = 0   →  v₀₁·b = 0
        Constraint 2 (equal norm):     h₁ᵀ·B·h₁ = h₂ᵀ·B·h₂
                                       →  (v₀₀ − v₁₁)·b = 0

      Stacking for all n views gives V·b = 0  (V is 2n×6).
      The least-squares solution is the right null-vector of V (last row of Vᵀ).

      Once b is known, K is extracted analytically using the formulae in
      Zhang (2000) Appendix B:

        v₀  = (B₁₂·B₁₃ − B₁₁·B₂₃) / (B₁₁·B₂₂ − B₁₂²)
        λ   = B₃₃ − [B₁₃² + v₀·(B₁₂·B₁₃ − B₁₁·B₂₃)] / B₁₁
        fx  = √(λ/B₁₁)
        fy  = √(λ·B₁₁ / (B₁₁·B₂₂ − B₁₂²))
        γ   = −B₁₂·fx²·fy / λ          (skew ≈ 0 for modern sensors)
        cx  = γ·v₀/fy − B₁₃·fx²/λ

    Parameters
    ----------
    Hs : list of n (3,3) homographies  (n ≥ 3)

    Returns
    -------
    K : (3,3) upper-triangular intrinsic matrix
        | fx  γ  cx |
        |  0  fy cy |
        |  0   0  1 |
    """
    # -- STEP 1: build V matrix  (2 rows per homography) ----------------------
    V_rows = []
    for H in Hs:
        # Orthogonality: h₁ᵀ B h₂ = 0  →  v₀₁ · b = 0
        V_rows.append(compute_v_ij(H, 0, 1))
        # Equal-norm: (h₁ᵀ B h₁ − h₂ᵀ B h₂) = 0  →  (v₀₀ − v₁₁) · b = 0
        V_rows.append(compute_v_ij(H, 0, 0) - compute_v_ij(H, 1, 1))
    V = np.array(V_rows, dtype=np.float64)      # shape: (2n, 6)

    # -- STEP 2: solve V·b = 0 via SVD ----------------------------------------
    # b is the right null vector of V → last row of Vᵀ (smallest singular value).
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]                              # (6,): [B₁₁, B₁₂, B₂₂, B₁₃, B₂₃, B₃₃]
    B11, B12, B22, B13, B23, B33 = b

    # -- STEP 3: sign disambiguation -------------------------------------------
    # B must be positive definite (B = K⁻ᵀK⁻¹ > 0).
    # The SVD null-space is defined only up to sign; flip if B₁₁ < 0.
    if B11 < 0:
        b  = -b
        B11, B12, B22, B13, B23, B33 = b

    # -- STEP 4: check denominator for numerical stability --------------------
    # denom = B₁₁·B₂₂ − B₁₂²  must be non-zero (positive for PD matrix).
    denom = B11 * B22 - B12 ** 2
    if abs(denom) < 1e-12:
        raise ValueError(
            "Degenerate V matrix — add more views or improve corner detection."
        )

    # -- STEP 5: analytically extract K from b --------------------------------

    # Principal point y-coordinate (cy):
    #   v₀ = cy = (B₁₂·B₁₃ − B₁₁·B₂₃) / (B₁₁·B₂₂ − B₁₂²)
    v0  = (B12 * B13 - B11 * B23) / denom

    # Scale factor λ:
    #   λ = B₃₃ − [B₁₃² + cy·(B₁₂·B₁₃ − B₁₁·B₂₃)] / B₁₁
    lam = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
    if lam < 0:
        lam = abs(lam)    # numerical guard — theoretically lam > 0

    # Focal length in x:  fx = √(λ / B₁₁)
    alpha = np.sqrt(abs(lam / B11))

    # Focal length in y:  fy = √(λ·B₁₁ / denom)
    beta  = np.sqrt(abs(lam * B11 / denom))

    # Skew coefficient:   γ = −B₁₂·fx²·fy / λ   (≈ 0 for most cameras)
    gamma = -B12 * alpha ** 2 * beta / lam

    # Principal point x-coordinate (cx):
    #   cx = γ·cy/fy − B₁₃·fx²/λ
    u0 = gamma * v0 / beta - B13 * alpha ** 2 / lam

    # -- STEP 6: assemble the intrinsic matrix K --------------------------------
    K = np.array([
        [alpha, gamma, u0 ],      # row 0: [fx,  γ, cx]
        [0,     beta,  v0 ],      # row 1: [ 0, fy, cy]
        [0,     0,     1.0],      # row 2: [ 0,  0,  1]
    ], dtype=np.float64)
    return K


# ═════════════════════════════════════════════════════════════════════════════
# 4. ROTATION UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def rodrigues_to_R(rvec):
    """
    Convert a Rodrigues rotation vector to a 3×3 rotation matrix.

    ENCODING: rvec = θ·k̂
      θ = ‖rvec‖   rotation angle in radians
      k̂ = rvec/θ  unit axis of rotation

    RODRIGUES' FORMULA:
      R = I  +  sin(θ)·[k̂]×  +  (1 − cos(θ))·[k̂]×²

    where the 3×3 skew-symmetric cross-product matrix [k̂]× is:
      [k̂]× = |  0   −k_z   k_y |
              |  k_z  0    −k_x |
              | −k_y  k_x   0   |

    Key properties of R:
      det(R) = +1     (proper rotation)
      R·Rᵀ = I        (orthonormal)

    Parameters
    ----------
    rvec : (3,) rotation vector

    Returns
    -------
    R : (3, 3) rotation matrix
    """
    # -- flatten input to always get shape (3,) --------------------------------
    rvec  = np.asarray(rvec, dtype=np.float64).flatten()

    # -- rotation angle θ = ‖rvec‖₂ -------------------------------------------
    theta = np.linalg.norm(rvec)

    # -- degenerate case: near-zero rotation → identity matrix ----------------
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64)

    # -- unit rotation axis k̂ = rvec / θ ----------------------------------------
    k = rvec / theta                     # shape (3,), ‖k‖ = 1

    # -- skew-symmetric cross-product matrix [k̂]× --------------------------------
    # For any vector v:  [k̂]× · v  =  k̂ × v
    K_skew = np.array([
        [ 0,    -k[2],  k[1]],           # row 0
        [ k[2],  0,    -k[0]],           # row 1
        [-k[1],  k[0],  0   ],           # row 2
    ], dtype=np.float64)

    # -- Rodrigues' formula: R = I + sin(θ)·[k̂]× + (1−cos(θ))·[k̂]×² ----------
    # [k̂]×²  =  k̂·k̂ᵀ − I  (can be expanded, but squaring K_skew is equivalent)
    return np.eye(3) + np.sin(theta) * K_skew + (1 - np.cos(theta)) * (K_skew @ K_skew)


def R_to_rodrigues(R):
    """
    Convert a 3×3 rotation matrix to its Rodrigues vector representation.

    INVERSE of rodrigues_to_R.

    RECOVERY FORMULAS:
      From trace:   tr(R) = 1 + 2·cos(θ)
                    θ = arccos((tr(R) − 1) / 2)

      From anti-symmetric part of R:
        R − Rᵀ = 2·sin(θ)·[k̂]×
        → axis components: k̂ ∝ [R₃₂−R₂₃, R₁₃−R₃₁, R₂₁−R₁₂]

      Combined Rodrigues vector:
        rvec = (θ / (2·sin(θ))) · [R₃₂−R₂₃, R₁₃−R₃₁, R₂₁−R₁₂]

    Parameters
    ----------
    R : (3, 3) rotation matrix

    Returns
    -------
    rvec : (3,) Rodrigues rotation vector
    """
    R = np.asarray(R, dtype=np.float64)

    # -- recover rotation angle θ from trace -----------------------------------
    # tr(R) = 1 + 2·cos(θ)  →  cos(θ) = (tr(R)−1)/2
    # clip to [−1, 1] to guard against float errors at θ=0 or θ=π
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta     = np.arccos(cos_theta)              # rotation angle in [0, π]

    # -- degenerate case: θ ≈ 0 → zero rotation vector ----------------------
    if theta < 1e-10:
        return np.zeros(3, dtype=np.float64)

    # -- extract axis direction from anti-symmetric part of R -----------------
    # R − Rᵀ = 2·sin(θ)·[k̂]×   ⟹   the axis components are proportional to:
    #   k_x ∝ R[2,1] − R[1,2],   k_y ∝ R[0,2] − R[2,0],   k_z ∝ R[1,0] − R[0,1]
    axis = np.array(
        [R[2, 1] - R[1, 2],
         R[0, 2] - R[2, 0],
         R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )

    # -- scale axis to length θ (the Rodrigues vector rvec = θ·k̂) --------------
    # The factor θ/(2·sin(θ)) normalizes the axis and scales by θ.
    return (theta / (2.0 * np.sin(theta))) * axis


# ═════════════════════════════════════════════════════════════════════════════
# 5. PER-VIEW EXTRINSICS  (L1)
# ═════════════════════════════════════════════════════════════════════════════

def compute_extrinsics(K, H):
    """
    Recover the per-view pose (R, t) from homography H and intrinsics K.

    THEORY:
      For a planar calibration target (z=0), the relationship between the
      world plane and the image is:

        H  ~  K · [r₁ | r₂ | t]

      where r₁,r₂ are the first two columns of R, and t is the translation.
      Left-multiplying by K⁻¹ isolates the extrinsic part:

        K⁻¹·H = λ·[r₁ | r₂ | t]

      The scale λ is chosen so that ‖r₁‖ = 1 (unit rotation column):
        λ = 1 / ‖K⁻¹·h₁‖

      r₃ = r₁ × r₂  (third rotation column from cross product of the first two)

    REFINEMENT:
      Due to noise, [r₁|r₂|r₃] may not be exactly orthogonal.
      We project it onto SO(3) via SVD:
        [U, S, Vᵀ] = SVD([r₁|r₂|r₃])
        R_clean = U·Vᵀ   (nearest orthogonal matrix in Frobenius norm)

    Parameters
    ----------
    K : (3, 3) intrinsic matrix
    H : (3, 3) homography matrix

    Returns
    -------
    rvec : (3,) Rodrigues rotation vector
    t    : (3,) translation vector (same unit as square_size)
    """
    # -- invert K: K·E = H  →  E = K⁻¹·H  where E = [r₁|r₂|t]·λ -----------
    K_inv = np.linalg.inv(K)

    # -- scale factor: λ = 1 / ‖K⁻¹·h₁‖ so that r₁ has unit norm ------------
    lam = 1.0 / np.linalg.norm(K_inv @ H[:, 0])

    # -- first rotation column r₁ = λ·K⁻¹·h₁ ---------------------------------
    r1 = lam * K_inv @ H[:, 0]          # should have ‖r1‖ ≈ 1

    # -- second rotation column r₂ = λ·K⁻¹·h₂ --------------------------------
    r2 = lam * K_inv @ H[:, 1]          # should have ‖r2‖ ≈ 1

    # -- third rotation column r₃ = r₁ × r₂ (must be orthogonal to r₁ & r₂) -
    r3 = np.cross(r1, r2)

    # -- translation t = λ·K⁻¹·h₃ --------------------------------------------
    t  = lam * K_inv @ H[:, 2]

    # -- project [r₁|r₂|r₃] onto SO(3) via SVD --------------------------------
    # Least-squares nearest proper rotation:  R = U·Vᵀ  if det(U·Vᵀ)=+1
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, r3]))
    R = U @ Vt

    # -- ensure proper rotation (det = +1), not reflection (det = −1) ---------
    # If det < 0, the SVD chose a reflection; negate R and t to fix it.
    if np.linalg.det(R) < 0:
        R = -R
        t = -t

    # -- convert R matrix to Rodrigues vector for compact storage + optimiser --
    return R_to_rodrigues(R), t


# ═════════════════════════════════════════════════════════════════════════════
# 6. FORWARD PROJECTION  (used by L1 linear init and O1 cost function)
# ═════════════════════════════════════════════════════════════════════════════

def project_points(obj_pts, rvec, t, K, dist):
    """
    Project 3-D world points to 2-D image points (pinhole + radial distortion).

    FULL PROJECTION CHAIN:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  World   →  Camera frame  →  Normalize  →  Distort  →  Pixel       │
    │                                                                      │
    │  P_w       P_c = R·P_w + t                                          │
    │            (Xc, Yc, Zc)                                             │
    │                           x_n = Xc/Zc                              │
    │                           y_n = Yc/Zc                              │
    │                                          r² = x_n²+y_n²            │
    │                                          fac= 1+k1·r²+k2·r⁴        │
    │                                          x_d = x_n·fac             │
    │                                          y_d = y_n·fac             │
    │                                                        u=fx·x_d+cx  │
    │                                                        v=fy·y_d+cy  │
    └─────────────────────────────────────────────────────────────────────┘

    Distortion model: Brown-Conrady (radial only, 2 coefficients):
      fac = 1 + k1·r² + k2·r⁴

    Parameters
    ----------
    obj_pts : (N, 3) world-frame 3-D points  (z=0 for planar checkerboard)
    rvec    : (3,) Rodrigues rotation vector
    t       : (3,) translation vector
    K       : (3,3) intrinsic matrix  [[fx,0,cx],[0,fy,cy],[0,0,1]]
    dist    : [k1, k2] radial distortion coefficients

    Returns
    -------
    (N, 2) float64  projected pixel coordinates  [(u₀,v₀), (u₁,v₁), ...]
    """
    # -- STEP 1: Rodrigues rvec → 3×3 rotation matrix -------------------------
    R = rodrigues_to_R(rvec)
    t = np.asarray(t, dtype=np.float64).flatten()   # ensure shape (3,)

    # -- STEP 2: world → camera frame -----------------------------------------
    # P_c = R·P_w + t  for each point
    # R @ obj_pts.T has shape (3, N); transposing gives (N, 3)
    # Broadcasting adds t (shape 3,) to each of the N rows
    pts = (R @ obj_pts.T).T + t          # (N, 3): [Xc, Yc, Zc] per point

    # -- STEP 3: perspective divide → normalized image plane (z=1) ------------
    # x_n = Xc / Zc,   y_n = Yc / Zc
    x_n = pts[:, 0] / pts[:, 2]          # (N,)
    y_n = pts[:, 1] / pts[:, 2]          # (N,)

    # -- STEP 4: radial distortion --------------------------------------------
    # r²  = x_n² + y_n²    (squared radius in normalized plane)
    # fac = 1 + k1·r² + k2·r⁴   (polynomial distortion factor)
    k1, k2 = float(dist[0]), float(dist[1])
    r2  = x_n ** 2 + y_n ** 2            # (N,)  squared radius
    fac = 1.0 + k1 * r2 + k2 * r2**2    # (N,)  distortion factor

    # -- STEP 5: apply distortion factor to normalized coords -----------------
    x_d = x_n * fac                      # distorted normalized x
    y_d = y_n * fac                      # distorted normalized y

    # -- STEP 6: map to pixel coordinates via intrinsic matrix K --------------
    # u = fx·x_d + cx,   v = fy·y_d + cy
    fx, fy = K[0, 0], K[1, 1]            # focal lengths
    cx, cy = K[0, 2], K[1, 2]            # principal point
    u = fx * x_d + cx                    # (N,) pixel column
    v = fy * y_d + cy                    # (N,) pixel row

    return np.stack([u, v], axis=1)      # (N, 2)


# ═════════════════════════════════════════════════════════════════════════════
# 7. LINEAR DISTORTION INITIALIZATION  (L1 → warm-start for O1)
# ═════════════════════════════════════════════════════════════════════════════

def init_distortion(K, Hs, obj_pts_list, img_pts_list):
    """
    Obtain an initial estimate of k1, k2 by linear least-squares.

    IDEA:
      With K fixed and k1=k2=0, project every world point to get the "ideal"
      (undistorted) pixel position (u_id, v_id).  The observed position
      (u_obs, v_obs) differs from (u_id, v_id) due to radial distortion.

      The distortion residual is approximated (linearized around k1=k2=0) by:
        Δu = u_obs − u_id  ≈  fx · x_n · (k1·r² + k2·r⁴)
        Δv = v_obs − v_id  ≈  fy · y_n · (k1·r² + k2·r⁴)

      where x_n = (u_id−cx)/fx, y_n = (v_id−cy)/fy, r² = x_n²+y_n².

      This gives, for each point, 2 equations in [k1, k2]:
        [fx·x_n·r²,  fx·x_n·r⁴] · [k1, k2]ᵀ = Δu
        [fy·y_n·r²,  fy·y_n·r⁴] · [k1, k2]ᵀ = Δv

      Stacking all N×V points gives: A · [k1,k2]ᵀ = b  (overconstrained).
      Solved by: [k1,k2] = (AᵀA)⁻¹Aᵀb  (normal equations, via lstsq).

    Parameters
    ----------
    K            : (3,3) initial intrinsic matrix from solve_intrinsics…
    Hs           : list of n (3,3) homographies (one per view)
    obj_pts_list : list of (N,3) world-point arrays (one per view)
    img_pts_list : list of (N,2) observed image-point arrays (one per view)

    Returns
    -------
    [k1, k2] : list of two floats
    """
    A_rows, b_rows = [], []
    fx, fy = K[0, 0], K[1, 1]    # focal lengths (constant across all views)
    cx, cy = K[0, 2], K[1, 2]    # principal point

    for H, obj_pts, img_pts in zip(Hs, obj_pts_list, img_pts_list):

        # -- STEP 1: recover pose for this view (with the current K) ----------
        rvec, t = compute_extrinsics(K, H)

        # -- STEP 2: ideal (zero-distortion) projection for this view ----------
        # Project all world points ignoring distortion (k1=k2=0).
        proj_ideal = project_points(obj_pts, rvec, t, K, [0.0, 0.0])
        # proj_ideal shape: (N, 2),  columns: [u_ideal, v_ideal]

        # -- STEP 3: build 2 equation rows per point ---------------------------
        for (u_obs, v_obs), (u_id, v_id) in zip(img_pts, proj_ideal):

            # Normalized ideal image coordinates (divide by focal length)
            x_n = (u_id - cx) / fx     # x in normalized image plane
            y_n = (v_id - cy) / fy     # y in normalized image plane

            # Squared radius in normalized plane
            r2  = x_n ** 2 + y_n ** 2

            # Design matrix rows:
            # u-equation:  fx·x_n·r² ↔ k1 coeff,  fx·x_n·r⁴ ↔ k2 coeff
            A_rows.append([fx * x_n * r2,   fx * x_n * r2**2])

            # v-equation:  fy·y_n·r² ↔ k1 coeff,  fy·y_n·r⁴ ↔ k2 coeff
            A_rows.append([fy * y_n * r2,   fy * y_n * r2**2])

            # RHS: observed minus ideal position
            b_rows.append(u_obs - u_id)    # Δu
            b_rows.append(v_obs - v_id)    # Δv

    # -- STEP 4: assemble and solve A·[k1,k2]ᵀ = b --------------------------
    A = np.array(A_rows, dtype=np.float64)    # (2·Σ·N,  2)
    b = np.array(b_rows, dtype=np.float64)    # (2·Σ·N,)

    # lstsq computes the minimum-norm least-squares solution:
    #   [k1,k2] = argmin ‖A·x − b‖²
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return [float(result[0]), float(result[1])]   # [k1, k2]
