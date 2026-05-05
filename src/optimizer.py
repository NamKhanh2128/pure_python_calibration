"""
optimizer.py — O1 (Gradient Descent): Hand-coded Adam optimiser for joint
refinement of camera intrinsics, distortion, and per-view poses.

PROBLEM:
  After the closed-form Zhang initialization (L1), the parameters
  K = [fx, fy, cx, cy], dist = [k1, k2], and per-view {R_i, t_i} are only
  rough estimates (linear approximation, no cross-coupling).  O1 refines them
  jointly by minimizing total reprojection error using gradient descent.

PARAMETER VECTOR layout:
  params = [ fx, fy, cx, cy, k1, k2,          ← indices 0–5
             rx₀, ry₀, rz₀, tx₀, ty₀, tz₀,   ← indices 6–11  (view 0)
             rx₁, ry₁, rz₁, tx₁, ty₁, tz₁,   ← indices 12–17 (view 1)
             … ]                               ← 6 more per additional view

OPTIMIZER: Adam (Kingma & Ba 2015)  — hand-coded, no scipy.
  Adam adapts the learning rate per-parameter using estimates of the first
  and second moments of the gradient, making it robust to different parameter
  scales (e.g. focal length ≈ 1000, k1 ≈ 0.01).
"""

import numpy as np
from .math_core import project_points, rodrigues_to_R, R_to_rodrigues


# ─────────────────────────────────────────────────────────────────────────────
# Constants describing the parameter vector layout
# ─────────────────────────────────────────────────────────────────────────────
_N_INTR = 6    # number of intrinsic/distortion parameters: [fx,fy,cx,cy,k1,k2]
_N_EXTR = 6    # number of extrinsic parameters per view:   [rx,ry,rz,tx,ty,tz]


# ═════════════════════════════════════════════════════════════════════════════
# 1. PACK / UNPACK — flat vector ↔ structured calibration parameters
# ═════════════════════════════════════════════════════════════════════════════

def pack_params(K, dist, rvecs, ts):
    """
    Concatenate all calibration parameters into a single 1-D float64 vector.

    LAYOUT:
      params[0]   = fx       ← K[0,0]
      params[1]   = fy       ← K[1,1]
      params[2]   = cx       ← K[0,2]
      params[3]   = cy       ← K[1,2]
      params[4]   = k1       ← dist[0]
      params[5]   = k2       ← dist[1]
      params[6:9] = rvec₀    ← Rodrigues rotation of view 0
      params[9:12]= t₀       ← translation of view 0
      params[12:18]= (rvec₁, t₁)   ← view 1
      …             (repeating) …

    Total length = 6 + 6·n_views.

    Parameters
    ----------
    K     : (3,3) intrinsic matrix
    dist  : [k1, k2] radial distortion
    rvecs : list of n (3,) Rodrigues vectors
    ts    : list of n (3,) translation vectors

    Returns
    -------
    params : (6 + 6·n,) float64 array
    """
    # -- collect the 6 intrinsic/distortion parameters -------------------------
    intr = [
        K[0, 0],          # fx
        K[1, 1],          # fy
        K[0, 2],          # cx
        K[1, 2],          # cy
        float(dist[0]),   # k1
        float(dist[1]),   # k2
    ]

    # -- append extrinsics: for each view, [rx,ry,rz,tx,ty,tz] ----------------
    extr = []
    for rv, t in zip(rvecs, ts):
        extr.extend(list(np.asarray(rv).flatten()))   # [rx, ry, rz]
        extr.extend(list(np.asarray(t).flatten()))    # [tx, ty, tz]

    return np.array(intr + extr, dtype=np.float64)   # (6+6n,)


def unpack_params(params, n_views):
    """
    Restore structured calibration parameters from the flat vector.

    Inverse of pack_params.

    Parameters
    ----------
    params  : (6+6n,) flat float64 array (as produced by pack_params/adam_optimize)
    n_views : number of calibration views n

    Returns
    -------
    K     : (3,3) intrinsic matrix reconstructed from [fx,fy,cx,cy]
    dist  : [k1, k2]
    rvecs : list of n (3,) Rodrigues rotation vectors
    ts    : list of n (3,) translation vectors
    """
    # -- extract the first 6 elements: intrinsics + distortion ----------------
    fx, fy, cx, cy, k1, k2 = params[:_N_INTR]

    # -- rebuild the 3×3 upper-triangular intrinsic matrix --------------------
    # K = | fx  0  cx |
    #     |  0  fy cy |
    #     |  0   0  1 |
    K = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = [k1, k2]

    # -- extract per-view extrinsics: 6 parameters per view -------------------
    rvecs, ts = [], []
    for i in range(n_views):
        base = _N_INTR + i * _N_EXTR    # starting index for view i
        rvecs.append(params[base    : base + 3])   # [rx, ry, rz]
        ts.append(   params[base + 3: base + 6])   # [tx, ty, tz]

    return K, dist, rvecs, ts


# ═════════════════════════════════════════════════════════════════════════════
# 2. COST FUNCTION — Reprojection Error
# ═════════════════════════════════════════════════════════════════════════════

def reprojection_error_vector(params, obj_pts_list, img_pts_list):
    """
    Compute the flat residual vector for all observed corners in all views.

    DEFINITION:
      For each view i and each corner j, the residual is:
        Δu_ij = proj_u(X_j; params) − u_ij_obs
        Δv_ij = proj_v(X_j; params) − v_ij_obs

    The residuals are stacked into a single 1-D vector:
      [Δu₀₀, Δv₀₀, Δu₀₁, Δv₀₁, …,  Δu₁₀, Δv₁₀, …]

    Total length = 2 · Σᵢ Nᵢ  where Nᵢ is the number of corners in view i.

    Parameters
    ----------
    params       : flat parameter vector (from pack_params / adam_optimize)
    obj_pts_list : list of (N, 3) world-point arrays  (one per view)
    img_pts_list : list of (N, 2) observed image-point arrays (one per view)

    Returns
    -------
    residuals : (2·Σ·N,) float64 array of signed pixel residuals
    """
    n_views = len(obj_pts_list)

    # -- decode parameters ----------------------------------------------------
    K, dist, rvecs, ts = unpack_params(params, n_views)

    residuals = []
    for obj_pts, img_pts, rv, t in zip(obj_pts_list, img_pts_list, rvecs, ts):
        # project all world points of this view using current parameters
        proj = project_points(obj_pts, rv, t, K, dist)   # (N, 2)

        # residual = projected − observed
        diff = proj - img_pts                             # (N, 2)

        # flatten to [Δu₀, Δv₀, Δu₁, Δv₁, …] for this view
        residuals.append(diff.flatten())                  # (2N,)

    return np.concatenate(residuals)   # (2·Σ·N,)


def reprojection_rmse(params, obj_pts_list, img_pts_list):
    """
    Compute the scalar Root-Mean-Square reprojection Error (RMSE) in pixels.

    FORMULA:
      RMSE = √( (1/(2·Σ·N)) · Σᵢⱼ (Δuᵢⱼ² + Δvᵢⱼ²) )
           = √( mean(residuals²) )

    This is the single scalar that O1 minimizes.
    A well-calibrated camera typically achieves RMSE < 1.0 pixel.

    Parameters
    ----------
    params       : flat parameter vector
    obj_pts_list : list of (N,3) world-point arrays
    img_pts_list : list of (N,2) observed image-point arrays

    Returns
    -------
    rmse : float  (in pixel units)
    """
    # -- get all residuals as a flat vector -----------------------------------
    r = reprojection_error_vector(params, obj_pts_list, img_pts_list)

    # -- RMSE = √(mean(r²)) = √( (Σ rᵢ²) / (2·Σ·N) ) -----------------------
    return float(np.sqrt(np.mean(r ** 2)))


# ═════════════════════════════════════════════════════════════════════════════
# 3. GRADIENT via Central Finite Differences
# ═════════════════════════════════════════════════════════════════════════════

def _numerical_gradient(params, obj_pts_list, img_pts_list, eps=1e-5):
    """
    Compute the gradient of reprojection_rmse w.r.t. every parameter
    using the central finite-difference formula.

    FORMULA (for each parameter pᵢ):
      ∂RMSE/∂pᵢ  ≈  [RMSE(p + ε·eᵢ) − RMSE(p − ε·eᵢ)] / (2ε)

    where eᵢ is the i-th standard basis vector.

    Central differences are second-order accurate (O(ε²) error) vs the
    forward-difference which is only first-order accurate (O(ε) error).

    WHY NUMERICAL: The RMSE involves Rodrigues → R → projection → rounding
    in a complex non-linear chain.  Deriving an analytic Jacobian requires
    significant effort; numerical differentiation is simpler and sufficient
    for the Adam optimizer used here.

    Parameters
    ----------
    params       : (6+6n,) flat parameter vector  (current Adam iterate)
    obj_pts_list : list of (N,3) world-point arrays
    img_pts_list : list of (N,2) observed image-point arrays
    eps          : finite-difference step size  (default 1e-5)

    Returns
    -------
    grad : (6+6n,) float64 gradient vector  ∂RMSE/∂params
    """
    grad = np.zeros_like(params)

    for i in range(len(params)):
        # -- perturb parameter i upward: p_plus[i] = pᵢ + ε ------------------
        p_p     = params.copy()
        p_p[i] += eps

        # -- perturb parameter i downward: p_minus[i] = pᵢ − ε ---------------
        p_m     = params.copy()
        p_m[i] -= eps

        # -- central difference ------------------------------------------------
        # ∂f/∂pᵢ ≈ [f(p+εeᵢ) − f(p−εeᵢ)] / (2ε)
        grad[i] = (
            reprojection_rmse(p_p, obj_pts_list, img_pts_list)
          - reprojection_rmse(p_m, obj_pts_list, img_pts_list)
        ) / (2.0 * eps)

    return grad   # (6+6n,)


# ═════════════════════════════════════════════════════════════════════════════
# 4. O1 — ADAM OPTIMIZER  (hand-coded, no scipy)
#    Reference: Kingma & Ba "Adam: A Method for Stochastic Optimization" (2015)
# ═════════════════════════════════════════════════════════════════════════════

def lm_optimize(
    params,
    obj_pts_list,
    img_pts_list,
    lam_init  = 1e-3,
    lam_mult  = 10.0,
    max_iter  = 100,
    tol       = 1e-7,
    verbose   = True,
    log_every = 10,
):
    """
    Minimize reprojection RMSE using the Levenberg-Marquardt (LM) algorithm.
    This exactly matches Zhang's non-linear optimization step.

    LM UPDATE RULE:
      Δp = (Jᵀ·J + λ·diag(Jᵀ·J))⁻¹ · Jᵀ·r
    """
    p = params.copy().astype(np.float64)
    
    def calc_residuals(p_curr):
        return reprojection_error_vector(p_curr, obj_pts_list, img_pts_list)
        
    def calc_jacobian(p_curr, r_curr):
        eps = 1e-5
        # Jacobian shape: (num_residuals, num_params)
        J = np.zeros((len(r_curr), len(p_curr)), dtype=np.float64)
        for i in range(len(p_curr)):
            p_plus = p_curr.copy()
            p_plus[i] += eps
            r_plus = calc_residuals(p_plus)
            J[:, i] = (r_plus - r_curr) / eps
        return J

    r = calc_residuals(p)
    rmse = np.sqrt(np.mean(r**2))
    lam = lam_init
    
    loss_hist = [rmse]
    
    if verbose:
        print(f"    [LM] start  RMSE = {rmse:.6f} px  "
              f"(params={len(p)}, views={len(obj_pts_list)})")

    for it in range(1, max_iter + 1):
        J = calc_jacobian(p, r)
        JtJ = J.T @ J
        Jtr = J.T @ r
        
        # Levenberg-Marquardt step
        step_accepted = False
        for _ in range(10): # try up to 10 lambda increases
            # H = JᵀJ + λ·diag(JᵀJ)
            H = JtJ + lam * np.diag(np.diag(JtJ))
            
            # Guard against completely zero diagonal
            if np.all(np.diag(H) == 0):
                H = JtJ + lam * np.eye(len(p))
                
            try:
                dp = np.linalg.solve(H, -Jtr)
            except np.linalg.LinAlgError:
                lam *= lam_mult
                continue
                
            p_new = p + dp
            r_new = calc_residuals(p_new)
            rmse_new = float(np.sqrt(np.mean(r_new**2)))
            
            if rmse_new < rmse:
                p = p_new
                r = r_new
                rmse = rmse_new
                lam = max(lam / lam_mult, 1e-7)
                step_accepted = True
                break
            else:
                lam *= lam_mult
                
        loss_hist.append(rmse)
        
        if verbose and it % log_every == 0:
            print(f"    [LM iter {it:3d}] RMSE = {rmse:.6f} px (λ={lam:.1e})")
            
        if not step_accepted or np.linalg.norm(dp) < tol:
            if verbose:
                print(f"    [LM] converged at iter {it},  RMSE = {rmse:.6f} px")
            break

    return p, loss_hist
