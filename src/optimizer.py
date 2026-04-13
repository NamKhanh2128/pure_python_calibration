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

def adam_optimize(
    params,
    obj_pts_list,
    img_pts_list,
    lr        = 1e-3,
    beta1     = 0.9,
    beta2     = 0.999,
    epsilon   = 1e-8,
    max_iter  = 500,
    tol       = 1e-7,
    verbose   = True,
    log_every = 50,
):
    """
    Minimize reprojection RMSE using the Adam first-order gradient optimizer.

    ADAM UPDATE RULE (at each iteration t):
      g    = ∂RMSE/∂p                     (gradient, computed numerically)

      m    = β₁·m + (1−β₁)·g              (1st-moment estimate: EMA of g)
      v    = β₂·v + (1−β₂)·g²             (2nd-moment estimate: EMA of g²)

      m̂  = m / (1 − β₁ᵗ)                 (bias-corrected 1st moment)
      v̂  = v / (1 − β₂ᵗ)                 (bias-corrected 2nd moment)

      p   ← p − lr · m̂ / (√v̂ + ε)        (parameter update)

    WHY ADAM:
      • m̂ / (√v̂) is an adaptive per-parameter step: parameters with large
        historic gradients get smaller updates (self-tuning).
      • Bias correction terms (1−β₁ᵗ) and (1−β₂ᵗ) prevent under-estimation
        of the moments during the early iterations when m and v are near 0.
      • Works well when parameters have very different scales (e.g. fx≈800,
        k1≈0.01) which is exactly the situation in camera calibration.

    Parameters
    ----------
    params       : initial flat parameter vector  (from pack_params)
    obj_pts_list : list of (N,3) world-point arrays (one per view)
    img_pts_list : list of (N,2) observed image-point arrays (one per view)
    lr           : learning rate α (default 1e-3)
    beta1        : exponential decay for 1st moment  (default 0.9)
    beta2        : exponential decay for 2nd moment  (default 0.999)
    epsilon      : small constant for numerical stability  (default 1e-8)
    max_iter     : maximum number of Adam update steps  (default 500)
    tol          : stop if |RMSE[t] − RMSE[t-1]| < tol  (default 1e-7)
    verbose      : whether to print progress
    log_every    : print frequency in iterations

    Returns
    -------
    best_params  : (6+6n,) parameter vector that achieved the lowest RMSE
    loss_history : list of RMSE values, one per iteration
    """
    # -- INITIALIZATION -------------------------------------------------------
    p = params.copy().astype(np.float64)  # working parameter copy

    # 1st-moment vector m, initialized to 0 (unbiased before any update)
    m = np.zeros_like(p)

    # 2nd-moment vector v, initialized to 0
    v = np.zeros_like(p)

    loss_hist = []                        # track RMSE per iteration

    # evaluate initial loss
    best_loss = reprojection_rmse(p, obj_pts_list, img_pts_list)
    best_p    = p.copy()                  # best params seen so far

    if verbose:
        print(f"    [Adam] start  RMSE = {best_loss:.6f} px  "
              f"(params={len(p)}, views={len(obj_pts_list)})")

    # -- MAIN LOOP ------------------------------------------------------------
    for t in range(1, max_iter + 1):

        # ── STEP A: compute gradient via central finite differences ───────────
        # g[i] ≈ [RMSE(p+ε·eᵢ) − RMSE(p−ε·eᵢ)] / (2ε)
        g = _numerical_gradient(p, obj_pts_list, img_pts_list)

        # ── STEP B: update biased 1st moment  m = β₁·m + (1−β₁)·g ───────────
        # m is an exponential moving average of the gradient history.
        m = beta1 * m + (1.0 - beta1) * g

        # ── STEP C: update biased 2nd moment  v = β₂·v + (1−β₂)·g² ──────────
        # v is an exponential moving average of the squared gradient.
        v = beta2 * v + (1.0 - beta2) * g ** 2

        # ── STEP D: bias correction ───────────────────────────────────────────
        # At iteration t, m and v are initialized at 0, so they are biased
        # towards zero for small t.  Correcting: m̂ = m/(1−β₁ᵗ)
        m_hat = m / (1.0 - beta1 ** t)   # bias-corrected 1st moment
        v_hat = v / (1.0 - beta2 ** t)   # bias-corrected 2nd moment

        # ── STEP E: parameter update ──────────────────────────────────────────
        # p ← p − lr · m̂ / (√v̂ + ε)
        # The adaptive step size per parameter is: lr·|m̂ᵢ| / (√v̂ᵢ + ε)
        # ε prevents division by zero when v̂ ≈ 0.
        p -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

        # ── STEP F: evaluate updated RMSE and track best ──────────────────────
        loss = reprojection_rmse(p, obj_pts_list, img_pts_list)
        loss_hist.append(loss)

        # keep the best parameter set seen across all iterations
        if loss < best_loss:
            best_loss = loss
            best_p    = p.copy()

        # ── STEP G: optional progress log ─────────────────────────────────────
        if verbose and t % log_every == 0:
            print(f"    [Adam iter {t:4d}] RMSE = {loss:.6f} px")

        # ── STEP H: convergence check ─────────────────────────────────────────
        # Stop early if the improvement between consecutive iterations is below tol.
        # |RMSE[t] − RMSE[t-1]| < tol  ⟹  converged
        if len(loss_hist) > 1 and abs(loss_hist[-2] - loss_hist[-1]) < tol:
            if verbose:
                print(f"    [Adam] converged at iter {t},  RMSE = {loss:.6f} px")
            break

    return best_p, loss_hist   # best_p achieves best_loss; loss_hist for plotting
