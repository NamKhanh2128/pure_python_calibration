"""pure_python_calibration.src — pure-NumPy camera calibration stack.

Stack:
  F1 (Harris)          feature_extractor.py
  L1 (SVD)             math_core.py
  O1 (Gradient Desc.)  optimizer.py
  R2 (Inverse+Nearest) undistorter.py
"""

from .feature_extractor import (
    extract_all_views,
    load_image_gray,
    load_image_rgb,
    harris_response,
    detect_corners,
    subpixel_refine,
    order_checkerboard_corners,
    make_object_points,
)
from .math_core import (
    normalize_points,
    dlt_homography,
    compute_v_ij,
    solve_intrinsics_from_homographies,
    rodrigues_to_R,
    R_to_rodrigues,
    compute_extrinsics,
    project_points,
    init_distortion,
)
from .optimizer import (
    pack_params,
    unpack_params,
    reprojection_rmse,
    lm_optimize,
)
from .undistorter import (
    undistort_image,
    undistort_image_file,
)

__all__ = [
    # F1
    "extract_all_views", "load_image_gray", "load_image_rgb",
    "harris_response", "detect_corners", "subpixel_refine",
    "order_checkerboard_corners", "make_object_points",
    # L1
    "normalize_points", "dlt_homography", "compute_v_ij",
    "solve_intrinsics_from_homographies", "rodrigues_to_R",
    "R_to_rodrigues", "compute_extrinsics", "project_points",
    "init_distortion",
    # O1
    "pack_params", "unpack_params", "reprojection_rmse", "adam_optimize",
    # R2
    "undistort_image", "undistort_image_file",
]
