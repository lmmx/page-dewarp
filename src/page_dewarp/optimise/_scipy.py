"""SciPy-based optimization backend for page dewarping."""

from __future__ import annotations

import sys
from datetime import datetime as dt

import numpy as np
from cv2 import Rodrigues
from scipy.optimize import minimize

from ..debug_utils import debug_show
from ..keypoints import make_keypoint_index, project_keypoints
from ..options import cfg
from ._base import draw_correspondences, make_objective


__all__ = ["optimise_params_scipy"]


def optimise_params_scipy(
    name: str,
    small: np.ndarray,
    dstpoints: np.ndarray,
    span_counts: list[int],
    params: np.ndarray,
    debug_lvl: int,
    method: str,
) -> np.ndarray:
    """Refine the parameter vector using SciPy optimization.

    Args:
        name: Image name for debug output.
        small: Downscaled image for visualization.
        dstpoints: Target points to match.
        span_counts: Number of keypoints per span.
        params: Initial parameter vector.
        debug_lvl: Debug verbosity level.
        method: Optimization method (e.g., 'Powell', 'L-BFGS-B').

    Returns:
        Optimized parameter vector.

    """
    keypoint_index = make_keypoint_index(span_counts)
    objective = make_objective(
        dstpoints,
        keypoint_index,
        cfg.SHEAR_COST,
        slice(*cfg.RVEC_IDX),
    )

    print("  initial objective is", objective(params))
    if debug_lvl >= 1:
        projpts = project_keypoints(params, keypoint_index)
        display = draw_correspondences(small, dstpoints, projpts)
        debug_show(name, 4, "keypoints before", display)

    print(f"  optimizing {len(params)} parameters...")

    # Hint about JAX for L-BFGS-B
    if method == "L-BFGS-B":
        print(
            "OPT_METHOD='L-BFGS-B' can use JAX for fast autodiff. "
            "Install with: pip install page-dewarp[jax]",
            file=sys.stderr,
        )

    start = dt.now()
    result = minimize(
        objective,
        params,
        method=method,
        options={"maxiter": cfg.OPT_MAX_ITER},
    )
    elapsed = (dt.now() - start).total_seconds()
    print(f"  optimization ({method}) took {elapsed:.2f}s, {result.nfev} evals")

    print(f"  final objective is {result.fun:.6f}")
    params = result.x

    if debug_lvl >= 1:
        _print_diagnostics(params, keypoint_index, small, dstpoints, name)

    return params


def _print_diagnostics(
    params: np.ndarray,
    keypoint_index: np.ndarray,
    small: np.ndarray,
    dstpoints: np.ndarray,
    name: str,
) -> None:
    """Print parameter diagnostics and show debug visualization."""
    projpts = project_keypoints(params, keypoint_index)
    display = draw_correspondences(small, dstpoints, projpts)
    debug_show(name, 5, "keypoints after", display)

    print("  === Parameter Diagnostics ===")
    rvec = params[slice(*cfg.RVEC_IDX)]
    tvec = params[slice(*cfg.TVEC_IDX)]
    alpha, beta = params[slice(*cfg.CUBIC_IDX)]

    print(f"  Rotation vector: {rvec}")
    print(f"  Rotation angles (degrees): {np.degrees(rvec)}")
    print(f"  Translation vector: {tvec}")
    print(f"  Cubic params - alpha: {alpha}, beta: {beta}")

    R, _ = Rodrigues(rvec)
    print(f"  Rotation matrix determinant: {np.linalg.det(R)}")
    print(f"  Rotation matrix condition number: {np.linalg.cond(R)}")
