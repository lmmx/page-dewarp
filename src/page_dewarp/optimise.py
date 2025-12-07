"""Parameter optimization routines for page dewarping.

This module provides:

- A function (`draw_correspondences`) to visualize matched points (dstpoints/projpts).
- A function (`optimise_params`) that uses an objective function and scipy's `minimize`
  to refine the parameter vector for page dewarping.
"""

from datetime import datetime as dt

import numpy as np
from cv2 import LINE_AA, Rodrigues, circle, line
from scipy.optimize import minimize

from .debug_utils import debug_show
from .keypoints import make_keypoint_index, project_keypoints
from .normalisation import norm2pix
from .options import cfg
from .simple_utils import fltp


__all__ = ["draw_correspondences", "optimise_params"]


def draw_correspondences(
    img: np.ndarray,
    dstpoints: np.ndarray,
    projpts: np.ndarray,
) -> np.ndarray:
    """Draw matching points (projected vs. desired) on a copy of the image.

    Args:
        img: The base image to overlay points onto.
        dstpoints: The "destination" points (desired).
        projpts: The "projected" points (computed from a parameter vector).

    Returns:
        A copy of `img` with circles for each set of points and lines connecting them.

    """
    display = img.copy()
    dstpoints = norm2pix(img.shape, dstpoints, True)
    projpts = norm2pix(img.shape, projpts, True)
    for pts, color in [(projpts, (255, 0, 0)), (dstpoints, (0, 0, 255))]:
        for point in pts:
            circle(display, fltp(point), 3, color, -1, LINE_AA)
    for point_a, point_b in zip(projpts, dstpoints):
        line(display, fltp(point_a), fltp(point_b), (255, 255, 255), 1, LINE_AA)
    return display


def optimise_params(
    name: str,
    small: np.ndarray,
    dstpoints: np.ndarray,
    span_counts: list[int],
    params: np.ndarray,
    debug_lvl: int,
) -> np.ndarray:
    """Refine the parameter vector (params) for page dewarping via optimization.

    Uses scipy's Powell method to minimize the squared distance between
    `dstpoints` (desired) and the projected points (via `project_keypoints`).

    Args:
        name: A string identifier for debugging/logging.
        small: A downsampled image for optional visualization.
        dstpoints: A NumPy array of target 2D points (normalized or pixel coords).
        span_counts: A list of how many keypoints belong to each text/line span.
        params: An initial parameter vector (rotation, translation, cubic slopes, etc.).
        debug_lvl: The debug verbosity level.

    Returns:
        A 1D NumPy array of optimized parameters (same shape as `params`).

    """
    keypoint_index = make_keypoint_index(span_counts)

    def objective(pvec: np.ndarray) -> float:
        ppts = project_keypoints(pvec, keypoint_index)
        return np.sum((dstpoints - ppts) ** 2)

    print("  initial objective is", objective(params))
    if debug_lvl >= 1:
        projpts = project_keypoints(params, keypoint_index)
        display = draw_correspondences(small, dstpoints, projpts)
        debug_show(name, 4, "keypoints before", display)

    print("  optimizing", len(params), "parameters...")
    start = dt.now()
    min_opts = {"maxiter": cfg.OPT_MAX_ITER}
    res = minimize(objective, params, method="Powell", options=min_opts)
    end = dt.now()
    print(f"  optimization took {round((end - start).total_seconds(), 2)} sec.")
    print(f"  final objective is {res.fun}")

    params = res.x
    if debug_lvl >= 1:
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

        # Check rotation matrix conditioning
        R, _ = Rodrigues(rvec)
        print(f"  Rotation matrix determinant: {np.linalg.det(R)}")
        print(f"  Rotation matrix condition number: {np.linalg.cond(R)}")

    return params
