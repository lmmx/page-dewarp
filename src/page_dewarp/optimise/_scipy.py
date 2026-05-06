"""SciPy-based optimization backend for page dewarping."""

from __future__ import annotations

from datetime import datetime as dt

import numpy as np
from cv2 import Rodrigues
from scipy.optimize import minimize

from ..debug_utils import debug_show
from ..keypoints import make_keypoint_index, project_keypoints
from ..logging_config import get_logger
from ..options import cfg
from ._base import draw_correspondences, make_objective


__all__ = ["optimise_params_scipy"]

logger = get_logger("optimise.scipy")


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

    initial_obj = objective(params)
    logger.info(
        "Optimization starting",
        extra={
            "file": name,
            "n_params": len(params),
            "method": method,
            "initial_objective": initial_obj,
        },
    )

    if debug_lvl >= 1:
        projpts = project_keypoints(params, keypoint_index)
        display = draw_correspondences(small, dstpoints, projpts)
        debug_show(name, 4, "keypoints before", display)

    # Hint about JAX for L-BFGS-B
    if method == "L-BFGS-B":
        logger.info(
            "L-BFGS-B can use JAX for faster autodiff. "
            "Install with: pip install page-dewarp[jax]",
        )

    start = dt.now()
    result = minimize(
        objective,
        params,
        method=method,
        options={"maxiter": cfg.OPT_MAX_ITER},
    )
    elapsed = (dt.now() - start).total_seconds()

    logger.info(
        "Optimization complete",
        extra={
            "file": name,
            "method": method,
            "backend": "scipy",
            "elapsed_s": round(elapsed, 2),
            "n_evals": result.nfev,
            "final_objective": round(result.fun, 6),
        },
    )

    params = result.x

    if debug_lvl >= 1:
        _log_diagnostics(name, params, keypoint_index, small, dstpoints)

    return params


def _log_diagnostics(
    name: str,
    params: np.ndarray,
    keypoint_index: np.ndarray,
    small: np.ndarray,
    dstpoints: np.ndarray,
) -> None:
    """Log parameter diagnostics and show debug visualization."""
    projpts = project_keypoints(params, keypoint_index)
    display = draw_correspondences(small, dstpoints, projpts)
    debug_show(name, 5, "keypoints after", display)

    rvec = params[slice(*cfg.RVEC_IDX)]
    tvec = params[slice(*cfg.TVEC_IDX)]
    alpha, beta = params[slice(*cfg.CUBIC_IDX)]

    R, _ = Rodrigues(rvec)

    logger.debug(
        "Optimization diagnostics",
        extra={
            "file": name,
            "rvec": rvec.tolist(),
            "rvec_degrees": np.degrees(rvec).tolist(),
            "tvec": tvec.tolist(),
            "alpha": alpha,
            "beta": beta,
            "rotation_det": np.linalg.det(R),
            "rotation_cond": np.linalg.cond(R),
        },
    )
