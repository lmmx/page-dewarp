"""Parameter optimization routines for page dewarping.

This subpackage provides:
- A function (`draw_correspondences`) to visualize matched points.
- A function (`optimise_params`) that dispatches to the appropriate backend.

The implementation uses JAX (if available) or SciPy based on configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..backends import HAS_JAX, get_default_method
from ..logging_config import get_logger
from ..options import cfg
from ._base import draw_correspondences


if TYPE_CHECKING:
    import numpy as np

__all__ = ["draw_correspondences", "optimise_params"]

logger = get_logger("optimise")


def optimise_params(
    name: str,
    small: np.ndarray,
    dstpoints: np.ndarray,
    span_counts: list[int],
    params: np.ndarray,
    debug_lvl: int,
) -> np.ndarray:
    """Refine the parameter vector for page dewarping via optimization.

    Dispatches to the appropriate backend (JAX or SciPy) based on configuration.

    Args:
        name: Image name for debug output.
        small: Downscaled image for visualization.
        dstpoints: Target points to match.
        span_counts: Number of keypoints per span.
        params: Initial parameter vector.
        debug_lvl: Debug verbosity level.

    Returns:
        Optimized parameter vector.

    """
    method = cfg.OPT_METHOD
    if method == "auto":
        method = get_default_method()

    jax_supported_method = method == "L-BFGS-B"
    use_jax = HAS_JAX and jax_supported_method

    logger.debug(
        "Optimization backend selected",
        extra={
            "method": method,
            "use_jax": use_jax,
            "n_params": len(params),
        },
    )

    if use_jax:
        from ._jax import optimise_params_jax

        return optimise_params_jax(
            name,
            small,
            dstpoints,
            span_counts,
            params,
            debug_lvl,
        )
    else:
        from ._scipy import optimise_params_scipy

        return optimise_params_scipy(
            name,
            small,
            dstpoints,
            span_counts,
            params,
            debug_lvl,
            method,
        )
