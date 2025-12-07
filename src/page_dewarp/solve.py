"""Parameter initialization and solve routines for page flattening.

This module contains a function (`get_default_params`) that:

- Uses four corner correspondences to estimate rotation/translation (solvePnP).
- Optionally uses Toeplitz-based curl estimation for initial cubic slopes.
- Includes default cubic slopes and any y/x coordinates from sampled spans.
"""

import numpy as np
from cv2 import solvePnP

from .curl_estimation import estimate_cubic_params_from_spans
from .options import cfg
from .options.k_opt import K


__all__ = ["get_default_params"]


def get_default_params(
    corners: np.ndarray,
    ycoords: np.ndarray,
    xcoords: list[np.ndarray],
    span_points: list[np.ndarray] | None = None,
) -> tuple[tuple[float, float], list[int], np.ndarray]:
    """Assemble an initial parameter vector for page flattening.

    Args:
        corners: A (4,1,2) array of corner points in image coords.
        ycoords: A 1D array of average vertical positions (per span).
        xcoords: A list of x-coordinates arrays for each span.
        span_points: Optional list of span point arrays for Toeplitz-based
            curl estimation. If None, defaults to zero slopes.

    Returns:
        A tuple of:
            (page_width, page_height): The estimated physical page dims.
            span_counts: A list with the length of each `xcoords[i]`.
            params: A 1D array combining rotation, translation, cubic slopes, etc.

    """
    page_width, page_height = (np.linalg.norm(corners[i] - corners[0]) for i in (1, -1))

    # Estimate cubic slopes using Toeplitz analysis if span data available
    if span_points is not None and len(span_points) > 0:
        alpha, beta = estimate_cubic_params_from_spans(
            span_points,
            ycoords,
            page_width,
        )
        cubic_slopes = [alpha, beta]
        if cfg.DEBUG_LEVEL >= 1:
            print(f"  Toeplitz curl estimate: α={alpha:.4f}, β={beta:.4f}")
    else:
        cubic_slopes = [0.0, 0.0]  # Fallback to flat page assumption

    # Object points of a flat page in 3D coordinates
    corners_object3d = np.array(
        [
            [0, 0, 0],
            [page_width, 0, 0],
            [page_width, page_height, 0],
            [0, page_height, 0],
        ],
    )
    # Estimate rotation and translation from four 2D-to-3D point correspondences
    _, rvec, tvec = solvePnP(corners_object3d, corners, K(cfg=cfg), np.zeros(5))

    span_counts = [len(xc) for xc in xcoords]
    params = np.hstack(
        (
            np.array(rvec).flatten(),
            np.array(tvec).flatten(),
            np.array(cubic_slopes).flatten(),
            ycoords.flatten(),
        )
        + tuple(xcoords),
    )
    return (page_width, page_height), span_counts, params
