"""Project 3D points into image space using a cubic warp model.

This module provides a `project_xy` function, which:

- Constructs a cubic polynomial from alpha/beta slopes.
- Uses OpenCV's `projectPoints` to map the resulting 3D points into image coordinates.
"""

import numpy as np
from cv2 import projectPoints

from .options import cfg
from .options.k_opt import K


__all__ = ["project_xy"]


def project_xy(xy_coords: np.ndarray, pvec: np.ndarray) -> np.ndarray:
    """Get cubic polynomial coefficients for the specified boundary conditions.

    f(0) = 0, f'(0) = alpha
    f(1) = 0, f'(1) = beta.

    The polynomial is used to determine the z-coordinate of each point in `xy_coords`,
    which is then projected into image space via `projectPoints`.

    Args:
        xy_coords: An (N,2) array of (x, y) points (float32).
        pvec: The parameter vector, from which we extract alpha/beta and rvec/tvec.

    Returns:
        An (N,1,2) array of the projected 2D image points.

    """
    alpha, beta = tuple(pvec[slice(*cfg.CUBIC_IDX)])

    # Clamp the cubic‐warp coefficients to a safe range.
    # Prevents runaway horizontal stretching, see:
    # https://github.com/lmmx/page-dewarp/issues/17
    alpha = np.clip(alpha, -0.5, 0.5)
    beta = np.clip(beta, -0.5, 0.5)

    poly = np.array([alpha + beta, -2 * alpha - beta, alpha, 0])

    xy_coords = xy_coords.reshape((-1, 2))
    z_coords = np.polyval(poly, xy_coords[:, 0])

    objpoints = np.hstack((xy_coords, z_coords.reshape((-1, 1))))
    image_points, _ = projectPoints(
        objpoints,
        pvec[slice(*cfg.RVEC_IDX)],
        pvec[slice(*cfg.TVEC_IDX)],
        K(cfg=cfg),
        np.zeros(5),
    )
    return image_points
