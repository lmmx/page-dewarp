"""Parameter initialization and solve routines for page flattening.

This module contains a function (`get_default_params`) that:

- Uses four corner correspondences to estimate rotation/translation (solvePnP).
- Includes default cubic slopes and any y/x coordinates from sampled spans.
"""

import numpy as np
from cv2 import solvePnP

from .options import cfg
from .options.k_opt import K


__all__ = ["get_default_params"]


def get_default_params(
    corners: np.ndarray,
    ycoords: np.ndarray,
    xcoords: list[np.ndarray],
) -> tuple[tuple[float, float], list[int], np.ndarray]:
    """Assemble an initial parameter vector for page flattening.

    Args:
        corners: A (4,1,2) array of corner points in image coords.
        ycoords: A 1D array of average vertical positions (per span).
        xcoords: A list of x-coordinates arrays for each span.

    Returns:
        A tuple of:
            (page_width, page_height): The estimated physical page dims.
            span_counts: A list with the length of each `xcoords[i]`.
            params: A 1D array combining rotation, translation, cubic slopes, etc.

    """
    page_width, page_height = (np.linalg.norm(corners[i] - corners[0]) for i in (1, -1))
    cubic_slopes = [0.0, 0.0]  # initial guess for the cubic has no slope

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
