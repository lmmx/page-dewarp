"""Shared utilities for optimization backends."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from cv2 import LINE_AA, circle, line

from ..normalisation import norm2pix
from ..simple_utils import fltp


__all__ = ["draw_correspondences", "make_objective"]


def draw_correspondences(
    img: np.ndarray,
    dstpoints: np.ndarray,
    projpts: np.ndarray,
) -> np.ndarray:
    """Draw matching points (projected vs. desired) on a copy of the image."""
    display = img.copy()
    dstpoints = norm2pix(img.shape, dstpoints, True)
    projpts = norm2pix(img.shape, projpts, True)
    for pts, color in [(projpts, (255, 0, 0)), (dstpoints, (0, 0, 255))]:
        for point in pts:
            circle(display, fltp(point), 3, color, -1, LINE_AA)
    for point_a, point_b in zip(projpts, dstpoints):
        line(display, fltp(point_a), fltp(point_b), (255, 255, 255), 1, LINE_AA)
    return display


def make_objective(
    dstpoints: np.ndarray,
    keypoint_index: np.ndarray,
    shear_cost: float,
    rvec_slice: slice,
) -> Callable[[np.ndarray], float]:
    """Create an objective function for scipy.optimize.minimize.

    Args:
        dstpoints: Target points to match.
        keypoint_index: Index array for keypoint extraction.
        shear_cost: Cost coefficient for shear penalty.
        rvec_slice: Slice object for extracting rotation vector from params.

    Returns:
        Objective function suitable for scipy.optimize.minimize.

    """
    from ..keypoints import project_keypoints

    def objective(pvec: np.ndarray) -> float:
        ppts = project_keypoints(pvec, keypoint_index)
        error = np.sum((dstpoints - ppts) ** 2)
        if shear_cost > 0.0:
            rvec = pvec[rvec_slice]
            rotation_penalty = shear_cost * rvec[0] ** 2
            return error + rotation_penalty
        else:
            return error

    return objective
