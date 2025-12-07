"""Keypoint indexing and projection utilities.

This module provides helper functions to:

- Generate a "keypoint index" array from span counts (for referencing parameters).
- Project keypoints (in a parameter vector) into 2D coordinates using a cubic model.
"""

import numpy as np

from .projection import project_xy


__all__ = ["make_keypoint_index", "project_keypoints"]


def make_keypoint_index(span_counts: list[int]) -> np.ndarray:
    """Construct an index array to map spans to keypoint parameters.

    Given a list of `span_counts`, where each value indicates how many keypoints
    belong to that span, this function returns a 2D integer array of shape
    `(total_points + 1, 2)`. The indexing scheme is used elsewhere for referencing
    keypoints in a parameter vector.

    Args:
        span_counts: A list of integers, each giving the number of keypoints in a span.

    Returns:
        A 2D numpy array of shape (N+1, 2), where N is the sum of `span_counts`.
        It encodes both a "point index" (in `[:, 0]`) and a "span index" (in `[:, 1]`).
        The first row is reserved or initialized to zero, and subsequent rows are
        populated to align with internal parameter-vector indexing conventions.

    """
    nspans, npts = len(span_counts), sum(span_counts)
    keypoint_index = np.zeros((npts + 1, 2), dtype=int)
    start = 1
    for i, count in enumerate(span_counts):
        end = start + count
        keypoint_index[start : start + end, 1] = 8 + i
        start = end
    keypoint_index[1:, 0] = np.arange(npts) + 8 + nspans
    return keypoint_index


def project_keypoints(pvec: np.ndarray, keypoint_index: np.ndarray) -> np.ndarray:
    """Project parameter-vector keypoints into 2D coordinates.

    Given a parameter vector `pvec` and an indexing array `keypoint_index`,
    this function extracts the relevant keypoints, sets the first row to the origin,
    and then applies a cubic projection via `project_xy`.

    Args:
        pvec: A 2D parameter array or vector from which to extract keypoints.
        keypoint_index: The index array mapping each keypoint row to a location in `pvec`.

    Returns:
        A numpy array of projected XY coordinates, as returned by `project_xy`.

    """
    xy_coords = pvec[keypoint_index]
    xy_coords[0, :] = 0
    return project_xy(xy_coords, pvec)
