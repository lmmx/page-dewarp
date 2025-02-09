"""Pixel-to-normalized (and vice versa) coordinate transformations.

This module provides:
- `pix2norm`: Convert pixel coordinates in an image to normalized coordinates.
- `norm2pix`: Convert normalized coordinates back into pixel coordinates.
"""

import numpy as np


__all__ = ["pix2norm", "norm2pix"]


def pix2norm(shape: tuple[int, int], pts: np.ndarray) -> np.ndarray:
    """Convert image-space coordinates to normalized coordinates.

    Args:
        shape: A tuple (height, width) for the original image shape.
        pts: A NumPy array of shape (..., 1, 2) holding (x, y) coordinates.

    Returns:
        A NumPy array of the same shape as `pts`, with coordinates in a
        roughly [-1, +1] range, scaled and offset based on `shape`.

    """
    height, width = shape[:2]
    scl = 2.0 / max(height, width)
    offset = np.array([width, height], dtype=pts.dtype).reshape((-1, 1, 2)) * 0.5
    return (pts - offset) * scl


def norm2pix(shape: tuple[int, int], pts: np.ndarray, as_integer: bool) -> np.ndarray:
    """Convert normalized coordinates to image-space pixel coordinates.

    Args:
        shape: A tuple (height, width) for the target image shape.
        pts: A NumPy array of shape (..., 1, 2) containing normalized (x, y) coordinates.
        as_integer: If True, return integer pixel coordinates (rounded).
            Otherwise, return floating-point pixel coordinates.

    Returns:
        A NumPy array of the same shape as `pts`, representing pixel coordinates
        in the specified image dimensions.

    """
    height, width = shape[:2]
    scl = max(height, width) * 0.5
    offset = np.array([0.5 * width, 0.5 * height], dtype=pts.dtype).reshape((-1, 1, 2))
    rval = pts * scl + offset
    return (rval + 0.5).astype(int) if as_integer else rval
