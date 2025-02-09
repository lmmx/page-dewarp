"""Defines the default intrinsic camera matrix (K) using a configurable focal length.

This module exports a single function, `K`, which returns a 3x3 NumPy array
representing the camera's intrinsic matrix based on `Config.FOCAL_LENGTH`.
"""

import numpy as np

from .core import Config


__all__ = ("K",)


def K(cfg: Config) -> np.ndarray:
    """Return the default intrinsic parameter matrix, derived from `cfg.FOCAL_LENGTH`.

    Args:
        cfg: The configuration object, which includes `FOCAL_LENGTH`.

    Returns:
        A 3x3 NumPy array representing the intrinsic camera matrix.

    """
    return np.array(
        [
            [cfg.FOCAL_LENGTH, 0, 0],
            [0, cfg.FOCAL_LENGTH, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
