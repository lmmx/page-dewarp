import numpy as np

from .core import Config

__all__ = ("K",)


def K(cfg: Config):
    "Default intrinsic parameter matrix, depends on FOCAL_LENGTH"
    return np.array(
        [
            [cfg.FOCAL_LENGTH, 0, 0],
            [0, cfg.FOCAL_LENGTH, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
