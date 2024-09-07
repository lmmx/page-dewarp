import numpy as np

from .core import Config

__all__ = ("K",)


def K(cfg: Config):
    "Default intrinsic parameter matrix, depends on FOCAL_LENGTH"
    return np.array(
        [
            [cfg.camera_opts.FOCAL_LENGTH, 0, 0],
            [0, cfg.camera_opts.FOCAL_LENGTH, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
