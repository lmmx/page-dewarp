import numpy as np
from cv2 import solvePnP

from .options import K, cfg

__all__ = ["get_default_params"]

# TODO refactor this to be a class for both the flattening and parsing?
def get_default_params(corners, ycoords, xcoords):
    page_width, page_height = [np.linalg.norm(corners[i] - corners[0]) for i in (1, -1)]
    cubic_slopes = [0.0, 0.0]  # initial guess for the cubic has no slope
    # object points of flat page in 3D coordinates
    corners_object3d = np.array(
        [
            [0, 0, 0],
            [page_width, 0, 0],
            [page_width, page_height, 0],
            [0, page_height, 0],
        ]
    )
    # estimate rotation and translation from four 2D-to-3D point correspondences
    _, rvec, tvec = solvePnP(corners_object3d, corners, K(), np.zeros(5))
    span_counts = [*map(len, xcoords)]
    params = np.hstack(
        (
            np.array(rvec).flatten(),
            np.array(tvec).flatten(),
            np.array(cubic_slopes).flatten(),
            ycoords.flatten(),
        )
        + tuple(xcoords)
    )
    return (page_width, page_height), span_counts, params
