from datetime import datetime as dt

import numpy as np
from cv2 import LINE_AA, circle, line
from scipy.optimize import minimize

from .debug_utils import debug_show
from .keypoints import make_keypoint_index, project_keypoints
from .normalisation import norm2pix
from .simple_utils import fltp

__all__ = ["optimise_params"]


def draw_correspondences(img, dstpoints, projpts):
    display = img.copy()
    dstpoints = norm2pix(img.shape, dstpoints, True)
    projpts = norm2pix(img.shape, projpts, True)
    for pts, color in [(projpts, (255, 0, 0)), (dstpoints, (0, 0, 255))]:
        for point in pts:
            circle(display, fltp(point), 3, color, -1, LINE_AA)
    for point_a, point_b in zip(projpts, dstpoints):
        line(display, fltp(point_a), fltp(point_b), (255, 255, 255), 1, LINE_AA)
    return display


def optimise_params(name, small, dstpoints, span_counts, params, debug_lvl):
    keypoint_index = make_keypoint_index(span_counts)

    def objective(pvec):
        ppts = project_keypoints(pvec, keypoint_index)
        return np.sum((dstpoints - ppts) ** 2)

    print("  initial objective is", objective(params))
    if debug_lvl >= 1:
        projpts = project_keypoints(params, keypoint_index)
        display = draw_correspondences(small, dstpoints, projpts)
        debug_show(name, 4, "keypoints before", display)
    print("  optimizing", len(params), "parameters...")
    start = dt.now()
    res = minimize(objective, params, method="Powell")
    end = dt.now()
    print(f"  optimization took {round((end - start).total_seconds(), 2)} sec.")
    print(f"  final objective is {res.fun}")
    params = res.x
    if debug_lvl >= 1:
        projpts = project_keypoints(params, keypoint_index)
        display = draw_correspondences(small, dstpoints, projpts)
        debug_show(name, 5, "keypoints after", display)
    return params
