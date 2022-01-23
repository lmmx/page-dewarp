import numpy as np
from cv2 import (
    circle,
    line,
    LINE_AA,
    moments as cv2_moments,
    SVDecomp,
    drawContours,
    boundingRect,
    findContours,
    CHAIN_APPROX_NONE,
    RETR_EXTERNAL,
)
from .debug_utils import cCOLOURS, debug_show
from .options import cfg
from .simple_utils import fltp

__all__ = [
    "blob_mean_and_tangent",
    "interval_measure_overlap",
    "ContourInfo",
    "make_tight_mask",
    "get_contours",
    "visualize_contours",
]


def blob_mean_and_tangent(contour):
    """
    Construct blob image's covariance matrix from second order central moments
    (i.e. dividing them by the 0-order 'area moment' to make them translationally
    invariant), from the eigenvectors of which the blob orientation can be
    extracted (they are its principle components).
    """
    moments = cv2_moments(contour)
    area = moments["m00"]
    mean_x = moments["m10"] / area
    mean_y = moments["m01"] / area
    covariance_matrix = np.divide(
        [[moments["mu20"], moments["mu11"]], [moments["mu11"], moments["mu02"]]], area
    )
    _, svd_u, _ = SVDecomp(covariance_matrix)
    center = np.array([mean_x, mean_y])
    tangent = svd_u[:, 0].flatten().copy()
    return center, tangent


def interval_measure_overlap(int_a, int_b):
    return min(int_a[1], int_b[1]) - max(int_a[0], int_b[0])


class ContourInfo:
    def __init__(self, contour, rect, mask):
        self.contour = contour
        self.rect = rect
        self.mask = mask
        self.center, self.tangent = blob_mean_and_tangent(contour)
        self.angle = np.arctan2(self.tangent[1], self.tangent[0])
        clx = [self.proj_x(point) for point in self.contour]
        lxmin, lxmax = min(clx), max(clx)
        self.local_xrng = (lxmin, lxmax)
        self.point0 = self.center + self.tangent * lxmin
        self.point1 = self.center + self.tangent * lxmax
        self.pred = None
        self.succ = None

    def proj_x(self, point):
        return np.dot(self.tangent, point.flatten() - self.center)

    def local_overlap(self, other):
        xmin = self.proj_x(other.point0)
        xmax = self.proj_x(other.point1)
        return interval_measure_overlap(self.local_xrng, (xmin, xmax))


def make_tight_mask(contour, xmin, ymin, width, height):
    tight_mask = np.zeros((height, width), dtype=np.uint8)
    tight_contour = contour - np.array((xmin, ymin)).reshape((-1, 1, 2))
    drawContours(tight_mask, [tight_contour], contourIdx=0, color=1, thickness=-1)
    return tight_mask


def get_contours(name, small, mask):
    contours, _ = findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_NONE)
    contours_out = []
    for contour in contours:
        rect = boundingRect(contour)
        xmin, ymin, width, height = rect
        if (
            width < cfg.contour_opts.TEXT_MIN_WIDTH
            or height < cfg.contour_opts.TEXT_MIN_HEIGHT
            or width < cfg.contour_opts.TEXT_MIN_ASPECT * height
        ):
            continue
        tight_mask = make_tight_mask(contour, xmin, ymin, width, height)
        if tight_mask.sum(axis=0).max() > cfg.contour_opts.TEXT_MAX_THICKNESS:
            continue
        contours_out.append(ContourInfo(contour, rect, tight_mask))
    if cfg.debug_lvl_opt.DEBUG_LEVEL >= 2:
        visualize_contours(name, small, contours_out)
    return contours_out


def visualize_contours(name, small, cinfo_list):
    regions = np.zeros_like(small)
    for j, cinfo in enumerate(cinfo_list):
        drawContours(regions, [cinfo.contour], 0, cCOLOURS[j % len(cCOLOURS)], -1)
    mask = regions.max(axis=2) != 0
    display = small.copy()
    display[mask] = (display[mask] / 2) + (regions[mask] / 2)
    for j, cinfo in enumerate(cinfo_list):
        color = cCOLOURS[j % len(cCOLOURS)]
        color = tuple(c // 4 for c in color)
        circle(display, fltp(cinfo.center), 3, (255, 255, 255), 1, LINE_AA)
        line(
            display,
            fltp(cinfo.point0),
            fltp(cinfo.point1),
            (255, 255, 255),
            1,
            LINE_AA,
        )
    debug_show(name, 1, "contours", display)
