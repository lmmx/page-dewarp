"""Utilities for detecting, filtering, and analyzing contours within images.

This module provides functions to:

- Find external contours in a binary mask,
- Calculate centroid and orientation for each contour (via SVD on central moments),
- Filter contours by size and shape,
- Construct minimal masks,
- And visualize the resulting contours for debugging.

"""

from __future__ import annotations

import numpy as np
from cv2 import (
    CHAIN_APPROX_NONE,
    LINE_AA,
    RETR_EXTERNAL,
    SVDecomp,
    boundingRect,
    circle,
    drawContours,
    findContours,
    line,
)
from cv2 import moments as cv2_moments

from .debug_utils import cCOLOURS, debug_show
from .options import cfg
from .simple_utils import fltp
from .snoopy import snoop


__all__ = [
    "blob_mean_and_tangent",
    "interval_measure_overlap",
    "ContourInfo",
    "make_tight_mask",
    "get_contours",
    "visualize_contours",
]


@snoop()
def blob_mean_and_tangent(contour: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute the centroid and principal orientation of a contour.

    Constructs the blob image's covariance matrix from second-order central moments
    by dividing them by the 0th-order 'area moment' to make them translation-invariant.
    The eigenvectors of the covariance matrix provide the blob's principal axes,
    from which we extract a 2D orientation vector (`tangent`) and a centroid (`center`).

    Args:
        contour: A single contour (numpy array) representing the boundary of a shape.

    Returns:
        A tuple `(center, tangent)` where:

            - `center` is (x, y) for the contour's centroid,
            - `tangent` is the principal orientation as a unit vector.

        Returns `None` if the contour area is zero.

    """
    moments = cv2_moments(contour)
    area = moments["m00"]
    if area:
        mean_x = moments["m10"] / area
        mean_y = moments["m01"] / area
        covariance_matrix = np.divide(
            [[moments["mu20"], moments["mu11"]], [moments["mu11"], moments["mu02"]]],
            area,
        )
        _, svd_u, _ = SVDecomp(covariance_matrix)
        center = np.array([mean_x, mean_y])
        tangent = svd_u[:, 0].flatten().copy()
        if cfg.DEBUG_LEVEL > 2:
            print(f"Got contour with {center=} {tangent=}")
        return center, tangent
    else:
        # Sometimes `cv2.moments()` returns all-zero moments. Prevent ZeroDivisionError:
        if cfg.DEBUG_LEVEL > 0:
            print("Discarding contour with zero moments")
        return None


def interval_measure_overlap(
    int_a: tuple[float, float],
    int_b: tuple[float, float],
) -> float:
    """Return the overlap length of two 1D intervals.

    Each interval is given as (start, end). The overlap is computed as:
    min(a_end, b_end) - max(a_start, b_start).

    Args:
        int_a: The first interval, e.g. (a_start, a_end).
        int_b: The second interval, e.g. (b_start, b_end).

    Returns:
        The overlap length (which may be negative if no overlap exists).

    """
    return min(int_a[1], int_b[1]) - max(int_a[0], int_b[0])


class ContourInfo:
    """Holds geometric and orientation data about a single contour."""

    def __init__(
        self,
        contour: np.ndarray,
        moments: tuple[np.ndarray, np.ndarray],
        rect: tuple[int, int, int, int],
        mask: np.ndarray,
    ) -> None:
        """Initialize a contour's geometry, orientation, bounding rect, and mask.

        Args:
            contour: The raw points making up the contour (numpy array).
            moments: A tuple `(center, tangent)` from `blob_mean_and_tangent`.
            rect: A bounding rectangle `(xmin, ymin, width, height)`.
            mask: A binary mask of just this contour, cropped to `rect`.

        """
        self.contour = contour
        self.rect = rect
        self.mask = mask
        self.center, self.tangent = moments
        self.angle = np.arctan2(self.tangent[1], self.tangent[0])

        # Project each point onto the local tangent axis.
        clx = [self.proj_x(point) for point in self.contour]
        lxmin, lxmax = min(clx), max(clx)
        self.local_xrng = (lxmin, lxmax)
        self.point0 = self.center + self.tangent * lxmin
        self.point1 = self.center + self.tangent * lxmax
        self.pred = None
        self.succ = None

    def __repr__(self) -> str:
        """Return a string representation of the ContourInfo object."""
        return (
            f"ContourInfo: contour={self.contour}, rect={self.rect}, mask={self.mask}, "
            f"center={self.center}, tangent={self.tangent}, angle={self.angle}"
        )

    def proj_x(self, point: np.ndarray) -> float:
        """Compute the scalar projection of a point onto this contour's tangent axis.

        The tangent axis is defined by `self.center` and `self.tangent`.
        """
        return np.dot(self.tangent, point.flatten() - self.center)

    def local_overlap(self, other: ContourInfo) -> float:
        """Compute the overlap of this contour's local axis range with another contour's.

        Args:
            other: Another ContourInfo instance.

        Returns:
            The 1D overlap along the tangent axis, as computed by `interval_measure_overlap`.

        """
        xmin = self.proj_x(other.point0)
        xmax = self.proj_x(other.point1)
        return interval_measure_overlap(self.local_xrng, (xmin, xmax))


def make_tight_mask(
    contour: np.ndarray,
    xmin: int,
    ymin: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Create a minimal binary mask for a contour.

    The mask is cropped to the bounding rectangle `(xmin, ymin, width, height)`.
    The contour is shifted so it fits in the top-left corner of the mask.

    Args:
        contour: The raw contour points (numpy array).
        xmin: X coordinate of the bounding rect's top-left corner.
        ymin: Y coordinate of the bounding rect's top-left corner.
        width: Width of the bounding rectangle.
        height: Height of the bounding rectangle.

    Returns:
        A 2D uint8 mask with the same shape as the bounding rectangle,
        filled in for the region covered by `contour`.

    """
    import numpy as np

    tight_mask = np.zeros((height, width), dtype=np.uint8)
    tight_contour = contour - np.array((xmin, ymin)).reshape((-1, 1, 2))
    drawContours(tight_mask, [tight_contour], contourIdx=0, color=1, thickness=-1)
    return tight_mask


def get_contours(name: str, small: np.ndarray, mask: np.ndarray) -> list[ContourInfo]:
    """Detect and filter contours in a binary mask, returning their ContourInfo objects.

    This function finds external contours, filters them by size/aspect,
    computes the centroid/orientation, and wraps everything in `ContourInfo`.
    If DEBUG_LEVEL >= 2, it visualizes the resulting contours.

    Args:
        name: A string identifier for debugging/logging.
        small: A downsampled image (for visualization).
        mask: A 2D binary mask in which to find contours.

    Returns:
        A list of `ContourInfo` objects for those contours that pass size/aspect checks.

    """
    contours, _ = findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_NONE)
    contours_out = []
    for contour in contours:
        rect = boundingRect(contour)
        xmin, ymin, width, height = rect
        if (
            width < cfg.TEXT_MIN_WIDTH
            or height < cfg.TEXT_MIN_HEIGHT
            or width < cfg.TEXT_MIN_ASPECT * height
        ):
            continue
        tight_mask = make_tight_mask(contour, xmin, ymin, width, height)
        if tight_mask.sum(axis=0).max() > cfg.TEXT_MAX_THICKNESS:
            continue
        if (moments := blob_mean_and_tangent(contour)) is None:
            continue
        info = ContourInfo(contour=contour, moments=moments, rect=rect, mask=tight_mask)
        contours_out.append(info)
    if cfg.DEBUG_LEVEL >= 2:
        visualize_contours(name, small, contours_out)
    return contours_out


def visualize_contours(
    name: str,
    small: np.ndarray,
    cinfo_list: list[ContourInfo],
) -> None:
    """Overlay colored contours on a copy of the image for debugging or inspection.

    Each contour is filled with a unique color. The center and principal axis are
    drawn in white lines. A half-and-half blend of the filled contour and the original
    image is used for better visibility.

    Args:
        name: A string identifier for debugging/logging.
        small: The downsampled image in which to draw.
        cinfo_list: A list of `ContourInfo` objects (contours to visualize).

    """
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
