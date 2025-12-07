"""Spanning and keypoint logic for page dewarping.

This module handles:

- Determining candidate edges between adjacent contours to form "spans."
- Grouping individual contours into spans.
- Sampling keypoint positions along spans.
- Computing overall orientation from those keypoints (to find x/y directions).
- Visualizing spans and their keypoints.
"""

import numpy as np
from cv2 import LINE_AA, PCACompute, circle, convexHull, drawContours, line, polylines

from .contours import ContourInfo
from .debug_utils import cCOLOURS, debug_show
from .normalisation import norm2pix, pix2norm
from .options import cfg
from .simple_utils import fltp


__all__ = [
    "angle_dist",
    "generate_candidate_edge",
    "assemble_spans",
    "sample_spans",
    "keypoints_from_samples",
    "visualize_spans",
    "visualize_span_points",
]


def angle_dist(angle_b: float, angle_a: float) -> float:
    """Compute the signed angular distance between two angles.

    The distance is corrected to lie within [-π, π].
    """
    diff = angle_b - angle_a
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return np.abs(diff)


def generate_candidate_edge(
    cinfo_a: ContourInfo,
    cinfo_b: ContourInfo,
) -> tuple[float, ContourInfo, ContourInfo] | None:
    """Compute a left-to-right candidate edge between two contours.

    We want `cinfo_a` (left) to come before `cinfo_b` (right), so `cinfo_a`’s successor
    is `cinfo_b` and `cinfo_b`’s predecessor is `cinfo_a`. Specifically, the right
    endpoint of `cinfo_b` should be to the right of the left endpoint of `cinfo_a`.  If
    not, we swap them. Then we compute how far apart they are, how much they overlap,
    and how different their orientations are. If these checks pass certain thresholds,
    we return `(score, cinfo_a, cinfo_b)`, otherwise `None`.
    """
    if cinfo_a.point0[0] > cinfo_b.point1[0]:
        cinfo_a, cinfo_b = cinfo_b, cinfo_a
    x_overlap_a = cinfo_a.local_overlap(cinfo_b)
    x_overlap_b = cinfo_b.local_overlap(cinfo_a)
    overall_tangent = cinfo_b.center - cinfo_a.center
    overall_angle = np.arctan2(overall_tangent[1], overall_tangent[0])
    delta_angle = np.divide(
        max(
            angle_dist(cinfo_a.angle, overall_angle),
            angle_dist(cinfo_b.angle, overall_angle),
        )
        * 180,
        np.pi,
    )
    # We want the largest overlap in x to be small
    x_overlap = max(x_overlap_a, x_overlap_b)
    dist = np.linalg.norm(cinfo_b.point0 - cinfo_a.point1)
    if not (
        dist > cfg.EDGE_MAX_LENGTH
        or x_overlap > cfg.EDGE_MAX_OVERLAP
        or delta_angle > cfg.EDGE_MAX_ANGLE
    ):
        score = dist + delta_angle * cfg.EDGE_ANGLE_COST
        return (score, cinfo_a, cinfo_b)
    return None


def assemble_spans(
    name: str,
    small: np.ndarray,
    pagemask: np.ndarray,
    cinfo_list: list[ContourInfo],
) -> list[list[ContourInfo]]:
    """Assemble spans of contours from a list of ContourInfo objects.

    A 'span' is a left-to-right chain of contours. We generate candidate edges,
    sort them by a "score," and build spans by linking successors/predecessors.
    Spans are retained only if their total width exceeds `SPAN_MIN_WIDTH`.

    If DEBUG_LEVEL >= 2, the resulting spans are visualized.

    Args:
        name: A string identifier used for debug display.
        small: A downsampled image (for visualization).
        pagemask: A mask for the page region.
        cinfo_list: A list of ContourInfo objects to link into spans.

    Returns:
        A list of spans, where each span is a list of ContourInfo objects.

    """
    cinfo_list = sorted(cinfo_list, key=lambda cinfo: cinfo.rect[1])
    candidate_edges = []
    for i, cinfo_i in enumerate(cinfo_list):
        for j in range(i):
            # note e is of the form (score, left_cinfo, right_cinfo)
            edge = generate_candidate_edge(cinfo_i, cinfo_list[j])
            if edge is not None:
                candidate_edges.append(edge)

    # Sort candidate edges by score (lower is better)
    candidate_edges.sort(key=lambda e: e[0])

    # TODO: implement comparison operators on ContourInfo class to permit tie breaking,
    # see: https://github.com/lmmx/page-dewarp/pull/6/

    # Link contours using successor/predecessor
    for _, cinfo_a, cinfo_b in candidate_edges:
        # If both left and right are unassigned, join them
        if cinfo_a.succ is None and cinfo_b.pred is None:
            cinfo_a.succ = cinfo_b
            cinfo_b.pred = cinfo_a

    # Build spans by walking from each "head" to the end
    spans = []
    while cinfo_list:
        # get the first on the list
        cinfo = cinfo_list[0]
        # keep following predecessors until none exists
        while cinfo.pred:
            cinfo = cinfo.pred

        # start a new span
        cur_span = []
        width = 0.0
        # follow successors til end of span
        while cinfo:
            # remove from list (sadly making this loop *also* O(n^2)
            cinfo_list.remove(cinfo)
            # add to span
            cur_span.append(cinfo)
            width += cinfo.local_xrng[1] - cinfo.local_xrng[0]
            # set successor
            cinfo = cinfo.succ

        # add if long enough
        if width > cfg.SPAN_MIN_WIDTH:
            spans.append(cur_span)

    if cfg.DEBUG_LEVEL >= 2:
        visualize_spans(name, small, pagemask, spans)

    return spans


def sample_spans(
    shape: tuple[int, int],
    spans: list[list[ContourInfo]],
) -> list[np.ndarray]:
    """Extract regularly spaced keypoints from each span.

    Within each contour's bounding rectangle, we measure the vertical average
    of pixels in the contour's mask at certain horizontal steps. We then convert
    all points to normalized coordinates.

    Args:
        shape: The (height, width) of the downsampled image.
        spans: A list of spans, where each span is a list of ContourInfo objects.

    Returns:
        A list of arrays, each array containing the sampled points (in normalized coords).

    """
    span_points = []
    for span in spans:
        contour_points = []
        for cinfo in span:
            yvals = np.arange(cinfo.mask.shape[0]).reshape((-1, 1))
            totals = (yvals * cinfo.mask).sum(axis=0)
            means = np.divide(totals, cinfo.mask.sum(axis=0))
            xmin, ymin = cinfo.rect[:2]
            step = cfg.SPAN_PX_PER_STEP
            start = np.floor_divide((np.mod((len(means) - 1), step)), 2)
            contour_points.extend(
                [(x + xmin, means[x] + ymin) for x in range(start, len(means), step)],
            )
        contour_points = np.array(contour_points, dtype=np.float32).reshape((-1, 1, 2))
        contour_points = pix2norm(shape, contour_points)
        span_points.append(contour_points)
    return span_points


def keypoints_from_samples(
    name: str,
    small: np.ndarray,
    pagemask: np.ndarray,
    page_outline: np.ndarray,
    span_points: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Compute page corner keypoints and local x/y directions from span samples.

    Performs a PCA on the combined sample points to estimate a horizontal axis (x_dir).
    The vertical axis (y_dir) is orthogonal to x_dir. Then computes the page's four
    corners by projecting the page outline onto x_dir and y_dir, and forms a convex hull.

    Args:
        name: A string identifier for debugging.
        small: Downsampled image (for optional visualization).
        pagemask: Mask of the page region (for dimension references).
        page_outline: The polygon outlining the page boundary (numpy array).
        span_points: List of arrays of normalized keypoints from each span.

    Returns:
        A tuple `(corners, ycoords, xcoords)` where:
            - `corners` is an (4,1,2) array of the page corners in normalized coords.
            - `ycoords` is a 1D array of the average y-locations of each span.
            - `xcoords` is a list of x-locations of each span's sample points.

    """
    all_evecs = np.array([[0.0, 0.0]])
    all_weights = 0
    for points in span_points:
        _, evec = PCACompute(points.reshape((-1, 2)), mean=None, maxComponents=1)
        weight = np.linalg.norm(points[-1] - points[0])
        all_evecs += evec * weight
        all_weights += weight
    evec = all_evecs / all_weights
    x_dir = evec.flatten()
    if x_dir[0] < 0:
        x_dir = -x_dir
    y_dir = np.array([-x_dir[1], x_dir[0]])

    pagecoords = convexHull(page_outline)
    pagecoords = pix2norm(pagemask.shape, pagecoords.reshape((-1, 1, 2))).reshape(
        (-1, 2),
    )

    px_coords = np.dot(pagecoords, x_dir)
    py_coords = np.dot(pagecoords, y_dir)
    px0, px1 = px_coords.min(), px_coords.max()
    py0, py1 = py_coords.min(), py_coords.max()
    # [px0,px1,px1,px0] for first bit of p00,p10,p11,p01
    x_dir_coeffs = np.pad([px0, px1], 2, mode="symmetric")[2:].reshape(-1, 1)
    # [py0,py0,py1,py1] for second bit of p00,p10,p11,p01
    y_dir_coeffs = np.repeat([py0, py1], 2).reshape(-1, 1)
    corners = np.expand_dims((x_dir_coeffs * x_dir) + (y_dir_coeffs * y_dir), 1)

    xcoords, ycoords = [], []
    for points in span_points:
        pts = points.reshape((-1, 2))
        px_coords, py_coords = np.dot(pts, np.transpose([x_dir, y_dir])).T
        xcoords.append(px_coords - px0)
        ycoords.append(py_coords.mean() - py0)
    if cfg.DEBUG_LEVEL >= 2:
        visualize_span_points(name, small, span_points, corners)
    return corners, np.array(ycoords), xcoords


def visualize_spans(
    name: str,
    small: np.ndarray,
    pagemask: np.ndarray,
    spans: list[list[ContourInfo]],
) -> None:
    """Render spans as colored regions for debugging.

    Each span is drawn as a set of filled contours. The final image dims outside
    the page mask.
    """
    regions = np.zeros_like(small)
    for i, span in enumerate(spans):
        contours = [cinfo.contour for cinfo in span]
        drawContours(regions, contours, -1, cCOLOURS[i * 3 % len(cCOLOURS)], -1)
    mask = regions.max(axis=2) != 0
    display = small.copy()
    display[mask] = (display[mask] / 2) + (regions[mask] / 2)
    display[pagemask == 0] //= 4
    debug_show(name, 2, "spans", display)


def visualize_span_points(
    name: str,
    small: np.ndarray,
    span_points: list[np.ndarray],
    corners: np.ndarray,
) -> None:
    """Draw keypoints from the spans and highlight the page corners.

    Performs a quick PCA per span to show each span's approximate orientation axis.
    The final page corners are polylined in white.
    """
    display = small.copy()
    for i, points in enumerate(span_points):
        points = norm2pix(small.shape, points, False)
        mean, small_evec = PCACompute(points.reshape((-1, 2)), None, maxComponents=1)
        dps = np.dot(points.reshape((-1, 2)), small_evec.reshape((2, 1)))
        dpm = np.dot(mean.flatten(), small_evec.flatten())
        point0 = mean + small_evec * (dps.min() - dpm)
        point1 = mean + small_evec * (dps.max() - dpm)
        for point in points:
            circle(display, fltp(point), 3, cCOLOURS[i % len(cCOLOURS)], -1, LINE_AA)
        line(display, fltp(point0), fltp(point1), (255, 255, 255), 1, LINE_AA)
    polylines(display, [norm2pix(small.shape, corners, True)], True, (255, 255, 255))
    debug_show(name, 3, "span points", display)
