from cv2 import PCACompute, convexHull, drawContours, circle, line, polylines, LINE_AA
import numpy as np
from .debug_utils import cCOLOURS, debug_show
from .options import cfg
from .normalisation import norm2pix, pix2norm
from .simple_utils import fltp

__all__ = ["assemble_spans", "sample_spans", "keypoints_from_samples"]


def angle_dist(angle_b, angle_a):
    diff = angle_b - angle_a
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return np.abs(diff)


def generate_candidate_edge(cinfo_a, cinfo_b):
    """
    We want a left of b (so a's successor will be b and b's
    predecessor will be a). Make sure right endpoint of b is to the
    right of left endpoint of a (swap them if not the case).
    """
    if cinfo_a.point0[0] > cinfo_b.point1[0]:
        tmp = cinfo_a
        cinfo_a = cinfo_b
        cinfo_b = tmp
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
    # we want the largest overlap in x to be small
    x_overlap = max(x_overlap_a, x_overlap_b)
    dist = np.linalg.norm(cinfo_b.point0 - cinfo_a.point1)
    if not (
        dist > cfg.edge_opts.EDGE_MAX_LENGTH
        or x_overlap > cfg.edge_opts.EDGE_MAX_OVERLAP
        or delta_angle > cfg.edge_opts.EDGE_MAX_ANGLE
    ):
        score = dist + delta_angle * cfg.edge_opts.EDGE_ANGLE_COST
        return (score, cinfo_a, cinfo_b)
    # else return None


def assemble_spans(name, small, pagemask, cinfo_list):
    cinfo_list = sorted(cinfo_list, key=lambda cinfo: cinfo.rect[1])
    candidate_edges = []
    for i, cinfo_i in enumerate(cinfo_list):
        for j in range(i):
            # note e is of the form (score, left_cinfo, right_cinfo)
            edge = generate_candidate_edge(cinfo_i, cinfo_list[j])
            if edge is not None:
                candidate_edges.append(edge)
    candidate_edges.sort()  # sort candidate edges by score (lower is better)
    for _, cinfo_a, cinfo_b in candidate_edges:  # for each candidate edge
        # if left and right are unassigned, join them
        if cinfo_a.succ is None and cinfo_b.pred is None:
            cinfo_a.succ = cinfo_b
            cinfo_b.pred = cinfo_a
    spans = []
    while cinfo_list:
        cinfo = cinfo_list[0]  # get the first on the list
        # keep following predecessors until none exists
        while cinfo.pred:
            cinfo = cinfo.pred
        cur_span = []  # start a new span
        width = 0.0
        while cinfo:  # follow successors til end of span
            # remove from list (sadly making this loop *also* O(n^2)
            cinfo_list.remove(cinfo)
            cur_span.append(cinfo)  # add to span
            width += cinfo.local_xrng[1] - cinfo.local_xrng[0]
            cinfo = cinfo.succ  # set successor
        if width > cfg.span_opts.SPAN_MIN_WIDTH:
            spans.append(cur_span)  # add if long enough
    if cfg.debug_lvl_opt.DEBUG_LEVEL >= 2:
        visualize_spans(name, small, pagemask, spans)
    return spans


def sample_spans(shape, spans):
    span_points = []
    for span in spans:
        contour_points = []
        for cinfo in span:
            yvals = np.arange(cinfo.mask.shape[0]).reshape((-1, 1))
            totals = (yvals * cinfo.mask).sum(axis=0)
            means = np.divide(totals, cinfo.mask.sum(axis=0))
            xmin, ymin = cinfo.rect[:2]
            step = cfg.span_opts.SPAN_PX_PER_STEP
            start = np.floor_divide((np.mod((len(means) - 1), step)), 2)
            contour_points.extend(
                [(x + xmin, means[x] + ymin) for x in range(start, len(means), step)]
            )
        contour_points = np.array(contour_points, dtype=np.float32).reshape((-1, 1, 2))
        contour_points = pix2norm(shape, contour_points)
        span_points.append(contour_points)
    return span_points


def keypoints_from_samples(name, small, pagemask, page_outline, span_points):
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
        (-1, 2)
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
    if cfg.debug_lvl_opt.DEBUG_LEVEL >= 2:
        visualize_span_points(name, small, span_points, corners)
    return corners, np.array(ycoords), xcoords


def visualize_spans(name, small, pagemask, spans):
    regions = np.zeros_like(small)
    for i, span in enumerate(spans):
        contours = [cinfo.contour for cinfo in span]
        drawContours(regions, contours, -1, cCOLOURS[i * 3 % len(cCOLOURS)], -1)
    mask = regions.max(axis=2) != 0
    display = small.copy()
    display[mask] = (display[mask] / 2) + (regions[mask] / 2)
    display[pagemask == 0] //= 4
    debug_show(name, 2, "spans", display)


def visualize_span_points(name, small, span_points, corners):
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
