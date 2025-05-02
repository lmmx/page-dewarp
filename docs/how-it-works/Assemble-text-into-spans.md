After text contours were detected, spans are assembled.

- Note that the assembly of spans was introduced in the previous section

By this, we mean that the individual text contours (e.g. words)
are assembled into a "span" with a common vertical offset on the page
(in other words, a line or a partial line).

---

To recap, the candidate edges were generated as follows:

```py
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
```

- `cinfo_a.point0[0]` is the $x$ coord of the leftmost point of the contour `cinfo_a`
- `cinfo_b.point1[0]` is the $x$ coord of the rightmost point of the contour `cinfo_b`
- The first manoeuvre involving `tmp` simply swaps `cinfo_a` and `cinfo_b` so that
  `cinfo_a` is on the left of `cinfo_b`

The `local_overlap` method takes the inner (dot) product of a point relative to the
centre-point of the blob it's called from (giving a number).

The `overall_tangent` is the relative position of the righter-most blob (`cinfo_b`)
from the lefter-most blob (`cinfo_a`), then $atan2$ gives the corresponding `overall_angle`.

The blobs themselves have an `angle` attribute, calculated again with $atan2$, but
from the tangent of the contour [around the text in the blob].

The `delta_angle` indicates the difference in the angle of the blob (either blob, whichever
has the biggest delta) in degrees between the contour tangent angle and the inter-blob
angle. This is a way of measuring how aligned the two blobs are (so collinear text spans
will align and have a small `delta_angle` while comparing blobs on separate lines
will have a much higher angle difference between the blob tangent angle and inter-blob angle).

- Note that the angle is multiplied by $180$ and divided by $\pi$, which is equivalent to
  multiplying by $360/2\pi$, i.e. the `delta_angle` becomes stated in degrees whereas
  all the other angles are in radians.
  - This is so the default for `EDGE_MAX_ANGLE` can be in 'human-readable' degrees.

The exception to this would be if the two blobs were on different lines but far apart horizontally,
which would 'smooth out' the angle difference. To account for this, the score that is
calculated for the pair of contours [a candidate edge] also takes into account the distance between
the left contour's rightmost point and the right contour's leftmost point.

Next there comes a filtering step to skip any potential edges whose:

- distance between these points is greater than `EDGE_MAX_LENGTH` (default: 100px), or
- overlap between the contours is greater than `EDGE_MAX_OVERLAP` (default: 1px), or
- `delta_angle` is greater than `EDGE_MAX_ANGLE` (default: $7.5\degree$)

If none of these conditions are met, the edge is not skipped, and
the distance and the scaled `delta_angle` are simply added up
(after the `delta_angle` has been scaled by a `EDGE_ANGLE_COST`).

---

This is all in preparation to sort the candidate edges by their score,
then stepping through them to assign preceding and successive contours
stored on the `ContourInfo` objects themselves within the `candidate_edges`
list, then once this is done building these into spans given sufficient width.

```py
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
```

Note that in this step, some narrow contours (comprising short individual words
such as "and") may not make it through to form spans, instead becoming
a 'gap' in a line. This is due to the `SPAN_MIN_WIDTH` setting, which excludes
them.
