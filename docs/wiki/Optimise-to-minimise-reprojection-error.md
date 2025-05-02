After creating the parameter estimate, the parameters are optimised by
nonlinear least squares.

```py
dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) + tuple(span_points))
params = optimise_params(
    self.stem,
    self.small,
    dstpoints,
    span_counts,
    params,
    cfg.debug_lvl_opt.DEBUG_LEVEL,
)
page_dims = get_page_dims(corners, rough_dims, params)
self.threshold(page_dims, params)
self.written = True
```

The `optimise_params` function from `optimise.py` is:

```py
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
```

using `make_keypoint_index` and `project_keypoints` from `keypoints.py`

```py
def make_keypoint_index(span_counts):
    nspans, npts = len(span_counts), sum(span_counts)
    keypoint_index = np.zeros((npts + 1, 2), dtype=int)
    start = 1
    for i, count in enumerate(span_counts):
        end = start + count
        keypoint_index[start : start + end, 1] = 8 + i
        start = end
    keypoint_index[1:, 0] = np.arange(npts) + 8 + nspans
    return keypoint_index
```

I can't work out why the number 8 is used to increment the
span enumeration index `i` nor to increment the range of `npts`
along with `nspans`...

```py
def project_keypoints(pvec, keypoint_index):
    xy_coords = pvec[keypoint_index]
    xy_coords[0, :] = 0
    return project_xy(xy_coords, pvec)
```

Here `pvec` is the vector of parameters `hstack`ed in the previous
step from the rotation and translation vectors along with the slope
estimation, and x- and y coords of the keypoints.

The `threshold` method sets the `outfile` attribute by
constructing a `RemappedImage` class.

```py
def threshold(self, page_dims, params):
    remap = RemappedImage(self.stem, self.cv2_img, self.small, page_dims, params)
    self.outfile = remap.threshfile
```

This ends the initialisation of the `WarpedImage` class,
so the `written` attribute is toggled to True.
