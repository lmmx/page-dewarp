The following takes place in `solve.py`

In the previous section, `xcoords` and `ycoords` are created from sampling spans.

These variables are then used immediately to create
`rough_dims` (an estimate of the page width and height from the norm
[distance] of the bottom right and top right corner from the bottom left corner).

---

So far the spans have just been sampled as "contour points" and "span points",
but the blog post refers to them as "keypoints".

To become keypoints, they get processed further, in the initialisation of `WarpedImage`:

```py
corners, ycoords, xcoords = keypoints_from_samples(
    self.stem, self.small, self.pagemask, self.page_outline, span_points
)
rough_dims, span_counts, params = get_default_params(
    corners, ycoords, xcoords
)
dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) + tuple(span_points))
```

---

Additionally, `params` is returned from the same function `get_default_params`
which is made up of

```py
params = np.hstack(
    (
	np.array(rvec).flatten(),
	np.array(tvec).flatten(),
	np.array(cubic_slopes).flatten(),
	ycoords.flatten(),
    )
    + tuple(xcoords)
)
```

where:

- `rvec` and `tvec` are an estimate of rotation and translation from four 2D-to-3D point
  correspondences
- `cubic_slopes` is just an initial guess of $[0, 0]$ (no slope)

It is here that the spline model is implemented: the 3D height is modelled as uniformly flat
(the corners are listed in 3D anti-clockwise from bottom left)

```py
corners_object3d = np.array(
    [
	[0, 0, 0],
	[page_width, 0, 0],
	[page_width, page_height, 0],
	[0, page_height, 0],
    ]
)
```

and then a [Perspective-n-Point](https://en.wikipedia.org/wiki/Perspective-n-Point)
problem is solved from these (n=4) 3D corner points, giving a translation and rotation vector

```py
_, rvec, tvec = solvePnP(corners_object3d, corners, K(), np.zeros(5))
```
