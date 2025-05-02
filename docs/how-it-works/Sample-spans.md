Once the spans are assembled, they are sampled to give discrete keypoints
(representative points on each span), by default at one per 20 pixels of each
text contour (`SPAN_PX_PER_STEP`) in `spans.py`:

```py
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
            contour_points.extend([
                (x + xmin, means[x] + ymin)
                for x in range(start, len(means), step)
            ])  
        contour_points = np.array(contour_points, dtype=np.float32).reshape((-1, 1, 2))
        contour_points = pix2norm(shape, contour_points)
        span_points.append(contour_points)
    return span_points
```

This is called in the `image.py` module upon initialising `WarpedImage`,

```py
spans = self.iteratively_assemble_spans()
# Skip if no spans
if len(spans) < 1:
    print(f"skipping {self.stem} because only {len(spans)} spans")
else:
    span_points = sample_spans(self.small.shape, spans)
    n_pts = sum(map(len, span_points))
    print(f"  got {len(spans)} spans with {n_pts} points.")
```

The input is the shape of the shrunk image and the list of spans returned from
the assembly procedure as described in the previous section(s).

Note that the spans are really just `ContourInfo` classes, since the
span assembly procedure went like so:

```py
while cinfo_list:
    cinfo = cinfo_list[0]  # get the first on the list
    ...
    cur_span = []  # start a new span
    while cinfo:  # follow successors til end of span
        # remove from list (sadly making this loop *also* O(n^2)
        cinfo_list.remove(cinfo)
        cur_span.append(cinfo)  # add to span
```

> Recall: `cinfo_list` came from `WarpedImage.contour_list` which was returned
> from `WarpedImage.contour_info(text=True)` which wrapped a call to
> `Mask` and immediately returned the `Mask.contours()` method result,
> which called `get_contours` from `contours.py`... So from one end
> to the other, the `cinfo_list` is a list of the `ContourInfo` objects,
> and that's what's going in the "span": another list of `ContourInfo` objects.

Since the span is made up of `ContourInfo` objects, sampling a span
involves the mask [from step b) of the contour detection routine after dilation/erosion]

```py
yvals = np.arange(cinfo.mask.shape[0]).reshape((-1, 1))
totals = (yvals * cinfo.mask).sum(axis=0)
means = np.divide(totals, cinfo.mask.sum(axis=0))
```

`cinfo.mask.shape` is (height, width), which is also given from
`cinfo.rect` (from `cv2.boundingRect(contour)`), so the first line
creates a numpy range from 0 to the height [in pixels] of the contour
[within the current span] and then reshapes that range array
in what PyTorch calls "unsqueezing" or equivalently to calling
`np.expand_dims` on the range array with `axis=1`.

- In other words, each entry in the array becomes a singleton sub-array,
  `[1, 2, 3, ...]` becomes `[[1], [2], [3], ...]`

When this is multiplied by the mask itself, note that it's only composable
if it's a row vector (not a column vector), resulting in a row vector `totals`
with the same number of columns as the mask.

The multiplication of `yvals` (the range array indicating a row index) with
the binary `cinfo.mask` gives a mask whose entries are scaled to be the row
index, which has a side effect of turning any 1s in the top row into 0s
(and thus indistinguishable from the original 0s in the mask), but this has no
effect on the mean (as the denominator comes from the mask itself which preserves
the 1s in this top row).

Once this row vector is formed, it is divided by the column-wise sum of the
mask, i.e. it is scaled down as the mean row index of the rows of that column
that are active (i.e. 1) in the mask.

The sampling begins at a `start` point:

```py
start = np.floor_divide((np.mod((len(means) - 1), step)), 2)
```

This takes the floor of the remainder of one less than the length of the
column-wise `means` [i.e. one less than the width of the contour mask]
modulo the span sampling step size [default 20 pixels] divided by 2.

- Note that the sampling is not "every 20px along" but only where there is a
  contour. So if the line (i.e. the span) has some text on the left and some in
  the middle, the gap between the left and middle (gap with no contour) will
  _not_ be sampled, no points will 'cover' the gap.

To take the first contour in the _Boston Cooking_ example image,
`len(means) - 1` is 23 (because `cinfo.mask.shape` is (9,24) i.e.
the mask has 24 columns), and the remainder of 23 mod 20 is 3,
so the floor of 3 divided by 2 is 1, which is the value of `start`.
This means that the x coordinate is +1 pixel relative to the `xmin`
(which comes from the contour's bounding box).

- The significance of the floor division by 2 is roughly to place the
  sampling points equidistant from either end of the contour, with the
  first and last points with about the same number of pixels' gap.

After this the individual tuples of the points get unsqueezed again,
from `[[1,2], [3,4], [5,6]]` to `[[[1,2]], [[3,4]], [[5,6]]]` i.e.
each point becomes the singleton entry in the only column of the row,
i.e. the array is created with the points as rows and then unsqueezed
(or reshaped) so that the row entries become nested as a single entry.

The next thing that happens is the points are normalised by the
`pix2norm` function:

```py
def pix2norm(shape, pts):
    height, width = shape[:2]
    scl = 2.0 / (max(height, width))
    offset = np.array([width, height], dtype=pts.dtype).reshape((-1, 1, 2)) * 0.5
    return (pts - offset) * scl
```

The `shape` here was `WarpedImage.small.shape` in `image.py` (upon initialisation,
as mentioned above), and the general idea of `pix2norm` is to scale the absolute
pixel values through a homothetic transformation to leave it in relative coordinates
in the range [-1,+1].

The first step is to multiply by 2 and divide by the maximum of the height or width.
The second step is to calculate an offset: the width and height as an array,
again unsqueezing it so that it has a single row of a single column with the height and width,
then halving it.

Since it was unsqueezed, when subtracted from the points, the offset is applied
along the final axis (point-wise), so half the image dimensions is subtracted
from the coordinates, making the coordinates relative to the centre of the image.

Upon returning, these relative coordinates are scaled by `scl`, becoming
expressed in relative terms (in the range `[-1,1]`) from the centre of the image
rather than absolute.

---

A keypoint is
[a part of an image extracted for comparison against another image](https://dsp.stackexchange.com/questions/10423/why-do-we-use-keypoint-descriptors). In this case, it's for comparison
against the 'dewarped' image.

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

The `keypoints_from_samples` function goes from `span_points` (amongst other info)
to keypoints (`corners`, `ycoords` and `xcoords`).

This function is defined in `spans.py` as follows:

```py
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
```

- `PCACompute` estimates the mean orientations
  of all the spans (calculated because `mean` was
  [passed as None](https://stackoverflow.com/questions/47016617/how-to-use-the-pcacompute-function-from-python-in-opencv-3))
- `evec` is the eigenvector which indicates the major axis (just one since `maxComponents=1`
  retains only the first PC), as for when the SVD was used to compute blob tangents earlier.
- The `weight` here is the distance from the last point to the first.
- `evec` and `weight` are added to a 2-item ndarray `all_evecs` and the scalar `all_weights`

The `evec` can be flattened (equivalent to `squeeze`) to give the "x direction"
(`x_dir`), since it's assumed the spans are all running left to right,
so the vector from the first to last point in the span must be the x direction.

If the `x_dir` is negative then I presume this means the span was somehow backwards?
The code simply flips it around (put another way, the x direction is defined by convention
as positive).

The `y_dir` is defined as `[-x_dir[1], x_dir[0]])`, which is a vector
of equal magnitude to the `x_dir` vector but perpendicular (dot product 0).
The inner product of `x_dir` and `y_dir` = `(-x_dir[1]*x_dir[0]) + (x_dir[0]*x_dir[1])`
which clearly cancels to 0.

- If the x direction is a vector `(10,2)`, the y direction is defined as
  `(-2, 10)`.
  - Taking the inner product of `x_dir` and `y_dir` gives `(-2*10) + (2*10)` = 0

Next, the convex hull is taken of the `self.page_outline` which came from the first step
([[Obtain page boundaries]] in the call to `calculate_page_extents`):

> where the height of the shrunk page (to fit a HD screen) is
> offset by an x and y margin (each on both sides)

The 'page outline' corresponds to the rectangle drawn onto the pagemask using
the shrunk image's shape and the configured page margins:

```py
def calculate_page_extents(self):
    height, width = self.small.shape[:2]
    xmin = cfg.image_opts.PAGE_MARGIN_X
    ymin = cfg.image_opts.PAGE_MARGIN_Y
    xmax, ymax = (width - xmin), (height - ymin)
    self.pagemask = np.zeros((height, width), dtype=np.uint8)
    rectangle(self.pagemask, (xmin, ymin), (xmax, ymax), color=255, thickness=-1)
    self.page_outline = np.array(
        [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]
    )
```

So the page outline begins as the array of 4 corners of the page
(clockwise from bottom left), transformed by the convex hull into
the anti-clockwise direction (`clockwise=False`
[by
default](https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga014b28e56cb8854c0de4a211cb2be656)),
i.e. the order reverses so it goes anticlockwise from bottom right,
also becoming int32 dtype and getting 'squeezed' into an extra dimension.

Next `pix2norm` gets used a 2nd time (another homothety!).

Previously it was used to turn the absolute coordinates of span points
into relative coordinates with respect to the centre of the (shrunk) page.

Now it is used to turn the absolute coordinates of the corners of the page
into relative coordinates (with respect to the centre of the (shrunk) page.

For example, starting from the squeezed and anti-clockwise reordered `pagecoords`:

```py
array([[[440,  20]],
       [[440, 633]],
       [[ 50, 633]],
       [[ 50,  20]]], dtype=int32)
```

They become normalised by `pix2norm` as:

```py
array([[ 0.59724349, -0.93874426],
       [ 0.59724349,  0.93874426],
       [-0.59724349,  0.93874426],
       [-0.59724349, -0.93874426]])
```

- e.g. the first entry (the bottom right corner) `[ 0.59724349, -0.93874426]`
  can be interpreted as: "59% of the way from the centre to the right edge,
  93% of the way from the centre to the bottom edge".

It's obvious that the purpose of converting to relative values is that the
results will be applicable to the original size image, by just rescaling back up.

Next `px_coords` and `py_coords` are formed as the dot [inner] product of these
[normalised, page centre-relative] corner coords and the x and y directions.
In other words, the vectors for each of the corners are 'transported' along the vectors
for the x and y directions (which came from the average direction of the spans).

This is known as a change of basis, out of the Cartesian basis and into the (potentially
askew, not-perfectly-horizontal) one obtained from the text direction itself ("inherent")
to the text.

- Geometrically, the result is to slightly rotate the rectangle made by the corners

It would be reasonable to presume the purpose of doing so is to then un-project both the
changed-basis corners and the spans inside them back to the rectangular coordinates... 

> recall: eigenvectors from PCA calculated on the `span_points` which came from `contour_points`
> after `pix2norm` normalising against the `shape` from the shrunk image, which made
> them centre-relative, in the range `[-1,1]`)

The bounding box of these corners is calculated and stored as `corners`

```py
px0, px1 = px_coords.min(), px_coords.max()
py0, py1 = py_coords.min(), py_coords.max()
# [px0,px1,px1,px0] for first bit of p00,p10,p11,p01
x_dir_coeffs = np.pad([px0, px1], 2, mode="symmetric")[2:].reshape(-1, 1)
# [py0,py0,py1,py1] for second bit of p00,p10,p11,p01
y_dir_coeffs = np.repeat([py0, py1], 2).reshape(-1, 1)
```

The min and max of the px and py coords variables gets turned into x and
y dir coefficients as the comments say: p00,p10,p11,p01 (anti-clockwise
from bottom-left).

Then something strange happens:

```py
corners = np.expand_dims((x_dir_coeffs * x_dir) + (y_dir_coeffs * y_dir), 1)
```

The corners here came from the Cartesian bounding box of the 'rotated' (basis changed)
corners of the page mask, and in this line we re-project out of the Cartesian basis
and into the basis of the page itself...

So that is the "reprojection of the bounding box of the reprojection of the page"...
in fact it doesn't entirely undo the bending achieved by the previous projection,
it's hard to say exactly what it is doing in all honesty. The box made by
`corners` isn't rectangular

After `corners` is made, the `xcoords` and `ycoords` lists are created
from the span points, projected onto the page direction (`[x_dir,y_dir]`)
and stored relative to the bottom left corner `(px0,py0)`:

```py
px_coords, py_coords = np.dot(pts, np.transpose([x_dir, y_dir])).T
xcoords.append(px_coords - px0) 
ycoords.append(py_coords.mean() - py0)
```
