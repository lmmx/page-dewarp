Text contours are detected when `image.py`'s `WarpedImage` initialisation method
calls `self.contour_list = self.contour_info(text=True)`.

```py
def contour_info(self, text=True): 
    c_type = "text" if text else "line" 
    mask = Mask(self.stem, self.small, self.pagemask, c_type) 
    return mask.contours() 
```

The `contour_info` method is just a gate to detect either text contours
(i.e. to detect lines of text) or line contours (table borders etc.).

It makes a `Mask` and then calls the `contours` method on that `Mask`.

- The `Mask` class comes from the `mask.py` module, and uses the name (`stem`),
  shrunk image (`small`), rectangular black and white mask `pagemask` and
  contour type (here `"text"`).

The component steps are:

## Adaptive threshold

- The shrunk image is converted to a grayscale copy, `sgray`, by mixing the RGB channels
  into a single channel (reducing the final dimension of the image from 3 to 1)
- The grayscale image is binarised from 8-bit (the output is also 8-bit but only contains
  the values 0 or 255)
  - The threshold type is either binary (black stays black, white stays white) or inverse binary
    (white becomes black and vice versa, so black text pixels with low grayscale value near 0 become
    high values when binarised, near 255). We use the inverse binary so text becomes high valued
    and logically means "True" or "on" (as masks are used for logic operations).

```py
sgray = cvtColor(self.small, COLOR_RGB2GRAY)
mask = adaptiveThreshold(
    src=sgray,
    maxValue=255,
    adaptiveMethod=ADAPTIVE_THRESH_MEAN_C,
    thresholdType=THRESH_BINARY_INV,
    blockSize=cfg.mask_opts.ADAPTIVE_WINSZ,
    C=25 if self.text else 7,
)
```

## Dilation and erosion


(These steps are applied in reverse order for the table borders)

```py
mask = dilate(mask, box(9, 1)) if self.text else erode(mask, box(3, 1), iterations=3)
mask = erode(mask, box(1, 3)) if self.text else dilate(mask, box(8, 2))
```

The pagemask is then 'applied' to the dilated/eroded mask by choosing the minimum, i.e. all negative/'off'
pixels in the mask will be the minimum even if a text contour was detected there, so will be
'switched off' or ignored in the mask.

## Filtering step to eliminate blobs

> "which are too tall (compared to their width) or too thick to be text"

Before the connected component analysis happens, the filtering step happens

Back in `image.py`, the `WarpedImage` class immediately calls the `contours` method of the `Mask`,
which wraps a call to `get_contours` from `contours.py`.

```py
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
```

This procedure checks if any of the following conditions are met:

- the width of the bounding box of each [text] contour (i.e. the outline of some text) is 
  below the `TEXT_MIN_WIDTH` (default: 15px)
- its height is below `TEXT_MIN_HEIGHT` (default: 2px)
- its aspect ratio is below `TEXT_MIN_ASPECT` (default: 1.5 i.e. width:height 3:2),
  i.e. it should be significantly wider than it is tall

```py
def make_tight_mask(contour, xmin, ymin, width, height):
```

It then runs the `make_tight_mask` function (whose signature is given above) and
checks if the maximum of the column-wise (`axis=0`) totals is below the pre-set
`TEXT_MAX_THICKNESS` (default: 10px) before accepting the contour

- In other words, if any column in a detected piece of text has more than 10 pixels,
  the entire block will be discarded as "too thick"
  - You might imagine something like a shaded rectangle or ellipse in a diagram matching these
    criteria. Note that there are no other checks in place to prevent overly large objects
    being detected as 'text', so the 'thickness' check is a way of preventing large and
    'blocky' or 'chunky' marks from being registered as text. It probably wouldn't permit
    text drawn with a thick marker pen for example.

```py
tight_mask = np.zeros((height, width), dtype=np.uint8)
tight_contour = contour - np.array((xmin, ymin)).reshape((-1, 1, 2))
drawContours(tight_mask, [tight_contour], contourIdx=0, color=1, thickness=-1)
return tight_mask
```

- First the mask is initialised with all zeroes, with the same width and height as the text
  region described by the contour (note: __not__ simply the shape of the contour array)
- The `tight_contour` is formed by subtracting the contour's bottom left coordinate,
  "image"-wide (i.e. reshaped to match the dimension of the image: shape `1,1,2` to the image's
  `{number_of_contour_points},1,2`)
  - I would describe this as having an effect of making the coordinates of the contour relative
    to its bottom-left corner
- The contour is drawn by connecting the points on the mask (similar to the `cv2.rectangle`
  earlier), with `cv2.drawContours` (but passing a list of a single contour at a time)
  - Here the fill colour is 1 (so that the column total is a count of filled pixels)
  - Again, the thickness of `-1` means "filled" rather than outline
  - The `contourIdx` argument "indicates a contour to draw": so the 0 indicates the first item in
    the singleton list (the only item)

...and that's the end of the sequence of events that happened when `Mask.contours()` was called
within the `contour_info` method during initialisation of the `WarpedImage` class, to populate its
`contour_list` attribute:

```py
self.contour_list = self.contour_info(text=True)
```

- Recall that this call began in `image.py`, the mask was made in `mask.py` using the contour function
  from `contours.py`. Now step back to `image.py` to proceed.
- As mentioned above, this gets re-run with `text=False` to do table borders but we'll omit that as
  it's very similar to this part.

## Connected component analysis

Next in the `WarpedImage` initialisation comes `iteratively_assemble_spans`, whose docstring says:

> First try to assemble spans from contours, if too few spans then make spans by
> line detection (borders of a table box) rather than text detection.

This is referred to as "connected component analysis" (i.e. going from pixels to symbols, by
grouping or 'labeling' them according to some connectivity requirement, either 4- or 8-connected).

- See: [Connected-component labeling](https://en.wikipedia.org/wiki/Connected-component_labeling)
  - e.g. [`skimage.measure.label`](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label)

Here, we go from the pixel lines (contours) to symbols called 'spans'. The default variables in the
config for this section are `SPAN_MIN_WIDTH` of 30px and `SPAN_PX_PER_STEP` of 20px ("reduced
spacing for sampling along spans").

Again we step into a function: `assemble_spans`, from `spans.py`

```py
spans = assemble_spans(self.stem, self.small, self.pagemask, self.contour_list)
```
⇣
```py
def assemble_spans(name, small, pagemask, cinfo_list):
    cinfo_list = sorted(cinfo_list, key=lambda cinfo: cinfo.rect[1])
    candidate_edges = []
    for i, cinfo_i in enumerate(cinfo_list):
        for j in range(i):
            # note e is of the form (score, left_cinfo, right_cinfo)
            edge = generate_candidate_edge(cinfo_i, cinfo_list[j])
            if edge is not None:
                candidate_edges.append(edge)
```

- First the contours are sorted by the 2nd element of the `rect` (its `y` value), so
  contours are ordered from bottom-most to upper-most last
  - Note that they're not sorted by x value, just y value
  - Recall: the `rect` attribute was the `boundingRect` of the contour, whose elements are
    `x,y,w,h`
- The y-sorted contour list is iterated through (i.e. iterating "upwards") and `generate_candidate_edge`
  is called on all possible pairs of that contour and _every_ previous one in the list (i.e. every
  one with a bounding rectangle base below the current contour's bounding rectangle base)

Before we look at the rest of the `assemble_spans` function, let's look at what
`generate_candidate_edge` does (it's a little complicated, pay close attention).
It comes from the same module, `spans.py`

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

- The process of generating candidate edges is covered in more detail in the next section
  in the context of span assembly from the candidates

The attributes it's using (`point0`, `point1` [the leftmost and rightmost point in the contour],
`center`, and `angle`) were set in the initialisation of the `ContourInfo` class in `contours.py`:

```py
def __init__(self, contour, rect, mask):
    self.contour = contour
    self.rect = rect
    self.mask = mask
    self.center, self.tangent = blob_mean_and_tangent(contour)
    self.angle = np.arctan2(self.tangent[1], self.tangent[0])
    clx = [self.proj_x(point) for point in contour]
    lxmin, lxmax = min(clx), max(clx)
    self.local_xrng = (lxmin, lxmax)
    self.point0 = self.center + self.tangent * lxmin
    self.point1 = self.center + self.tangent * lxmax
    self.pred = None
    self.succ = None
```

where the `center` and `tangent` attributes were set by this function:

```py
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
```

- The "moments" here are image moments. I couldn't find a clearly written exposition
  of image moments so I wrote one: see [[Background on image moments]]
- Computing SVD of the covariance matrix (which you should note is a 2x2 matrix) gives
  the 2 eigenvalues: the principal components which give the orientation, the first of
  which is the major axis (`svd_u[:, 0]`)

The `local_overlap` method being used to calculate x axis overlap was also defined on
the `ContourInfo` class:

```py
def local_overlap(self, other):
    xmin = self.proj_x(other.point0)
    xmax = self.proj_x(other.point1)
    return interval_measure_overlap(self.local_xrng, (xmin, xmax))
```

where the `local_xrng` attribute is set in the `ContourInfo` initialisation as:

```py
clx = [self.proj_x(point) for point in contour]
lxmin, lxmax = min(clx), max(clx)
self.local_xrng = (lxmin, lxmax)
```

...using `proj_x` which takes the dot product `np.dot(self.tangent, point.flatten() - self.center)`
(i.e. between the contour direction, `tangent`, and the relative position vector of the point
w.r.t. the blob centre).

The title of this function indicates the assumption that the text we've contoured is running
from left to right: the tangent of the blob is in the x direction, and so the values of the
leftmost and rightmost will have the most negative and most positive values.

- The left- and right-most points on the contour will be the most on the tangent, and thus
  most in the range or column space of the tangent vector, whereas the intermediate points
  such as those above and below the centre will be more orthogonal to the tangent, and thus
  their projected value (dot product with the tangent vector) will fall nearer to zero.
- Long story short, the `local_xrng` indicates the min and max projections, from which the
  corresponding points are recreated by reprojecting the tangent along these values from the centre
  to regain `self.point0` and `self.point1` (leftmost and rightmost points on the contour)

The `interval_measure_overlap` function which `local_overlap` wraps is simply returning:

```py
min(int_a[1], int_b[1]) - max(int_a[0], int_b[0])
```

i.e. it's using its own projection of the other blob's leftmost and rightmost points

## Text contours are approximated by their best fitting line segment using PCA

This is just reuse of the aforementioned SVD PCA tangent-relative leftmost and
rightmost points, joined by a line in the `visualize_contours` function (with
a circle at the midpoint, `ContourInfo.center`)

```py
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
```

(This actually comes at the end of the span assembly, which is the next step:
see the next part of this series)
