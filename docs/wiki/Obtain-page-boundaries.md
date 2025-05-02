## Summary

X and Y margins are subtracted from a shrunk black and white copy of the image.

## Details

The story begins with the `WarpedImage` class in the
[`image.py`](https://github.com/lmmx/page-dewarp/blob/master/src/page_dewarp/image.py) module.

- When initialised, it reads in the image file

```py
self.cv2_img = cv2.imread(imgfile)
```

- A smaller version is made (stored in the `.small` attribute) by calling `resize_to_screen()`.

```py
def resize_to_screen(self, copy=False):
    height, width = self.cv2_img.shape[:2]
    scl_x = float(width) / cfg.image_opts.SCREEN_MAX_W
    scl_y = float(height) / cfg.image_opts.SCREEN_MAX_H
    scl = int(np.ceil(max(scl_x, scl_y)))
    if scl > 1.0:
        inv_scl = 1.0 / scl
        img = cv2.resize(
            self.cv2_img, (0, 0), None, inv_scl, inv_scl, cv2.INTER_AREA
        )
    elif copy:
        img = self.cv2_img.copy()
    else:
        img = self.cv2_img
    return img
```

- The `SCREEN_MAX_W` and `SCREEN_MAX_H` parameters are configured through the global
  `cfg` object. They assume a roughly HD screen size for viewing intermediate results (1280x700)
  - `cfg` is constructed by loading defaults from a TOML config file which ships with the package
    (see [`options.py`](https://github.com/lmmx/page-dewarp/blob/master/src/page_dewarp/options.py))
    but all of these settings can be modified using command line flags
    (see [`cli.py`](https://github.com/lmmx/page-dewarp/blob/master/src/page_dewarp/cli.py))

The page boundaries are obtained by the call to `calculate_page_extents`, where the height of the
shrunk page (to fit a HD screen) is offset by an x and y margin (each on both sides).

- An all-zero page mask [8-bit integer numpy array, i.e. 0-255] is created
  - (i.e. to initialise the mask as "non-page" or "off" but as a colour: black),
- a rectangle is overlaid (in-place) onto the pagemask according to the margin size,
  with colour `255` (i.e. white), with a negative value for `thickness` which is a
  sentinel value meaning 'filled' rather than outline of a rectangle

```py
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

Note that the page outline is a list (matrix) of the corners
clockwise from bottom left (BL, TL, TR, BR).
