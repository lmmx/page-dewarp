"""Remapping and thresholding logic for page dewarping.

This module provides:

- A helper function, `round_nearest_multiple`, to round integers up to the nearest multiple.
- A `RemappedImage` class that transforms (remaps) an input image to a
  rectified, thresholded output using a cubic parameterization of the page.
"""

import numpy as np
from cv2 import (
    ADAPTIVE_THRESH_MEAN_C,
    BORDER_REPLICATE,
    COLOR_RGB2GRAY,
    INTER_AREA,
    INTER_CUBIC,
    THRESH_BINARY,
    adaptiveThreshold,
    cvtColor,
    remap,
    resize,
)
from PIL import Image

from .debug_utils import debug_show
from .normalisation import norm2pix
from .options import Config
from .projection import project_xy


__all__ = ["round_nearest_multiple", "RemappedImage"]


def round_nearest_multiple(i: int, factor: int) -> int:
    """Round an integer `i` up to the nearest multiple of `factor`.

    If `i` is already a multiple of `factor`, it remains unchanged;
    otherwise, we add the difference to `i` to reach the multiple.
    """
    i = int(i)
    rem = i % factor
    return i + factor - rem if rem else i


class RemappedImage:
    """Rectify and threshold an image based on a cubic page parameterization.

    This class takes an input image and outputs a warped, thresholded version
    (optionally binarized) according to parameters specifying the page's layout.

    Note:
        It's currently implemented as a class but may be refactored into a
        standalone function, as it stores little permanent state.

    """

    def __init__(
        self,
        name: str,
        img: np.ndarray,
        small: np.ndarray,
        page_dims: list | np.ndarray,
        params: np.ndarray,
        config: Config = Config(),
    ) -> None:
        """Initialize the remapping process and save the thresholded image.

        Args:
            name: A string name or identifier for the output file.
            img: The original, full-resolution image as a NumPy array.
            small: A downsampled version of `img` for debugging or display.
            page_dims: A (width, height) tuple or array specifying the
                target page dimensions in normalized units.
            params: The cubic parameters for warping (rotation, translation, etc.).
            config: A `Config` object containing options like zoom, DPI, debug level, etc.

        """
        self.config = config
        height = 0.5 * page_dims[1] * config.OUTPUT_ZOOM * img.shape[0]
        height = round_nearest_multiple(height, config.REMAP_DECIMATE)
        width = round_nearest_multiple(
            height * page_dims[0] / page_dims[1],
            config.REMAP_DECIMATE,
        )
        print(f"  output will be {width}x{height}")
        height_small, width_small = np.floor_divide(
            [height, width],
            config.REMAP_DECIMATE,
        )
        page_x_range = np.linspace(0, page_dims[0], width_small)
        page_y_range = np.linspace(0, page_dims[1], height_small)
        page_x_coords, page_y_coords = np.meshgrid(page_x_range, page_y_range)
        page_xy_coords = np.hstack(
            (
                page_x_coords.flatten().reshape((-1, 1)),
                page_y_coords.flatten().reshape((-1, 1)),
            ),
        )
        page_xy_coords = page_xy_coords.astype(np.float32)
        image_points = project_xy(page_xy_coords, params)
        image_points = norm2pix(img.shape, image_points, False)
        image_x_coords = image_points[:, 0, 0].reshape(page_x_coords.shape)
        image_y_coords = image_points[:, 0, 1].reshape(page_y_coords.shape)

        image_x_coords = resize(
            image_x_coords,
            (width, height),
            interpolation=INTER_CUBIC,
        ).astype(np.float32)
        image_y_coords = resize(
            image_y_coords,
            (width, height),
            interpolation=INTER_CUBIC,
        ).astype(np.float32)

        img_gray = cvtColor(img, COLOR_RGB2GRAY)
        # Ensure image_x_coords and image_y_coords are of the correct type
        remapped = remap(
            img_gray,
            image_x_coords,
            image_y_coords,
            INTER_CUBIC,
            None,
            BORDER_REPLICATE,
        )
        if config.NO_BINARY:
            thresh = remapped
            pil_image = Image.fromarray(thresh)
        else:
            thresh = adaptiveThreshold(
                remapped,
                255,
                ADAPTIVE_THRESH_MEAN_C,
                THRESH_BINARY,
                config.ADAPTIVE_WINSZ,
                25,
            )
            pil_image = Image.fromarray(thresh)
            pil_image = pil_image.convert("1")

        self.threshfile = name + "_thresh.png"
        pil_image.save(
            self.threshfile,
            dpi=(config.OUTPUT_DPI, config.OUTPUT_DPI),
        )

        if config.DEBUG_LEVEL >= 1:
            height = small.shape[0]
            width = int(round(height * float(thresh.shape[1]) / thresh.shape[0]))
            display = resize(thresh, (width, height), interpolation=INTER_AREA)
            debug_show(name, 6, "output", display)
