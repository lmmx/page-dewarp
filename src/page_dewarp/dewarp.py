"""Remapping and thresholding logic for page dewarping.

This module provides:
- A helper function, `round_nearest_multiple`, to round integers.
- A `RemappedImage` class that transforms an input image to a rectified,
  thresholded output using a cubic parameterization.
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
from .logging_config import get_logger
from .normalisation import norm2pix
from .options import Config
from .projection import project_xy


__all__ = ["round_nearest_multiple", "RemappedImage"]

logger = get_logger("dewarp")


def round_nearest_multiple(i: int, factor: int) -> int:
    """Round an integer up to the nearest multiple of factor."""
    i = int(i)
    rem = i % factor
    return i + factor - rem if rem else i


class RemappedImage:
    """Rectify and threshold an image based on a cubic page parameterization."""

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
            name: A string name for the output file.
            img: The original, full-resolution image.
            small: A downsampled version for debugging/display.
            page_dims: (width, height) target page dimensions.
            params: The cubic parameters for warping.
            config: A Config object with output options.

        """
        self.config = config

        height = 0.5 * page_dims[1] * config.OUTPUT_ZOOM * img.shape[0]
        height = round_nearest_multiple(height, config.REMAP_DECIMATE)
        width = round_nearest_multiple(
            height * page_dims[0] / page_dims[1],
            config.REMAP_DECIMATE,
        )

        logger.debug(
            "Output dimensions calculated",
            extra={
                "name": name,
                "width": width,
                "height": height,
            },
        )

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

        logger.info(
            "Output saved",
            extra={
                "name": name,
                "output_file": self.threshfile,
                "dimensions": f"{width}x{height}",
            },
        )

        if config.DEBUG_LEVEL >= 1:
            height = small.shape[0]
            width = int(round(height * float(thresh.shape[1]) / thresh.shape[0]))
            display = resize(thresh, (width, height), interpolation=INTER_AREA)
            debug_show(name, 6, "output", display)
