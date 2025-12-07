"""Mask creation and manipulation for page regions and text/line detection.

This module provides:

- A helper function (`box`) to generate structuring elements for morphological ops.
- A `Mask` class, which thresholds the page image, applies morphological operations,
  and blends the resulting mask with a page mask.
- An interface to retrieve the final contours from this mask.
"""

import numpy as np
from cv2 import (
    ADAPTIVE_THRESH_MEAN_C,
    COLOR_RGB2GRAY,
    THRESH_BINARY_INV,
    adaptiveThreshold,
    cvtColor,
    dilate,
    erode,
)

from .contours import ContourInfo, get_contours
from .debug_utils import debug_show
from .options import cfg


__all__ = ["box", "Mask"]


def box(width: int, height: int) -> np.ndarray:
    """Return a structuring element of ones with shape (height, width).

    Used in morphological operations (e.g., dilate, erode).
    """
    return np.ones((height, width), dtype=np.uint8)


class Mask:
    """A thresholded mask builder for text (or line) detection.

    Combines adaptive thresholding, morphological dilations/erosions, and
    a given `pagemask` to produce the final mask used for contour extraction.
    """

    def __init__(
        self,
        name: str,
        small: np.ndarray,
        pagemask: np.ndarray,
        text: bool | str = True,
    ) -> None:
        """Initialize the Mask with the given image data and type.

        Args:
            name: A string identifier for debugging/logging.
            small: A reduced-size (downsampled) version of the original image.
            pagemask: A binary mask indicating the valid page region.
            text: If True, process as text; if False, process as lines.

        """
        self.name = name
        self.small = small
        self.pagemask = pagemask
        self.text = text
        self.calculate()

    def calculate(self) -> None:
        """Apply adaptive thresholding and morphological ops to create `self.value`.

        Steps:

        1. Convert `self.small` to grayscale.
        2. Use an adaptive threshold (binary inverse).
        3. Depending on `self.text`, either dilate or erode the result, log intermediate steps.
        4. Combine with `self.pagemask` to finalize the mask (store in `self.value`).
        """
        sgray = cvtColor(self.small, COLOR_RGB2GRAY)
        mask = adaptiveThreshold(
            src=sgray,
            maxValue=255,
            adaptiveMethod=ADAPTIVE_THRESH_MEAN_C,
            thresholdType=THRESH_BINARY_INV,
            blockSize=cfg.ADAPTIVE_WINSZ,
            C=25 if self.text else 7,
        )
        self.log(0.1, "thresholded", mask)

        # If text, dilate horizontally; if lines, erode to remove noise
        mask = (
            dilate(mask, box(9, 1))
            if self.text
            else erode(mask, box(3, 1), iterations=3)
        )
        self.log(0.2, "dilated" if self.text else "eroded", mask)

        # If text, erode vertically; if lines, dilate further
        mask = erode(mask, box(1, 3)) if self.text else dilate(mask, box(8, 2))
        self.log(0.3, "eroded" if self.text else "dilated", mask)

        self.value = np.minimum(mask, self.pagemask)

    def log(self, step: float, text: str, display: np.ndarray) -> None:
        """Optionally display or log the intermediate mask state at a given step.

        Args:
            step: A numeric code or fraction indicating the process step.
            text: A label describing what operation was just done (e.g. 'dilated').
            display: The mask or image array to show for debugging.

        """
        if cfg.DEBUG_LEVEL >= 3:
            if not self.text:
                # text images from 0.1 to 0.3, table images from 0.4 to 0.6
                step += 0.3
            debug_show(self.name, step, text, display)

    def contours(self) -> list[ContourInfo]:
        """Extract the final contours from `self.value`.

        Calls `get_contours` to find external contours in the thresholded,
        morphological-processed mask stored in `self.value`.

        Returns:
            A list of ContourInfo objects describing each discovered contour.

        """
        return get_contours(self.name, self.small, self.value)
