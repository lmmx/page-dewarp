"""Mask creation and manipulation for page regions and text/line detection.

This module provides:
- A helper function (`box`) for structuring elements.
- A `Mask` class for thresholding and morphological operations.
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
from .logging_config import get_logger
from .options import cfg


__all__ = ["box", "Mask"]

logger = get_logger("mask")


def box(width: int, height: int) -> np.ndarray:
    """Return a structuring element of ones with shape (height, width)."""
    return np.ones((height, width), dtype=np.uint8)


class Mask:
    """A thresholded mask builder for text (or line) detection."""

    def __init__(
        self,
        name: str,
        small: np.ndarray,
        pagemask: np.ndarray,
        text: bool | str = True,
    ) -> None:
        """Initialize the Mask with image data and detection type.

        Args:
            name: A string identifier for debugging/logging.
            small: A reduced-size version of the original image.
            pagemask: A binary mask indicating the valid page region.
            text: If True or "text", detect text; otherwise detect lines.

        """
        self.name = name
        self.small = small
        self.pagemask = pagemask
        self.text = text if isinstance(text, bool) else text == "text"
        self.calculate()

    def calculate(self) -> None:
        """Apply adaptive thresholding and morphological ops."""
        sgray = cvtColor(self.small, COLOR_RGB2GRAY)
        mask = adaptiveThreshold(
            src=sgray,
            maxValue=255,
            adaptiveMethod=ADAPTIVE_THRESH_MEAN_C,
            thresholdType=THRESH_BINARY_INV,
            blockSize=cfg.ADAPTIVE_WINSZ,
            C=25 if self.text else 7,
        )
        self._log_step(0.1, "thresholded", mask)

        if self.text:
            mask = dilate(mask, box(9, 1))
            self._log_step(0.2, "dilated", mask)
            mask = erode(mask, box(1, 3))
            self._log_step(0.3, "eroded", mask)
        else:
            mask = erode(mask, box(3, 1), iterations=3)
            self._log_step(0.5, "eroded", mask)
            mask = dilate(mask, box(8, 2))
            self._log_step(0.6, "dilated", mask)

        self.value = np.minimum(mask, self.pagemask)

        logger.debug(
            "Mask calculated",
            extra={
                "name": self.name,
                "mode": "text" if self.text else "line",
            },
        )

    def _log_step(self, step: float, text: str, display: np.ndarray) -> None:
        """Optionally display the intermediate mask state."""
        if cfg.DEBUG_LEVEL >= 3:
            if not self.text:
                step += 0.3
            debug_show(self.name, step, text, display)

    def contours(self) -> list[ContourInfo]:
        """Extract the final contours from the mask.

        Returns:
            A list of ContourInfo objects.

        """
        return get_contours(self.name, self.small, self.value)
