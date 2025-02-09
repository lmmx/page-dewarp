"""Displays or saves debug images for page-dewarp.

If `cfg.DEBUG_OUTPUT` is "file", the debug image is saved to disk with a filename
indicating `name`, `step`, and `text`. Otherwise (if `cfg.DEBUG_OUTPUT` is "screen"
or "both"), the image is displayed in an OpenCV window with an overlaid label,
and the script waits for a keypress to close the window.
"""

import numpy as np
from cv2 import FONT_HERSHEY_SIMPLEX, LINE_AA, imshow, imwrite, putText, waitKey

from ..options import cfg


__all__ = ["debug_show"]


def debug_show(
    name: str,
    step: float | int | str,
    text: str,
    display: np.ndarray,
) -> None:
    """Show or save a debug image, possibly with an overlay of text.

    Args:
        name: A string identifier for this debug image (e.g. "dewarp" or "contours").
        step: A numeric or string code indicating the processing step.
        text: A description to overlay on the image or use in the filename.
        display: A NumPy array (image) to show or save.

    """
    if cfg.DEBUG_OUTPUT != "screen":
        filetext = text.replace(" ", "_")
        outfile = f"{name}_debug_{step}_{filetext}.png"
        imwrite(outfile, display)

    if cfg.DEBUG_OUTPUT != "file":
        image = display.copy()
        height = image.shape[0]

        putText(
            image,
            text,
            (16, height - 16),
            FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            3,
            LINE_AA,
        )
        putText(
            image,
            text,
            (16, height - 16),
            FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            1,
            LINE_AA,
        )
        imshow("Dewarp", image)

        while waitKey(5) < 0:
            pass
