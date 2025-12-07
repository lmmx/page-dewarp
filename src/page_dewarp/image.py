"""Utilities for loading, resizing, and dewarping page images.

This module includes:

- A simple helper function (`imgsize`) to format an image's width/height into a string.
- A function (`get_page_dims`) to optimize final page dimensions via the cubic model.
- A class (`WarpedImage`) that loads an image, resizes it, finds page boundaries,
  and threshold-remaps the final dewarped image to disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from cv2 import INTER_AREA, imread, rectangle
from cv2 import resize as cv2_resize
from scipy.optimize import minimize

from .contours import ContourInfo
from .debug_utils import debug_show
from .dewarp import RemappedImage
from .mask import Mask
from .optimise import optimise_params
from .options import Config
from .projection import project_xy
from .solve import get_default_params
from .spans import assemble_spans, keypoints_from_samples, sample_spans


__all__ = ["imgsize", "get_page_dims", "WarpedImage"]


def imgsize(img: np.ndarray) -> str:
    """Return a string formatted as 'widthxheight' for the given image array."""
    height, width = img.shape[:2]
    return f"{width}x{height}"


def get_page_dims(
    corners: np.ndarray,
    rough_dims: np.ndarray | list,
    params: np.ndarray,
) -> np.ndarray:
    """Optimize final page dimensions using a cubic polynomial model.

    Args:
        corners: The four corner points of the page outline, in reduced coordinates.
        rough_dims: An initial (height, width) estimate for page dimensions.
        params: The optimization parameter vector, e.g. includes rotation/translation/cubic slopes.

    Returns:
        A 1D array of floats [height, width] representing the optimized page dimensions.

    """
    dst_br = corners[2].flatten()
    dims = np.array(rough_dims)

    def objective(dims_local):
        proj_br = project_xy(dims_local, params)
        return np.sum((dst_br - proj_br.flatten()) ** 2)

    res = minimize(objective, dims, method="Powell")
    dims = res.x
    print("  got page dims", dims[0], "x", dims[1])
    return dims


class WarpedImage:
    """Handles loading, resizing, and thresholding a page image.

    This class reads an image from disk, optionally downsamples it for display,
    detects page boundaries, and can threshold-remap the final dewarped image.
    """

    written = False  # Explicitly declare the file-write attribute
    config: Config

    def __init__(self, imgfile: str | Path, config: Config = Config()) -> None:
        """Initialize the WarpedImage with a source file and configuration.

        Args:
            imgfile: Path to the image file to load.
            config: A `Config` object that specifies various parameters and defaults.

        """
        self.config = config
        if isinstance(imgfile, Path):
            imgfile = str(imgfile)
        self.cv2_img = imread(imgfile)
        self.file_path = Path(imgfile).resolve()
        self.small = self.resize_to_screen()
        size, resized = self.size, self.resized
        print(f"Loaded {self.basename} at {size=} --> {resized=}")
        if self.config.DEBUG_LEVEL >= 3:
            debug_show(self.stem, 0.0, "original", self.small)

        self.calculate_page_extents()  # set pagemask & page_outline attributes
        self.contour_list = self.contour_info(text=True)
        spans = self.iteratively_assemble_spans()

        # Skip if no spans
        if len(spans) < 1:
            print(f"skipping {self.stem} because only {len(spans)} spans")
        else:
            span_points = sample_spans(self.small.shape, spans)
            n_pts = sum(map(len, span_points))
            print(f"  got {len(spans)} spans with {n_pts} points.")

            corners, ycoords, xcoords = keypoints_from_samples(
                self.stem,
                self.small,
                self.pagemask,
                self.page_outline,
                span_points,
            )
            rough_dims, span_counts, params = get_default_params(
                corners,
                ycoords,
                xcoords,
            )
            dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) + tuple(span_points))
            params = optimise_params(
                self.stem,
                self.small,
                dstpoints,
                span_counts,
                params,
                self.config.DEBUG_LEVEL,
            )
            page_dims = get_page_dims(corners, rough_dims, params)
            if np.any(page_dims < 0):
                # Fallback: see https://github.com/lmmx/page-dewarp/issues/9
                print("Got a negative page dimension! Falling back to rough estimate")
                page_dims = rough_dims
            self.threshold(page_dims, params)
            self.written = True

    def threshold(self, page_dims: np.ndarray, params: np.ndarray) -> None:
        """Construct a dewarped, thresholded image using the RemappedImage class.

        Args:
            page_dims: The final (height, width) dimensions for the page layout.
            params: The optimization parameters (e.g. rotation, translation, cubic slopes).

        """
        remap = RemappedImage(
            self.stem,
            self.cv2_img,
            self.small,
            page_dims,
            params,
            config=self.config,
        )
        self.outfile = remap.threshfile

    def iteratively_assemble_spans(self) -> list:
        """Assemble spans from contours; fallback to line detection if too few are found.

        First tries text contours to assemble spans. If fewer than three spans are found,
        attempts line detection (borders of a table box) rather than text detection,
        then re-assembles spans.
        """
        spans = assemble_spans(self.stem, self.small, self.pagemask, self.contour_list)
        # Retry if insufficient spans
        if len(spans) < 3:
            print(f"  detecting lines because only {len(spans)} text spans")
            self.contour_list = self.contour_info(text=False)  # lines not text
            spans = self.attempt_reassemble_spans(spans)
        return spans

    def attempt_reassemble_spans(self, prev_spans: list) -> list:
        """Attempt line-based re-assembly of spans, returning whichever set is larger.

        Args:
            prev_spans: The spans identified by text contour detection.

        Returns:
            The new line-detected spans if larger in number; else the original spans.

        """
        new_spans = assemble_spans(
            self.stem,
            self.small,
            self.pagemask,
            self.contour_list,
        )
        return new_spans if len(new_spans) > len(prev_spans) else prev_spans

    @property
    def basename(self) -> str:
        """Return the filename (with extension) of the loaded image."""
        return self.file_path.name

    @property
    def stem(self) -> str:
        """Return the filename (without extension) of the loaded image."""
        return self.file_path.stem

    def resize_to_screen(self, copy: bool = False) -> np.ndarray:
        """Downsample the loaded image to fit within SCREEN_MAX_W/H if needed.

        Args:
            copy: If True, returns a copy even if no resizing is needed.

        Returns:
            A potentially resized NumPy array.
            If the image is already smaller than SCREEN_MAX_W/H,
            the same array (or its copy) is returned.

        """
        height, width = self.cv2_img.shape[:2]
        scl_x = float(width) / self.config.SCREEN_MAX_W
        scl_y = float(height) / self.config.SCREEN_MAX_H
        scl = int(np.ceil(max(scl_x, scl_y)))
        if scl > 1.0:
            inv_scl = 1.0 / scl
            img = cv2_resize(self.cv2_img, (0, 0), None, inv_scl, inv_scl, INTER_AREA)
        elif copy:
            img = self.cv2_img.copy()
        else:
            img = self.cv2_img
        return img

    def calculate_page_extents(self) -> None:
        """Create a mask for the page region, ignoring margins around the edges."""
        height, width = self.small.shape[:2]
        xmin = self.config.PAGE_MARGIN_X
        ymin = self.config.PAGE_MARGIN_Y
        xmax, ymax = (width - xmin), (height - ymin)
        self.pagemask = np.zeros((height, width), dtype=np.uint8)
        rectangle(self.pagemask, (xmin, ymin), (xmax, ymax), color=255, thickness=-1)
        self.page_outline = np.array(
            [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]],
        )

    @property
    def size(self) -> str:
        """Return a formatted string 'widthxheight' for the original (full) image."""
        return imgsize(self.cv2_img)

    @property
    def resized(self) -> str:
        """Return a formatted string 'widthxheight' for the downsampled (small) image."""
        return imgsize(self.small)

    def contour_info(self, text: bool = True) -> list[ContourInfo]:
        """Compute contour information for either text or line detection.

        Args:
            text: If True, identifies text contours; otherwise detects lines.

        Returns:
            A list of contour objects (ContourInfo instances).

        """
        c_type = "text" if text else "line"
        mask = Mask(self.stem, self.small, self.pagemask, c_type)
        return mask.contours()
