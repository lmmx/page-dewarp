"""Core configuration structures for page-dewarp.

Defines:

- A helper function (`desc`) that annotates msgspec.Struct fields with a description.
- A global `Config` class specifying various parameters (camera, edge detection, etc.).
"""

from __future__ import annotations

from typing import Annotated

from msgspec import Meta, Struct


__all__ = ["Config", "cfg"]


def desc(typ, /, description: str):
    """Annotate a `msgspec.Struct` field with a description.

    Returns an `Annotated[typ, Meta(description=...)]` for additional metadata.
    """
    return Annotated[typ, Meta(description=description)]


class Config(Struct):
    """Global configuration for page-dewarp.

    Holds parameters controlling camera focal length, contour detection,
    output size, page margin, debug verbosity, etc.

    Attributes:
        FOCAL_LENGTH (float): Normalized focal length of camera.
        TEXT_MIN_WIDTH (int): Minimum reduced pixel width of detected text contour.
        TEXT_MIN_HEIGHT (int): Minimum reduced pixel height of detected text contour.
        TEXT_MIN_ASPECT (float): Filter out text contours below this width/height ratio.
        TEXT_MAX_THICKNESS (int): Maximum reduced pixel thickness of detected text contour.
        DEBUG_LEVEL (int): Debug verbosity level (0 = none).
        DEBUG_OUTPUT (str): Output mode for debug information ('file' by default).
        EDGE_MAX_OVERLAP (float): Maximum horizontal overlap of contours in a span.
        EDGE_MAX_LENGTH (float): Maximum length of edges connecting contours.
        EDGE_ANGLE_COST (float): Cost of angles in edges (tradeoff vs length).
        EDGE_MAX_ANGLE (float): Maximum allowed change in angle between contours.
        SCREEN_MAX_W (int): Viewing screen maximum width (for resizing to screen).
        SCREEN_MAX_H (int): Viewing screen maximum height (for resizing to screen).
        PAGE_MARGIN_X (int): Pixels to ignore near left/right edge.
        PAGE_MARGIN_Y (int): Pixels to ignore near top/bottom edge.
        ADAPTIVE_WINSZ (int): Window size for adaptive thresholding.
        OUTPUT_ZOOM (float): Zoom factor for output relative to original image.
        OUTPUT_DPI (int): Stated DPI of output PNG (does not affect appearance).
        REMAP_DECIMATE (int): Downscaling factor for remapping images.
        NO_BINARY (int): Disable output conversion to binary thresholded image.
        CONVERT_TO_PDF (bool): Merge dewarped images into a PDF.
        RVEC_IDX (tuple[int, int]): Index slice of rotation vector in parameter vector.
        TVEC_IDX (tuple[int, int]): Index slice of translation vector in parameter vector.
        CUBIC_IDX (tuple[int, int]): Index slice of cubic slopes in parameter vector.
        SPAN_MIN_WIDTH (int): Minimum width of a span in reduced pixels.
        SPAN_PX_PER_STEP (int): Pixel spacing for sampling along spans.

    """

    OPT_MAX_ITER: desc(int, "Maximum Powell's method optimisation iterations") = 600_000
    """
    Maximum Powell's method optimisation iterations.

    Note:
        For a fast optimisation (to see a quick 'draft'), set it to a low value like 1.

    This value is passed as `maxiter` to
    [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html),
    which defaults to `N*1000` where N is the number of parameter variables (in our
    case, 600).
    """
    # [camera_opts]
    FOCAL_LENGTH: desc(float, "Normalized focal length of camera") = 1.2
    """Normalized focal length of camera."""
    # [contour_opts]
    TEXT_MIN_WIDTH: desc(int, "Min reduced px width of detected text contour") = 15
    """Min reduced px width of detected text contour."""
    TEXT_MIN_HEIGHT: desc(int, "Min reduced px height of detected text contour") = 2
    """Min reduced px height of detected text contour."""
    TEXT_MIN_ASPECT: desc(float, "Filter out text contours below this w/h ratio") = 1.5
    """Filter out text contours below this w/h ratio."""
    TEXT_MAX_THICKNESS: desc(
        int,
        "Max reduced px thickness of detected text contour",
    ) = 10
    """Max reduced px thickness of detected text contour."""
    # [debug_lvl_opt]
    DEBUG_LEVEL: int = 0
    # [debug_out_opt]
    DEBUG_OUTPUT: str = "file"
    # [edge_opts]
    EDGE_MAX_OVERLAP: desc(
        float,
        "Max reduced px horiz. overlap of contours in span",
    ) = 1.0
    """Max reduced px horiz. overlap of contours in span."""
    EDGE_MAX_LENGTH: desc(
        float,
        "Max reduced px length of edge connecting contours",
    ) = 100.0
    """Max reduced px length of edge connecting contours."""
    EDGE_ANGLE_COST: desc(float, "Cost of angles in edges (tradeoff vs. length)") = 10.0
    """Cost of angles in edges (tradeoff vs. length)."""
    EDGE_MAX_ANGLE: desc(float, "Maximum change in angle allowed between contours") = (
        7.5
    )
    """Maximum change in angle allowed between contours."""
    # [image_opts]
    SCREEN_MAX_W: desc(int, "Viewing screen max width (for resizing to screen)") = 1280
    """Viewing screen max width (for resizing to screen)."""
    SCREEN_MAX_H: desc(int, "Viewing screen max height (for resizing to screen)") = 700
    """Viewing screen max height (for resizing to screen)."""
    PAGE_MARGIN_X: desc(int, "Reduced px to ignore near L/R edge") = 50
    """Reduced px to ignore near L/R edge."""
    PAGE_MARGIN_Y: desc(int, "Reduced px to ignore near T/B edge") = 20
    """Reduced px to ignore near T/B edge."""
    # [mask_opts]
    ADAPTIVE_WINSZ: desc(int, "Window size for adaptive threshold in reduced px") = 55
    """Window size for adaptive threshold in reduced px."""
    # [output_opts]
    OUTPUT_ZOOM: desc(float, "How much to zoom output relative to *original* image") = (
        1.0
    )
    """How much to zoom output relative to *original* image.

    Note:
        This controls output resolution, so 2.0 roughly (not exactly) doubles the
        size. For example in [#19](https://github.com/lmmx/page-dewarp/issues/19):

        - `1` => 800 x 1248 px (default)
        - `2` => 1568 x 2480 px
        - `3` => 2352 x 3712 px
    """
    OUTPUT_DPI: desc(int, "Just affects stated DPI of PNG, not appearance") = 300
    """Just affects stated DPI of PNG, not appearance."""
    REMAP_DECIMATE: desc(int, "Downscaling factor for remapping image") = 16
    """Downscaling factor for remapping image."""
    NO_BINARY: desc(int, "Disable output conversion to binary thresholded image") = 0
    """Disable output conversion to binary thresholded image."""
    # [pdf_opts]
    CONVERT_TO_PDF: desc(bool, "Merge dewarped images into a PDF") = False
    """Merge dewarped images into a PDF."""
    # [proj_opts]
    RVEC_IDX: desc(
        tuple[int, int],
        "Index of rvec in params vector (slice: pair of values)",
    ) = (0, 3)
    """Index of rvec in params vector (slice: pair of values)."""
    TVEC_IDX: desc(
        tuple[int, int],
        "Index of tvec in params vector (slice: pair of values)",
    ) = (3, 6)
    """Index of tvec in params vector (slice: pair of values)."""
    CUBIC_IDX: desc(
        tuple[int, int],
        "Index of cubic slopes in params vector (slice: pair of values)",
    ) = (6, 8)
    """Index of cubic slopes in params vector (slice: pair of values)."""
    # [span_opts]
    SPAN_MIN_WIDTH: desc(int, "Minimum reduced px width for span") = 30
    """Minimum reduced px width for span."""
    SPAN_PX_PER_STEP: desc(int, "Reduced px spacing for sampling along spans") = 20
    """Reduced px spacing for sampling along spans."""


cfg = Config()
