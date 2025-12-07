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
    """

    # [camera_opts]
    FOCAL_LENGTH: desc(float, "Normalized focal length of camera") = 1.2
    # [contour_opts]
    TEXT_MIN_WIDTH: desc(int, "Min reduced px width of detected text contour") = 15
    TEXT_MIN_HEIGHT: desc(int, "Min reduced px height of detected text contour") = 2
    TEXT_MIN_ASPECT: desc(float, "Filter out text contours below this w/h ratio") = 1.5
    TEXT_MAX_THICKNESS: desc(
        int,
        "Max reduced px thickness of detected text contour",
    ) = 10
    # [debug_lvl_opt]
    DEBUG_LEVEL: int = 0
    # [debug_out_opt]
    DEBUG_OUTPUT: str = "file"
    # [edge_opts]
    EDGE_MAX_OVERLAP: desc(
        float,
        "Max reduced px horiz. overlap of contours in span",
    ) = 1.0
    EDGE_MAX_LENGTH: desc(
        float,
        "Max reduced px length of edge connecting contours",
    ) = 100.0
    EDGE_ANGLE_COST: desc(float, "Cost of angles in edges (tradeoff vs. length)") = 10.0
    EDGE_MAX_ANGLE: desc(float, "Maximum change in angle allowed between contours") = (
        7.5
    )
    # [image_opts]
    SCREEN_MAX_W: desc(int, "Viewing screen max width (for resizing to screen)") = 1280
    SCREEN_MAX_H: desc(int, "Viewing screen max height (for resizing to screen)") = 700
    PAGE_MARGIN_X: desc(int, "Reduced px to ignore near L/R edge") = 50
    PAGE_MARGIN_Y: desc(int, "Reduced px to ignore near T/B edge") = 20
    # [mask_opts]
    ADAPTIVE_WINSZ: desc(int, "Window size for adaptive threshold in reduced px") = 55
    # [output_opts]
    OUTPUT_ZOOM: desc(float, "How much to zoom output relative to *original* image") = (
        1.0
    )
    OUTPUT_DPI: desc(int, "Just affects stated DPI of PNG, not appearance") = 300
    REMAP_DECIMATE: desc(int, "Downscaling factor for remapping image") = 16
    NO_BINARY: desc(int, "Disable output conversion to binary thresholded image") = 0
    # [pdf_opts]
    CONVERT_TO_PDF: desc(bool, "Merge dewarped images into a PDF") = False
    # [proj_opts]
    RVEC_IDX: desc(
        tuple[int, int],
        "Index of rvec in params vector (slice: pair of values)",
    ) = (0, 3)
    TVEC_IDX: desc(
        tuple[int, int],
        "Index of tvec in params vector (slice: pair of values)",
    ) = (3, 6)
    CUBIC_IDX: desc(
        tuple[int, int],
        "Index of cubic slopes in params vector (slice: pair of values)",
    ) = (6, 8)
    # [span_opts]
    SPAN_MIN_WIDTH: desc(int, "Minimum reduced px width for span") = 30
    SPAN_PX_PER_STEP: desc(int, "Reduced px spacing for sampling along spans") = 20


cfg = Config()
