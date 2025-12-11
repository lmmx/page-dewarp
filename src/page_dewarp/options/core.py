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
        OPT_MAX_ITER (int): Maximum optimisation iterations.
        OPT_METHOD (str): Name of the JAX/SciPy optimisation method to use.
        USE_BATCH (str): Whether to batch process images (JAX backend only).
        DEVICE (str): JAX device to select for optimisation.
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
        SHEAR_COST (float): Penalty against camera tilt (shear distortion).
        MAX_CORR (int): Maximum corrections used to approximate the inverse Hessian.
        RVEC_IDX (tuple[int, int]): Index slice of rotation vector in parameter vector.
        TVEC_IDX (tuple[int, int]): Index slice of translation vector in parameter vector.
        CUBIC_IDX (tuple[int, int]): Index slice of cubic slopes in parameter vector.
        SPAN_MIN_WIDTH (int): Minimum width of a span in reduced pixels.
        SPAN_PX_PER_STEP (int): Pixel spacing for sampling along spans.

    """

    OPT_MAX_ITER: desc(int, "Maximum optimisation iterations") = 600_000
    """
    Maximum optimisation iterations.

    Tip:
       For a fast 'draft' preview, set this to a low value like 1 with `-it 1`.

    Note:
       This value is passed as `maxiter` to JAX or
       [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html),
       which defaults to `N*1000` where N is the number of parameter variables (in our case, 600).
    """
    OPT_METHOD: desc(str, "Name of the JAX/SciPy optimisation method to use.") = "auto"
    """
    Name of the JAX/SciPy optimisation method to use.

    JAX supports L-BFGS-B only (its default). It is typically several times faster than
    Powell's method (SciPy's default), and more accurate than SciPy's L-BFGS-B.

    Tip:
       Install the `jax` Python package to use JAX reverse-mode autodifferentiation to
       produce gradients for L-BFGS-B (recommended). It is much faster than Powell's
       method with SciPy, typically with far fewer function evaluations and a better
       result.

    In SciPy, Powell's method is slower than methods like L-BFGS-B, but it avoids local minima
    better in high-dimensional parameter spaces because SciPy's gradients are lower
    quality so produce worse optimisations when used by gradient methods like L-BFGS-B.

    Note:
       This name is passed as `method` to
       [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/optimize.minimize.html),
       and defaults to `"Powell"` if unset.

       All options:

       - Nelder-Mead
       - Powell
       - CG
       - BFGS
       - Newton-CG
       - L-BFGS-B
       - TNC
       - COBYLA
       - COBYQA
       - SLSQP
       - trust-const
       - dogleg
       - trust-ncg
       - trust-exact
       - trust-krylov
    """
    USE_BATCH: desc(str, "Whether to batch process images (JAX backend only).") = "auto"
    """Whether to batch process images (JAX backend only).

    Options:
       - "auto": Use batch if multiple images, otherwise serial (default)
       - "on" or "1": Process images in batches.
       - "off" or "0": Process images serially. Recommended for debugging a single case.
    """
    DEVICE: desc(str, "Compute device to select for optimisation.") = "auto"
    """Compute device to select for optimisation.

    Options:
       - "auto": Use GPU if available, otherwise CPU (default)
       - "cpu": Force CPU execution
       - "gpu": Use the default GPU
       - "gpu:N": Use a specific GPU by index (e.g., "gpu:0", "gpu:1")

    Tip:
        CPU is typically faster for page-dewarp's optimization problem size.
        The JAX speedup comes from efficient autodiff, not GPU parallelism.
        GPU support is available for experimentation but rarely helps.

    Note:
        GPU support requires the `jax-cuda12` or `jax-cuda13` extra.
        Only applies when using the JAX backend (L-BFGS-B method).

        - `pip install page-dewarp[jax-cuda12]` (CUDA 12)
        - `pip install page-dewarp[jax-cuda13]` (CUDA 13, requires Python 3.11+)
    """
    # [camera_opts]
    FOCAL_LENGTH: desc(float, "Normalized focal length of camera") = 1.2
    """Normalized focal length of camera."""
    # [contour_opts]
    TEXT_MIN_WIDTH: desc(int, "Min reduced px width of detected text contour") = 15
    """Min reduced px width of detected text contour.

    Contours narrower than this are filtered out.

    Tip:
       Decrease for small text, increase to filter out noise.

    Question:
       [#78](https://github.com/lmmx/page-dewarp/issues/78) - Discussion of text detection robustness
    """
    TEXT_MIN_HEIGHT: desc(int, "Min reduced px height of detected text contour") = 2
    """Min reduced px height of detected text contour.

    Contours shorter than this are filtered out.

    Question:
       [#78](https://github.com/lmmx/page-dewarp/issues/78) - Discussion of text detection robustness
    """
    TEXT_MIN_ASPECT: desc(float, "Filter out text contours below this w/h ratio") = 1.5
    """Filter out text contours below this w/h ratio.

    Note:
       Text is typically wider than tall, so this filters vertical artifacts.
       Decrease for languages with tall characters or rotated text.

    Question:
       [#78](https://github.com/lmmx/page-dewarp/issues/78) - Discussion of text detection robustness
    """
    TEXT_MAX_THICKNESS: desc(
        int,
        "Max reduced px thickness of detected text contour",
    ) = 10
    """Max reduced px thickness of detected text contour.

    Contours thicker than this are filtered out (likely not text).

    Tip:
       For bold letters or close-up photos where letters are large, the morphological
       smearing may not connect letters into word blobs effectively. Consider
       adjusting this alongside `TEXT_MIN_WIDTH`.

    Question:
       [#78](https://github.com/lmmx/page-dewarp/issues/78) - Discussion of text detection limitations
       with close-up photos.
    """
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
    """Reduced px to ignore near L/R edge.

    Tip:
       Set to `0` when text extends to the page edges and you don't want
       content cropped from the sides at all.

    Question:
       [#83](https://github.com/lmmx/page-dewarp/issues/83): Dewarp failure example
       using `-x 0 -y 0`
    """
    PAGE_MARGIN_Y: desc(int, "Reduced px to ignore near T/B edge") = 20
    """Reduced px to ignore near T/B edge.

    Tip:
       Set to `0` when text extends to the top/bottom of the frame and you don't want
       content cropped from either end.

    Question:
       [#83](https://github.com/lmmx/page-dewarp/issues/83): Dewarp failure example
       using `-x 0 -y 0`.
   """
    # [mask_opts]
    ADAPTIVE_WINSZ: desc(int, "Window size for adaptive threshold in reduced px") = 55
    """Window size for adaptive threshold in reduced px.

    Warning:
       Must be an **odd number**.

    Tip:
       Increase this value when dealing with varying text sizes or when the default
       threshold produces poor results. For example, `-wz 105` resolved issues with
       mixed text sizes in [#48](https://github.com/lmmx/page-dewarp/issues/48).
    """
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
    SHEAR_COST: desc(float, "Penalty against camera tilt (shear distortion).") = 0.0
    """Penalty against camera tilt (shear distortion).

    Adds a penalty term to the optimization objective that discourages X-rotation,
    which manifests as sheared/slanted output.

    Tip:
       Increase if output appears sheared (parallelogram instead of rectangle).

    Note:
       The optimizer can mistake page curvature for camera tilt, producing sheared
       output even from flat scans. This penalty encourages modeling curvature via
       the cubic params instead of rotation.

    Warning:
       Using this at all may overcorrect, causing non-parallel sides. If edges look
       worse after enabling (typically only mildly), reduce the value.

    Question:
       [#83](https://github.com/lmmx/page-dewarp/issues/83): Discussion of shear
       distortion in flat document scans.
    """
    MAX_CORR: desc(
        int,
        "Maximum corrections used to approximate the inverse Hessian.",
    ) = 100
    """Maximum corrections used to approximate the inverse Hessian.

    Performance tuning parameter, only used in L-BFGS-B. For our problem in 600D, a high
    value (rather than the SciPy default of 10) is helpful to converge in fewer
    iterations.

    Question:
       [#135](https://github.com/lmmx/page-dewarp/pulls/135): performance measurements
       at SciPy default maxcorr = 10 and new page-dewarp configured default 100.

    See: [SciPy docs](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)

    See also: [Ceres docs](http://ceres-solver.org/nnls_solving.html), which describe
    the tradeoff of computation vs. quality of approximation.
    """
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
    """Index of cubic slopes in params vector (slice: pair of values).

    Note:
       These parameters control the cubic spline dewarping model. The cubic slopes
       determine how the page curvature is estimated.

    Todo:
       Document cubic param clamp control ([#67](https://github.com/lmmx/page-dewarp/issues/67))
    """
    # [span_opts]
    SPAN_MIN_WIDTH: desc(int, "Minimum reduced px width for span") = 30
    """Minimum reduced px width for span."""
    SPAN_PX_PER_STEP: desc(int, "Reduced px spacing for sampling along spans") = 20
    """Reduced px spacing for sampling along spans."""


cfg = Config()
