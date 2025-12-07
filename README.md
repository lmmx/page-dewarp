# page-dewarp

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)
[![PyPI](https://img.shields.io/pypi/v/page-dewarp.svg)](https://pypi.org/project/page-dewarp)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/page-dewarp.svg)](https://pypi.org/project/page-dewarp)
[![downloads](https://static.pepy.tech/badge/page-dewarp/month)](https://pepy.tech/project/page-dewarp)
[![License](https://img.shields.io/pypi/l/page-dewarp.svg)](https://pypi.python.org/pypi/page-dewarp)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/lmmx/page-dewarp/master.svg)](https://results.pre-commit.ci/latest/github/lmmx/page-dewarp/master)

Document image dewarping library using a cubic sheet model

Python 3 library for page dewarping and thresholding,
[available on PyPI](https://pypi.org/project/page_dewarp/).

## Installation

To install from PyPI, optionally using [uv](https://docs.astral.sh/uv/) (recommended), run:

- `pip install page-dewarp`
- or `uv pip install page-dewarp` (recommended)

## Dependencies

Python 3.10+ and NumPy, SciPy, SymPy, Matplotlib, OpenCV, and msgspec are required to run `page-dewarp`.

## Help

See [documentation](https://page-dewarp.vercel.app) for more details.

- **Update!**: the docs now have a [How It Works](https://page-dewarp.vercel.app/how-it-works/Introduction/) section

## Background

This library was renovated from the [original (2016) Python 2 script](https://github.com/mzucker/page_dewarp/)
by Matt Zucker, as Python 2 is now long since decommissioned.

## Usage

- See [config docs](https://page-dewarp.vercel.app/api/options) for a table of options

```
usage: page-dewarp [-h] [-d {0,1,2,3}] [-o {file,screen,both}] [-p]
                   [-it OPT_MAX_ITER] [-vw SCREEN_MAX_W] [-vh SCREEN_MAX_H]
                   [-x PAGE_MARGIN_X] [-y PAGE_MARGIN_Y] [-tw TEXT_MIN_WIDTH]
                   [-th TEXT_MIN_HEIGHT] [-ta TEXT_MIN_ASPECT]
                   [-tk TEXT_MAX_THICKNESS] [-wz ADAPTIVE_WINSZ]
                   [-ri RVEC_IDX] [-ti TVEC_IDX] [-ci CUBIC_IDX]
                   [-sw SPAN_MIN_WIDTH] [-sp SPAN_PX_PER_STEP]
                   [-eo EDGE_MAX_OVERLAP] [-el EDGE_MAX_LENGTH]
                   [-ec EDGE_ANGLE_COST] [-ea EDGE_MAX_ANGLE]
                   [-f FOCAL_LENGTH] [-z OUTPUT_ZOOM] [-dpi OUTPUT_DPI]
                   [-nb NO_BINARY] [-s REMAP_DECIMATE]
                   IMAGE_FILE_OR_FILES [IMAGE_FILE_OR_FILES ...]

positional arguments:
  IMAGE_FILE_OR_FILES   One or more images to process

options:
  -h, --help            show this help message and exit
  -d, --debug-level {0,1,2,3}
                        (type: int, default: 0)
  -o, --debug-output {file,screen,both}
                        (type: str, default: file)
  -p, --pdf             Merge images into a PDF
  -it, --max-iter OPT_MAX_ITER
                        Maximum Powell's method optimisation iterations (type:
                        int, default: 600000)
  -vw, --max-screen-width SCREEN_MAX_W
                        Viewing screen max width (for resizing to screen)
                        (type: int, default: 1280)
  -vh, --max-screen-height SCREEN_MAX_H
                        Viewing screen max height (for resizing to screen)
                        (type: int, default: 700)
  -x, --x-margin PAGE_MARGIN_X
                        Reduced px to ignore near L/R edge (type: int,
                        default: 50)
  -y, --y-margin PAGE_MARGIN_Y
                        Reduced px to ignore near T/B edge (type: int,
                        default: 20)
  -tw, --min-text-width TEXT_MIN_WIDTH
                        Min reduced px width of detected text contour (type:
                        int, default: 15)
  -th, --min-text-height TEXT_MIN_HEIGHT
                        Min reduced px height of detected text contour (type:
                        int, default: 2)
  -ta, --min-text-aspect TEXT_MIN_ASPECT
                        Filter out text contours below this w/h ratio (type:
                        float, default: 1.5)
  -tk, --max-text-thickness TEXT_MAX_THICKNESS
                        Max reduced px thickness of detected text contour
                        (type: int, default: 10)
  -wz, --adaptive-winsz ADAPTIVE_WINSZ
                        Window size for adaptive threshold in reduced px
                        (type: int, default: 55)
  -ri, --rotation-vec-param-idx RVEC_IDX
                        Index of rvec in params vector (slice: pair of values)
                        (type: tuple[int, int], default: (0, 3))
  -ti, --translation-vec-param-idx TVEC_IDX
                        Index of tvec in params vector (slice: pair of values)
                        (type: tuple[int, int], default: (3, 6))
  -ci, --cubic-slope-param-idx CUBIC_IDX
                        Index of cubic slopes in params vector (slice: pair of
                        values) (type: tuple[int, int], default: (6, 8))
  -sw, --min-span-width SPAN_MIN_WIDTH
                        Minimum reduced px width for span (type: int, default:
                        30)
  -sp, --span-spacing SPAN_PX_PER_STEP
                        Reduced px spacing for sampling along spans (type:
                        int, default: 20)
  -eo, --max-edge-overlap EDGE_MAX_OVERLAP
                        Max reduced px horiz. overlap of contours in span
                        (type: float, default: 1.0)
  -el, --max-edge-length EDGE_MAX_LENGTH
                        Max reduced px length of edge connecting contours
                        (type: float, default: 100.0)
  -ec, --edge-angle-cost EDGE_ANGLE_COST
                        Cost of angles in edges (tradeoff vs. length) (type:
                        float, default: 10.0)
  -ea, --max-edge-angle EDGE_MAX_ANGLE
                        Maximum change in angle allowed between contours
                        (type: float, default: 7.5)
  -f, --focal-length FOCAL_LENGTH
                        Normalized focal length of camera (type: float,
                        default: 1.2)
  -z, --output-zoom OUTPUT_ZOOM
                        How much to zoom output relative to *original* image
                        (type: float, default: 1.0)
  -dpi, --output-dpi OUTPUT_DPI
                        Just affects stated DPI of PNG, not appearance (type:
                        int, default: 300)
  -nb, --no-binary NO_BINARY
                        Disable output conversion to binary thresholded image
                        (type: int, default: 0)
  -s, --shrink REMAP_DECIMATE
                        Downscaling factor for remapping image (type: int,
                        default: 16)
```

To try out an example image, run

```sh
git clone https://github.com/lmmx/page-dewarp
cd page-dewarp
mkdir results && cd results
page-dewarp ../example_input/boston_cooking_a.jpg
```

## Explanation and extension to Gpufit

A book on a flat surface can be said to be 'fixed to zero' at the endpoints of a curve, which
you can model as a cubic (see
[`derive_cubic.py`](https://github.com/lmmx/page-dewarp/blob/master/derive_cubic.py))

The "cubic Hermite spline" is one of the models supported by
[Gpufit](https://github.com/gpufit/Gpufit/), a library for Levenberg Marquardt curve fitting in
CUDA (C++ with Python API).

_[Work in progress]_

- See full writeup on [Matt Zucker's blog](https://mzucker.github.io/2016/08/15/page-dewarping.html)
- See [lecture](https://www.cs.cornell.edu/courses/cs4620/2013fa/lectures/16spline-curves.pdf)
  on splines by Steve Marschner for more details and how a spline can be represented in matrix form.
- Brief notes on this project are over on [my website](https://doc.spin.systems/page-dewarp)

## Features

Improvements on the original include:

- [x] Banished Python 2
- [x] Command line interface
    - [x] Alterable config options
- [x] Repackage for pip installation
- [x] Refactor with modules and classes
- [ ] Speed up the optimisation
    - [x] Limit optimisation iterations (via `-it` flag)
    - [ ] Multiprocessing on CPU
    - [ ] Optional interface to use Gpufit on GPU (or Deep Declarative Networks?)
