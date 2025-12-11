# page-dewarp

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
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

```sh
uv pip install page-dewarp
```

### JAX

To install with JAX autodiff for ~11x faster optimisation on single images and ~33x faster for batches (CPU only), add the `jax` extra:

```sh
uv pip install page-dewarp[jax]
```

#### GPU support

To install with support for GPU execution instead of only CPU, choose one of:

```sh
uv pip install page-dewarp[jax-cuda12] # CUDA 12
uv pip install page-dewarp[jax-cuda13] # CUDA 13 (requires Python 3.11+)
```

> **Note**: CPU execution is the default `DEVICE` and can be faster than GPU for this workload, but this may vary
> depending on your relative CPU/GPU horsepower (cores, RAM, VRAM, etc.)

## Serial vs Batch

When the JAX backend is available, default behaviour when given multiple images is to use batch
mode. Performance benchmark on 40 images (via [#139](https://github.com/lmmx/page-dewarp/pull/139)):

| **Device** | **Serial** | **Batch** | **Speedup** |
|--------|--------|--------|--------|
| CPU | **36s** | **8.7s** | 4.1x |
| GPU | 53s | 11.2s | **4.7x**  |

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
usage: page-dewarp [-h] [-d {0,1,2,3}] [-o {file,screen,both}]
                   [-it OPT_MAX_ITER] [-m OPT_METHOD] [-dev DEVICE]
                   [-b USE_BATCH] [-vw SCREEN_MAX_W] [-vh SCREEN_MAX_H]
                   [-x PAGE_MARGIN_X] [-y PAGE_MARGIN_Y] [-tw TEXT_MIN_WIDTH]
                   [-th TEXT_MIN_HEIGHT] [-ta TEXT_MIN_ASPECT]
                   [-tk TEXT_MAX_THICKNESS] [-wz ADAPTIVE_WINSZ]
                   [-ri RVEC_IDX] [-ti TVEC_IDX] [-ci CUBIC_IDX]
                   [-sw SPAN_MIN_WIDTH] [-sp SPAN_PX_PER_STEP]
                   [-eo EDGE_MAX_OVERLAP] [-el EDGE_MAX_LENGTH]
                   [-ec EDGE_ANGLE_COST] [-ea EDGE_MAX_ANGLE]
                   [-f FOCAL_LENGTH] [-z OUTPUT_ZOOM] [-dpi OUTPUT_DPI]
                   [-nb NO_BINARY] [-sh SHEAR_COST] [-mc MAX_CORR]
                   [-s REMAP_DECIMATE]
                   IMAGE_FILE_OR_FILES [IMAGE_FILE_OR_FILES ...]

positional arguments:
  IMAGE_FILE_OR_FILES   One or more images to process

options:
  -h, --help            show this help message and exit
  -d {0,1,2,3}, --debug-level {0,1,2,3}
                        (type: int, default: 0)
  -o {file,screen,both}, --debug-output {file,screen,both}
                        (type: str, default: file)
  -it OPT_MAX_ITER, --max-iter OPT_MAX_ITER
                        Maximum optimisation iterations (type: int, default:
                        600000)
  -m OPT_METHOD, --method OPT_METHOD
                        Name of the JAX/SciPy optimisation method to use.
                        (type: str, default: auto)
  -dev DEVICE, --device DEVICE
                        Compute device to select for optimisation. (type: str,
                        default: auto)
  -b USE_BATCH, --batch USE_BATCH
                        Whether to batch process images (JAX backend only).
                        (type: str, default: auto)
  -vw SCREEN_MAX_W, --max-screen-width SCREEN_MAX_W
                        Viewing screen max width (for resizing to screen)
                        (type: int, default: 1280)
  -vh SCREEN_MAX_H, --max-screen-height SCREEN_MAX_H
                        Viewing screen max height (for resizing to screen)
                        (type: int, default: 700)
  -x PAGE_MARGIN_X, --x-margin PAGE_MARGIN_X
                        Reduced px to ignore near L/R edge (type: int,
                        default: 50)
  -y PAGE_MARGIN_Y, --y-margin PAGE_MARGIN_Y
                        Reduced px to ignore near T/B edge (type: int,
                        default: 20)
  -tw TEXT_MIN_WIDTH, --min-text-width TEXT_MIN_WIDTH
                        Min reduced px width of detected text contour (type:
                        int, default: 15)
  -th TEXT_MIN_HEIGHT, --min-text-height TEXT_MIN_HEIGHT
                        Min reduced px height of detected text contour (type:
                        int, default: 2)
  -ta TEXT_MIN_ASPECT, --min-text-aspect TEXT_MIN_ASPECT
                        Filter out text contours below this w/h ratio (type:
                        float, default: 1.5)
  -tk TEXT_MAX_THICKNESS, --max-text-thickness TEXT_MAX_THICKNESS
                        Max reduced px thickness of detected text contour
                        (type: int, default: 10)
  -wz ADAPTIVE_WINSZ, --adaptive-winsz ADAPTIVE_WINSZ
                        Window size for adaptive threshold in reduced px
                        (type: int, default: 55)
  -ri RVEC_IDX, --rotation-vec-param-idx RVEC_IDX
                        Index of rvec in params vector (slice: pair of values)
                        (type: tuple[int, int], default: (0, 3))
  -ti TVEC_IDX, --translation-vec-param-idx TVEC_IDX
                        Index of tvec in params vector (slice: pair of values)
                        (type: tuple[int, int], default: (3, 6))
  -ci CUBIC_IDX, --cubic-slope-param-idx CUBIC_IDX
                        Index of cubic slopes in params vector (slice: pair of
                        values) (type: tuple[int, int], default: (6, 8))
  -sw SPAN_MIN_WIDTH, --min-span-width SPAN_MIN_WIDTH
                        Minimum reduced px width for span (type: int, default:
                        30)
  -sp SPAN_PX_PER_STEP, --span-spacing SPAN_PX_PER_STEP
                        Reduced px spacing for sampling along spans (type:
                        int, default: 20)
  -eo EDGE_MAX_OVERLAP, --max-edge-overlap EDGE_MAX_OVERLAP
                        Max reduced px horiz. overlap of contours in span
                        (type: float, default: 1.0)
  -el EDGE_MAX_LENGTH, --max-edge-length EDGE_MAX_LENGTH
                        Max reduced px length of edge connecting contours
                        (type: float, default: 100.0)
  -ec EDGE_ANGLE_COST, --edge-angle-cost EDGE_ANGLE_COST
                        Cost of angles in edges (tradeoff vs. length) (type:
                        float, default: 10.0)
  -ea EDGE_MAX_ANGLE, --max-edge-angle EDGE_MAX_ANGLE
                        Maximum change in angle allowed between contours
                        (type: float, default: 7.5)
  -f FOCAL_LENGTH, --focal-length FOCAL_LENGTH
                        Normalized focal length of camera (type: float,
                        default: 1.2)
  -z OUTPUT_ZOOM, --output-zoom OUTPUT_ZOOM
                        How much to zoom output relative to *original* image
                        (type: float, default: 1.0)
  -dpi OUTPUT_DPI, --output-dpi OUTPUT_DPI
                        Just affects stated DPI of PNG, not appearance (type:
                        int, default: 300)
  -nb NO_BINARY, --no-binary NO_BINARY
                        Disable output conversion to binary thresholded image
                        (type: int, default: 0)
  -sh SHEAR_COST, --shear-cost SHEAR_COST
                        Penalty against camera tilt (shear distortion). (type:
                        float, default: 0.0)
  -mc MAX_CORR, --max-corrections MAX_CORR
                        Maximum corrections used to approximate the inverse
                        Hessian. (type: int, default: 100)
  -s REMAP_DECIMATE, --shrink REMAP_DECIMATE
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

## Explanation and further reading

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
- [x] Speed up the optimisation
    - [x] Limit optimisation iterations (via `-it` flag)
    - [x] Optional GPU interface (via `-dev` flag)
    - [ ] Multiprocessing on CPU
