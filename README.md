# page-dewarp

Document image dewarping library using a cubic sheet model

Python 3 library for page dewarping and thresholding,
[available on PyPI](https://pypi.org/project/page_dewarp/).

## Requirements

Python 3 and NumPy, SciPy, SymPy, Matplotlib, OpenCV, and TOML Kit are required to run `page-dewarp`.

- See [`CONDA_SETUP.md`](https://github.com/lmmx/page-dewarp/blob/master/CONDA_SETUP.md) for
  an example conda environment 
- If you prefer to install everything from pip, run `pip install page-dewarp`

This library was renovated from the [original (2016) Python 2 script](https://github.com/mzucker/page_dewarp/)
by Matt Zucker, which you can use if you are somehow still using Python 2.

## Usage

```
usage: page-dewarp [-h] [-d {0,1,2,3}] [-o {file,screen,both}] [-p]
                   [-vw SCREEN_MAX_W] [-vh SCREEN_MAX_H] [-x PAGE_MARGIN_X]
                   [-y PAGE_MARGIN_Y] [-tw TEXT_MIN_WIDTH] [-th TEXT_MIN_HEIGHT]
                   [-ta TEXT_MIN_ASPECT] [-tk TEXT_MAX_THICKNESS]
                   [-wz ADAPTIVE_WINSZ] [-ri RVEC_IDX] [-ti TVEC_IDX]
                   [-ci CUBIC_IDX] [-sw SPAN_MIN_WIDTH] [-sp SPAN_PX_PER_STEP]
                   [-eo EDGE_MAX_OVERLAP] [-el EDGE_MAX_LENGTH]
                   [-ec EDGE_ANGLE_COST] [-ea EDGE_MAX_ANGLE] [-f FOCAL_LENGTH]
                   [-z OUTPUT_ZOOM] [-dpi OUTPUT_DPI] [-s REMAP_DECIMATE]
                   IMAGE_FILE_OR_FILES [IMAGE_FILE_OR_FILES ...]

positional arguments:
  IMAGE_FILE_OR_FILES   One or more images to process

optional arguments:
  -h, --help            show this help message and exit
  -d {0,1,2,3}, --debug-level {0,1,2,3}
  -o {file,screen,both}, --debug-output {file,screen,both}
  -p, --pdf             Merge dewarped images into a PDF
  -vw SCREEN_MAX_W, --max-screen-width SCREEN_MAX_W
                        Viewing screen max width (for resizing to screen)
  -vh SCREEN_MAX_H, --max-screen-height SCREEN_MAX_H
                        Viewing screen max height (for resizing to screen)
  -x PAGE_MARGIN_X, --x-margin PAGE_MARGIN_X
                        Reduced px to ignore near L/R edge
  -y PAGE_MARGIN_Y, --y-margin PAGE_MARGIN_Y
                        Reduced px to ignore near T/B edge
  -tw TEXT_MIN_WIDTH, --min-text-width TEXT_MIN_WIDTH
                        Min reduced px width of detected text contour
  -th TEXT_MIN_HEIGHT, --min-text-height TEXT_MIN_HEIGHT
                        Min reduced px height of detected text contour
  -ta TEXT_MIN_ASPECT, --min-text-aspect TEXT_MIN_ASPECT
                        Filter out text contours below this w/h ratio
  -tk TEXT_MAX_THICKNESS, --max-text-thickness TEXT_MAX_THICKNESS
                        Max reduced px thickness of detected text contour
  -wz ADAPTIVE_WINSZ, --adaptive-winsz ADAPTIVE_WINSZ
                        Window size for adaptive threshold in reduced px
  -ri RVEC_IDX, --rotation-vec-param-idx RVEC_IDX
                        Index of rvec in params vector (slice: pair of values)
  -ti TVEC_IDX, --translation-vec-param-idx TVEC_IDX
                        Index of tvec in params vector (slice: pair of values)
  -ci CUBIC_IDX, --cubic-slope-param-idx CUBIC_IDX
                        Index of cubic slopes in params vector (slice: pair of
                        values)
  -sw SPAN_MIN_WIDTH, --min-span-width SPAN_MIN_WIDTH
                        Minimum reduced px width for span
  -sp SPAN_PX_PER_STEP, --span-spacing SPAN_PX_PER_STEP
                        Reduced px spacing for sampling along spans
  -eo EDGE_MAX_OVERLAP, --max-edge-overlap EDGE_MAX_OVERLAP
                        Max reduced px horiz. overlap of contours in span
  -el EDGE_MAX_LENGTH, --max-edge-length EDGE_MAX_LENGTH
                        Max reduced px length of edge connecting contours
  -ec EDGE_ANGLE_COST, --edge-angle-cost EDGE_ANGLE_COST
                        Cost of angles in edges (tradeoff vs. length)
  -ea EDGE_MAX_ANGLE, --max-edge-angle EDGE_MAX_ANGLE
                        Maximum change in angle allowed between contours
  -f FOCAL_LENGTH, --focal-length FOCAL_LENGTH
                        Normalized focal length of camera
  -z OUTPUT_ZOOM, --output-zoom OUTPUT_ZOOM
                        How much to zoom output relative to *original* image
  -dpi OUTPUT_DPI, --output-dpi OUTPUT_DPI
                        Just affects stated DPI of PNG, not appearance
  -s REMAP_DECIMATE, --shrink REMAP_DECIMATE
                        Downscaling factor for remapping image
```

- PDF conversion not yet implemented

To try out an example image, run

```sh
git clone https://github.com/lmmx/page-dewarp
cd page-dewarp
mkdir results && cd results
page-dewarp ../example_input/boston_cooking_a.jpg
```

## Explanation and extension to Gpufit

A book on a flat surface can be said to be 'fixed to zero' at the endpoints of a curve, which
you can model as a cubic (see [`derive_cubic.py`](derive_cubic.py))

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
  - [ ] Multiprocessing on CPU
  - [ ] Optional interface to use Gpufit on GPU (or Deep Declarative Networks?)

