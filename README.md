# page-dewarp

Document image dewarping library using a cubic sheet model

Python 3 library renovated from the
[original (2016) Python 2 script](https://github.com/mzucker/page_dewarp/) by Matt Zucker
for page dewarping and thresholding, now with more advanced command line interface and modular package organisation,
and [distributed on PyPI](https://pypi.org/project/page-dewarp/).

- See full writeup on [Matt's blog](https://mzucker.github.io/2016/08/15/page-dewarping.html)

## Explanation and extension to Gpufit

A book on a flat surface can be said to be 'fixed to zero' at the endpoints of a curve, which
you can model as a cubic (see [`derive_cubic.py`](derive_cubic.py))

The "cubic spline" is one of the models supported by
[Gpufit](https://github.com/gpufit/Gpufit/), a library for Levenberg Marquardt curve fitting in
CUDA (C++ with Python API).

- See [lecture](https://www.cs.cornell.edu/courses/cs4620/2013fa/lectures/16spline-curves.pdf)
  on splines for more details and how a spline can be represented in matrix form.

## Features

Improvements on the original include:

- [x] Banished Python 2
- [ ] Add both batched mode and input directory input mode (plus other command line flags)
- [x] Repackage for pip installation
- [ ] Refactor with modules and classes
- [ ] Speed up the optimisation


## Requirements

Currently matching those in [@bertsky's Python 3 fork](https://github.com/bertsky/page_dewarp/tree/support-python3)
of [@mzucker's `page_dewarp` repo](https://github.com/mzucker/page_dewarp/):

- Python 3
- scipy
- OpenCV 3.0 or greater
- Image module from PIL or Pillow

## Usage

```
TBC
```
