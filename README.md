# page-dewarp

Document image dewarping library using a cubic sheet model

Python 3 library for page dewarping and thresholding,
[available on PyPI](https://pypi.org/project/page-dewarp/).

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

## Features

Improvements on the original include:

- [x] Banished Python 2
- [ ] Command line interface
  - Both batched mode and input directory input mode
  - Alterable debug level
  - ...
- [x] Repackage for pip installation
- [ ] Refactor with modules and classes
- [ ] Speed up the optimisation
  - Multiprocessing on CPU
  - Optional interface to use Gpufit on GPU


## Requirements

Python 3 and NumPy, SciPy, SymPy, Matplotlib and OpenCV are required to run `page-dewarp`.

- See [`CONDA_SETUP.md`](https://github.com/lmmx/page-dewarp/blob/master/CONDA_SETUP.md) for
  an example conda environment 
- If you must install everything from pip, `pip install` will retrieve
  [`requirements.txt`](https://github.com/lmmx/page-dewarp/blob/master/requirements.txt)

This library was renovated from the [original (2016) Python 2 script](https://github.com/mzucker/page_dewarp/)
by Matt Zucker, which you can use if you are somehow still using Python 2.

## Usage

```
TBC
```
