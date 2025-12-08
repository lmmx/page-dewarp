"""Page-dewarp: a document image dewarping library using a cubic sheet model.

This package offers functionality for:
- Dewarping scanned pages (using a cubic sheet model).
- Thresholding to produce clear, high-contrast page images.

It is a refactored version of Matt Zucker's original (2016) Python 2 script,
updated for Python 3.9+ and packaged for easy installation from PyPI.

Dependencies include NumPy, SciPy, SymPy, Matplotlib, OpenCV, and msgspec.
See the project's README for a complete list of installation and usage details.

References and further reading:
- Matt Zucker’s original project: https://github.com/mzucker/page_dewarp/
- Write-up on cubic sheet dewarping: https://mzucker.github.io/2016/08/15/page-dewarping.html
- Additional notes and advanced examples: https://doc.spin.systems/page-dewarp

Package exports:
- `main` (from `. __main__`): The CLI entry point for dewarping images.
"""

from .__main__ import main


__all__ = ("main",)
