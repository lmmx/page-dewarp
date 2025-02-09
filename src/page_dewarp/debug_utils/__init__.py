"""Utilities for debugging and visualization in page_dewarp.

This package contains modules for coloring contours (`colours.py`) and
displaying images (`viewer.py`) via debug hooks.
"""

from .colours import cCOLOURS
from .viewer import debug_show


__all__ = ["cCOLOURS", "debug_show"]
