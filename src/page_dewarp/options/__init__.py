"""Option-handling and configuration definitions for page_dewarp.

This package provides:
- A global config instance (`cfg`) containing default parameters.
- The `Config` class, which defines the structure and types of these parameters.
"""

from .core import Config, cfg


__all__ = ("cfg", "Config")
