# src/page_dewarp/backends.py
"""Backend availability detection."""

__all__ = ["HAS_JAX", "HAS_SCIPY"]

HAS_JAX = False
try:
    import jax  # noqa: F401

    HAS_JAX = True
except ImportError:
    pass

HAS_SCIPY = False
try:
    import scipy  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    pass

if not HAS_JAX and not HAS_SCIPY:
    raise ImportError(
        "page-dewarp requires at least one optimization backend. "
        "Install scipy (required) or jax (optional, faster).",
    )
