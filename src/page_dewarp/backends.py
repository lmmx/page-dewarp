"""Backend availability detection."""

from __future__ import annotations


__all__ = ["HAS_JAX", "HAS_SCIPY", "get_default_method"]

HAS_JAX = False
try:
    import jax  # noqa: F401

    HAS_JAX = True

    # Enable float64 for numerical stability (must be set before JIT compilation)
    jax.config.update("jax_enable_x64", True)
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


def get_default_method() -> str:
    """Return the default optimization method based on available backend."""
    return "L-BFGS-B" if HAS_JAX else "Powell"
