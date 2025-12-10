"""Backend availability detection."""

__all__ = ["HAS_JAX", "HAS_SCIPY", "HAS_JAXOPT", "get_default_method"]

HAS_JAX = False
try:
    import jax  # noqa: F401

    HAS_JAX = True
except ImportError:
    pass

HAS_JAXOPT = False
if HAS_JAX:
    try:
        import jaxopt  # noqa: F401

        HAS_JAXOPT = True
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
    if HAS_JAXOPT:
        return "LevenbergMarquardt"
    elif HAS_JAX:
        return "L-BFGS-B"
    else:
        return "Powell"
