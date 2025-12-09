"""Test backend detection and defaults."""

from page_dewarp.backends import HAS_JAX, HAS_SCIPY, get_default_method


def test_get_default_method_without_jax():
    """Verify scipy-only default is Powell."""
    if HAS_JAX:
        assert get_default_method() == "L-BFGS-B"
    else:
        assert get_default_method() == "Powell"


def test_scipy_always_available():
    """SciPy is a required dependency."""
    assert HAS_SCIPY
