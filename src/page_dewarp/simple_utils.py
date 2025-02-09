"""Simple utility functions used throughout page_dewarp.

Currently provides a small helper (`fltp`) to flatten integer coordinates.
"""

__all__ = ["fltp"]


def fltp(point):
    """Flatten and convert a NumPy coordinate to an (x, y) integer tuple.

    Args:
        point: A NumPy array containing [x, y] coordinates (possibly float).

    Returns:
        An (x, y) tuple of integers.

    """
    return tuple(point.astype(int).flatten())
