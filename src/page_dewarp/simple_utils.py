__all__ = ["fltp"]


def fltp(point):
    return tuple(point.astype(int).flatten())
