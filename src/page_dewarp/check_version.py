import sys

__all__ = ("enforce_version",)

MIN_SUPPORTED_V = (3, 9)

def enforce_version() -> None:
    """Raise `SystemExit` if running on an unsupported Python version."""
    if sys.version_info < MIN_SUPPORTED_V:
        user_warn = f"Python {sys.version_info.major}.{sys.version_info.minor} is not supported"
        msg = f"{user_warn}: Please use Python {'.'.join(min_supported_v)} or higher.\n"
        sys.stderr.write(msg)
        sys.exit(1)
    return
