"""Python version enforcement for page_dewarp.

This module checks the running Python version, and exits if it is below
the minimum supported version (3.9).
"""

import sys


__all__ = ("enforce_version",)

MIN_SUPPORTED_V = (3, 9)


def enforce_version() -> None:
    """Raise `SystemExit` if running on an unsupported Python version."""
    if (ver := sys.version_info) < MIN_SUPPORTED_V:
        user_warn = f"Python {ver.major}.{ver.minor} is not supported"
        msg = f"{user_warn}: Please use Python {'.'.join(MIN_SUPPORTED_V)} or higher.\n"
        sys.stderr.write(msg)
        sys.exit(1)
    return
