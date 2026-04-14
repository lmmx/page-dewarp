"""Logging configuration for page_dewarp.

This module provides centralized logging setup. Configure once at startup
via `configure_logging()`, then use `get_logger()` in each module.

Processing code logs unconditionally; verbosity is controlled by handler
configuration, not by conditional logic around log calls.
"""

import logging
import sys
from typing import TextIO


__all__ = ["configure_logging", "get_logger"]

LOGGER_NAME = "page_dewarp"


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger for the given module name.

    Args:
        name: Optional submodule name (e.g., "image", "optimise.jax").
              If None, returns the root package logger.

    Returns:
        A configured logger instance.

    """
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


def configure_logging(
    quiet: bool = False,
    verbose: bool = False,
    debug: bool = False,
    log_file: str | None = None,
    stream: TextIO | None = None,
) -> None:
    """Configure logging handlers based on CLI flags.

    This should be called once at startup, before any processing begins.
    Processing code simply logs; this function controls where output goes.

    Args:
        quiet: If True, suppress console logging except errors.
        verbose: If True, show INFO-level logs on console.
        debug: If True, show DEBUG-level logs on console.
        log_file: Optional path to write full debug logs.
        stream: Output stream for console handler (default: stderr).

    Flag precedence: debug > verbose > quiet > default.

    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)  # Let handlers do the filtering
    logger.handlers.clear()  # Avoid duplicate handlers on re-entry
    logger.propagate = False  # Don't bubble up to root logger

    if stream is None:
        stream = sys.stderr

    # Determine console log level
    if debug:
        console_level = logging.DEBUG
    elif verbose:
        console_level = logging.INFO
    elif quiet:
        console_level = logging.ERROR
    else:
        console_level = logging.WARNING

    # Console handler
    console = logging.StreamHandler(stream)
    console.setLevel(console_level)
    console.setFormatter(_make_console_formatter(debug))
    logger.addHandler(console)

    # File handler (always DEBUG level if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(_make_file_formatter())
        logger.addHandler(file_handler)


def _make_console_formatter(debug: bool) -> logging.Formatter:
    """Create a formatter for console output."""
    if debug:
        return logging.Formatter(
            "%(levelname)s [%(name)s] %(message)s",
        )
    return logging.Formatter("%(levelname)s: %(message)s")


def _make_file_formatter() -> logging.Formatter:
    """Create a formatter for file output."""
    return logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
