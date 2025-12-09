"""Fixtures for testing."""

import os
import sys

import pytest


if sys.version_info[:2] == (3, 12):
    assert False


@pytest.fixture
def IS_CI():
    """Fixture to determine if running in CI."""
    return bool(os.environ.get("CI"))
