"""Fixtures for testing."""

import os

import pytest


@pytest.fixture
def IS_CI():
    """Fixture to determine if running in CI."""
    return bool(os.environ.get("CI"))
