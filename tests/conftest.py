"""Fixtures for CI detection."""

import os

import pytest


@pytest.fixture
def IS_CI():
    """Fixture to determine if the code is running in a CI environment."""
    return bool(os.environ.get("CI", None))
