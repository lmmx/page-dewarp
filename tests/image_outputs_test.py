"""Integration test for page-dewarp CLI outputs.

This module tests that the `page-dewarp` command produces the expected
thresholded image output for a sample input.
"""

import subprocess
from pathlib import Path

import pytest
from czkawka import ImageSimilarity

from page_dewarp.backends import HAS_JAX, HAS_SCIPY


repo_root = Path(__file__).parents[1]
example_inputs_dir = repo_root / "example_input"
example_outputs_dir = repo_root / "example_output"


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for storing test outputs."""
    return tmp_path


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_page_dewarp_output_jax(temp_dir, IS_CI):
    """Check that the CLI produces the expected thresholded image from sample input."""
    input_file = example_inputs_dir / "boston_cooking_a.jpg"
    output_file = temp_dir / "boston_cooking_a_thresh.png"

    cmd = ["page-dewarp", str(input_file)]
    subprocess.run(cmd, cwd=temp_dir, check=True)

    assert output_file.exists(), "Output file was not created"

    finder = ImageSimilarity()
    expected_hash = "8LT0tOSwnKw" if IS_CI else "8LT0tOSwnKw"
    output_hash = finder.hash_image(output_file)
    distance = finder.compare_hashes(output_hash, expected_hash)

    assert distance == 0, (
        f"Output image too different: distance={distance}, hash={output_hash}"
    )


@pytest.mark.skipif(
    HAS_JAX or not HAS_SCIPY,
    reason="JAX must not be installed, needs SciPy only",
)
def test_page_dewarp_output_scipy(temp_dir, IS_CI):
    """Check CLI output when using scipy backend (no JAX)."""
    input_file = example_inputs_dir / "boston_cooking_a.jpg"
    output_file = temp_dir / "boston_cooking_a_thresh.png"

    cmd = ["page-dewarp", str(input_file)]
    subprocess.run(cmd, cwd=temp_dir, check=True)

    assert output_file.exists(), "Output file was not created"

    finder = ImageSimilarity()
    expected_hash = "8LS0tOSwvLQ"
    output_hash = finder.hash_image(output_file)
    distance = finder.compare_hashes(output_hash, expected_hash)

    assert distance == 0, (
        f"Output image too different: distance={distance}, hash={output_hash}"
    )

    assert output_file.exists(), "Output file was not created"

    finder = ImageSimilarity()
    expected_hash = "8LS0tOSwvLQ"  # SciPy is deterministic across CPUs
    output_hash = finder.hash_image(output_file)
    distance = finder.compare_hashes(output_hash, expected_hash)

    assert distance == 0, (
        f"Output too different: distance={distance}, hash={output_hash}"
    )
