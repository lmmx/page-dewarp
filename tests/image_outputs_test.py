"""Integration test for page-dewarp CLI outputs.

This module tests that the `page-dewarp` command produces the expected
thresholded image output for a sample input.
"""

import subprocess
from pathlib import Path

import pytest
from czkawka import ImageSimilarity


repo_root = Path(__file__).parents[1]
example_inputs_dir = repo_root / "example_input"
example_outputs_dir = repo_root / "example_output"


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for storing test outputs."""
    return tmp_path


def test_page_dewarp_output(temp_dir):
    """Check that the CLI produces the expected thresholded image from sample input."""
    input_file = example_inputs_dir / "boston_cooking_a.jpg"
    output_file = temp_dir / "boston_cooking_a_thresh.png"

    subprocess.run(["page-dewarp", str(input_file)], cwd=temp_dir, check=True)

    assert output_file.exists(), "Output file was not created"

    finder = ImageSimilarity()
    expected_hash = "8KT0pOS0tLQ"
    output_hash = finder.hash_image(output_file)
    distance = finder.compare_hashes(output_hash, expected_hash)

    assert distance <= 5, (
        f"Output image too different: distance={distance}, hash={output_hash}"
    )
