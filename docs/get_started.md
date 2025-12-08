---
title: "Get Started"
icon: material/human-greeting
---

# Getting Started

## 1. Installation

`page-dewarp` is [on PyPI](https://pypi.org/project/page-dewarp), the Python Package Index. Install with

```bash
pip install page-dewarp
```

!!! info "Use `uv` for the best developer experience"

    If you set up [uv](https://docs.astral.sh/uv/getting-started/installation/)
    (recommended) you can install with `uv pip install page-dewarp`
    or set up a project (e.g. with `uv init --app --package`, `uv venv`, then activate the venv
    following the instructions it gives) you can add it with `uv add page-dewarp`.

## 2. Usage

To run `page-dewarp` on a sample image:

page-dewarp input.jpg

This produces a `input_thresh.png` file with a thresholded and dewarped image. If you have multiple
images:

```bash
page-dewarp image1.jpg image2.jpg
```

That creates `image1_thresh.png` and `image2_thresh.png`. For more advanced details,
see the [API Reference](api/index.md).

## 3. Local Development

- **Set up dev environment**:
    1. Clone the repo: `git clone https://github.com/lmmx/page-dewarp.git`
    2. Install dev dependencies (e.g. via `requirements-dev.txt` or PDM).
    3. Optionally run `pre-commit install` to enable lint checks before every commit.

- **Test**:
  Run `pytest` (or `pdm run pytest`) to confirm everything works.

- **Build docs**:
  `mkdocs build` or `mkdocs serve` to view them locally, then `mkdocs gh-deploy` for GitHub Pages.

## 4. Example Workflow

1. Place one or more `.jpg` or `.png` files in a directory.
2. Run `page-dewarp myscan.jpg`.
3. A `_thresh.png` file is generated with the warped page corrected.

## 5. Configuration

`page-dewarp` uses a global config object (`cfg` in `options/core.py`) for parameters like
`DEBUG_LEVEL`, `FOCAL_LENGTH`, or `REMAP_DECIMATE`.

You can override them via CLI flags or by editing `cfg`.

See [CLI Usage](api/options.md) for argument specifics.
