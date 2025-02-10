#!/bin/bash
set -ex

# 1) Reactivate the same uv-managed venv.
#    (On Vercel, your workspace is ephemeral between these steps,
#     but re-sourcing ensures we’re in the same environment as deployed.)
source .venv/bin/activate

# 2) Double check that we have the correct interpreter.
python --version

# 3) Build the docs site with MkDocs:
mkdocs build
