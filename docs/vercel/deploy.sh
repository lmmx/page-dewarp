#!/bin/bash
set -euo pipefail

echo "INSTALLING UV"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "UV VERSION"
uv --version

echo "SYNCING DOCS + VERCEL DEPS"
uv sync --python 3.11 --frozen --no-default-groups --group docs
