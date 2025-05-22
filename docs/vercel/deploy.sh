#!/bin/bash
set -ex

# 1) Ensure wget is available (Vercel's Amazon Linux images may not include it by default).
yum install -y wget

# 2) Download and install uv. This script typically places uv into ~/.local/bin/.
wget -qO- https://astral.sh/uv/install.sh | sh

find ./ -iname uv -type f 2> /dev/null

# 3) Make sure ~/.local/bin is on PATH so that 'uv' can be used directly.
export PATH="$HOME/.local/bin:$PATH"

# 4) Create a Python 3.11 venv using uv’s built-in venv command.
uv venv --python 3.11

# 5) Activate the venv. (Alternatively, you could use $(uv python find) but activating is simpler.)
source .venv/bin/activate

# 6) Check the Python version, just to confirm everything is correct.
python --version

# 7) Install dependencies:
#    - First pin urllib3<2 (to avoid known breakage).
#    - Then install your docs extra so that mkdocs & related are available.
uv pip install "urllib3<2"
uv pip install .[docs]

# 8) Optionally run mkdocs here if you need it immediately in “deploy”
#    (e.g., if your older script used ‘pdm run mkdocs’ at this point).
#    Otherwise, you can defer building to build.sh. For parity with your old deploy script:
uv run mkdocs --help && echo $?
