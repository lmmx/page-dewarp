#!/bin/bash

yum install wget

wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

./bin/micromamba shell init -s bash -p ~/micromamba
# Python interpreter lives at /vercel/micromamba/bin/python
source ~/.bashrc

# activate the environment and install a new version of Python
micromamba activate
micromamba install python=3.11 -c conda-forge -y

# install the dependencies
python --version
python -m pip install pdm 'urllib3<2'
# pdm install -dG docs -v
pdm install --no-default -dG docs -v
pdm run mkdocs
