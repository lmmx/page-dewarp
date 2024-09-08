#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <version>"
  exit 1
fi

version="$1"

# Remove the 'v' prefix if present
version="${version#v}"

towncrier build --version "$version"
