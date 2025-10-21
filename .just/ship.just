publish *args:
  $(uv python find) -m pdm publish -u __token__ -P $(keyring get PYPIRC_TOKEN "") {{args}}

# Release a new version, pass --help for options to `uv version --bump`
release bump_level="patch":
    #!/usr/bin/env -S echo-comment --shell-flags="-e" --color bright-green

    ## Exit early if help was requested
    if [[ "{{ bump_level }}" == "--help" ]]; then
        uv version --help
        exit 0
    fi

    # 📈 Bump the version in pyproject.toml (patch/minor/major: {{ bump_level }})
    uv version --bump {{ bump_level }}

    # 📦 Stage all changes (including the version bump)
    git add --all

    # 🔄 Create a temporary commit to capture the new version
    git commit -m "chore(temp): version check"

    # ✂️  Extract the new version number that was just set, undo the commit
    new_version=$(uv version --short)
    git reset --soft HEAD~1

    # ✅ Stage everything again and create the real release commit
    git add --all
    git commit -m  "chore(release): bump 🐍 -> v$new_version"

    # 🏷️ Create the git tag for this release
    git tag -a "py-$new_version" -m "Python Release $new_version"

    branch_name=$(git rev-parse --abbrev-ref HEAD);
    # 🚀 Push the release commit to $branch_name
    git push origin $branch_name

    # 🚀 Push the commit tag to the remote
    git push origin "py-$new_version"

ship-wheels *dl_args:
  mkdir -p dist
  rm -rf dist/*
  gh run download -p wheel* {{dl_args}}
  mv wheel*/* dist/
  rm -rf wheel*
  just publish --no-build
  rm -rf dist/*
