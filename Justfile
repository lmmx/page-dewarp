# Just is a task runner, like Make but without the build system / dependency tracking part.
# docs: https://github.com/casey/just
#
# The `-ci` variants are ran in CI, they do command grouping on GitHub Actions, set consistent env vars etc.,
# but they require bash.
#
# The non`-ci` variants can be run locally without having bash installed.

set dotenv-load

default: precommit prepush

precommit: code-quality
prepush: ruff-check
precommit-fix: code-quality-fix

commit-msg message:
  printf "{{ message }}" | conventional_commits_linter --from-stdin --allow-angular-type-only

ci: precommit prepush docs

ruff-check:
    ruff check \
      --exclude ".git/|.venv/|site/|.pdm-build|target/|.json$|.lock$" \
      .

test *args:
    pytest {{args}} < /dev/null

test-ci *args:
    #!/usr/bin/env -S bash -euo pipefail
    source .envrc
    echo -e "\033[1;33m🏃 Running all but doc-tests with pytest...\033[0m"
    cmd_group "pytest {{args}} < /dev/null"

fix-eof-ws mode="":
    #!/usr/bin/env sh
    ARGS=''
    if [ "{{mode}}" = "check" ]; then
        ARGS="--check-only"
    fi
    whitespace-format --add-new-line-marker-at-end-of-file \
          --new-line-marker=linux \
          --normalize-new-line-markers \
          --exclude ".git/|.*cache/|.venv/|site/|.pdm-build/|.json$|.lock|.sw[op]|.png|.jpg$" \
          $ARGS \
          .

code-quality:
    taplo lint
    taplo format --check
    just fix-eof-ws check

code-quality-fix:
    taplo lint
    taplo format
    just fix-eof-ws

docs:
    mkdocs build

lockfile:
    cargo update --workspace --locked
