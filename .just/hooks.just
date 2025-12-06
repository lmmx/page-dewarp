ci_opt := if env("PRE_COMMIT_HOME", "") != "" { "-ci" } else { "" }
pc_cmd := if env("PRE_COMMIT_HOME", "") != "" { "pre-commit" } else { "prek" }

precommit:
    just pc{{ci_opt}}

pc:     fmt code-quality
pc-fix: fmt code-quality-fix
pc-ci:  pc-fix

fmt:     code-quality-fix

full:    pc test
full-ci: pc-ci

# -------------------------------------

install-hooks:
   {{pc_cmd}} install

run-pc:
   {{pc_cmd}} run --all-files

# -------------------------------------

fix-eof-ws mode="":
    #!/usr/bin/env bash
    ARGS=''
    if [ "{{mode}}" = "check" ]; then
        ARGS="--check-only"
    fi
    whitespace-format --add-new-line-marker-at-end-of-file \
          --new-line-marker=linux \
          --normalize-new-line-markers \
          --exclude ".git/|target/|tests/fixtures/|snapshots/|dist/|\.diff|\.png$|\.jpg$|\.sw[op]$|\.so$|.json$|.lock$|.snap$|.parquet$|.venv/|.stubs/|\..*cache/" \
          $ARGS \
          .

code-quality:
    # just ty-ci
    taplo lint
    taplo format --check
    just fix-eof-ws check

code-quality-fix:
    taplo lint
    taplo format
    just fix-eof-ws
