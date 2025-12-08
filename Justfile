import ".just/commit.just"
import ".just/hooks.just"
import ".just/ship.just"
import ".just/test.just"

[working-directory: "example_output"]
refresh-examples:
    #!/usr/bin/env bash
    ../.venv/bin/page-dewarp ../example_input/*
