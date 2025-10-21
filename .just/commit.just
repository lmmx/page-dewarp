# NO_STASH is a git stash `:(exclude)` pattern. We never stash the Justfile
NO_STASH := "Justfile"

# FIXER must be the name of a Justfile task
FIXER := "pc-fix"

check-fixer:
    #!/usr/bin/env echo-comment

    just -s {{FIXER}} >/dev/null 2>&1
    test $? -eq 0 || echo "# just: No such task '{{FIXER}}'" | echo-comment /dev/stdin --color=bold-red

no-fix-stash msg="pre-commit-temp":
    git stash push --keep-index --include-untracked -m "{{msg}}" -- \
        ':/*' \
        ':(exclude){{NO_STASH}}'

stash-restore *changes:
    #!/usr/bin/env -S echo-comment --color=bold-yellow

    # Restoring {{changes}}
    git stash pop --quiet


# Special commit flag that will run a given command
commit msg *args:
    #!/usr/bin/env -S echo-comment --color bold-yellow

    # Confirm fixer {{FIXER}} is a valid Justfile task
    check-fixer

    # Halt at non-zero return codes from now on
    set -e

    # Count unstaged/untracked files
    unstaged_count=$(git diff --name-only -- . ':(exclude){{NO_STASH}}' | wc -l)
    untracked_count=$(git ls-files --others --exclude-standard -- ':/*' ':(exclude){{NO_STASH}}' | wc -l)
    stash_count=$((unstaged_count + untracked_count))

    # Files to stash: $stash_count
    if [ $stash_count -ne 0 ]; then
      case "$unstaged_count-$untracked_count" in
        0-*) kinds="untracked" ;;
        *-0) kinds="unstaged" ;;
        *-*) kinds="unstaged and untracked" ;;
      esac

      # Stashing $stash_count $kinds files
      just no-fix-stash

      ## Always restore later, even on failure
      trap 'just stash-restore $stash_count $kinds changes' EXIT
    fi

    # Autofix run before calling commit (which triggers pre-commit hooks)
    just {{FIXER}}

    # Count any files which now have unstaged changes
    fixer_unstaged_count=$(git diff --name-only -- . ':(exclude){{NO_STASH}}' | wc -l)

    if [ $fixer_unstaged_count -ne 0 ]; then
      s="${fixer_unstaged_count#1}"  # strip leading "1", empty if 1 → "", else → "s"
      # Fixer modified $fixer_unstaged_count file${s:+s}, staging...
      git add --all
    fi

    # Commit the changes
    git commit -m "{{msg}}" {{args}}
