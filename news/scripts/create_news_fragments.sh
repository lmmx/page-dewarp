#!/bin/bash

# Ensure the news directory exists: do not proceed otherwise
root_hint="(Hint: run this script from the repo root)"
newsdir="news"
[ -d "$newsdir" ] || { echo "Error: The directory $newsdir does not exist. $root_hint" >&2; exit 1; }
# Ensure the fragments directory does not exist: do not assume it can be deleted (use PDM erase-history script)
frags="$newsdir/fragments"
[ ! -d "$frags" ] || { echo "Error: The directory $frags already exists. $root_hint" >&2; exit 1; }
mkdir "$frags"

# Function to determine change type based on commit message or PR labels
get_change_type() {
    local message="$1"
    local labels="$2"

    # Exclude messages that just say "ran the linter" (including pre-commit CI auto-linting bot)
    linting_messages=("style: lint" "style: pre-commit linting" "chore(pre-commit): autofix run")
    if [[ " ${linting_messages[@]} " =~ " $message " ]]; then
        echo "exclude"
        return
    fi

    # Define prefix mappings
    local -A prefix_map=(
        ["feat:"]="feature"
        ["fix:"]="bugfix"
        ["docs:"]="doc"
        ["build:"]="packaging"
        ["refactor:"]="refactor"
        ["chore:"]="misc"
        ["style:"]="misc"
    )

    # Check for prefixes
    for prefix in "${!prefix_map[@]}"; do
        if [[ "$message" == "$prefix"* ]]; then
            echo "${prefix_map[$prefix]}"
            return
        fi
    done

    # Fall back to label-based categorization if no prefix is found
    if [[ $labels == *"bug"* ]]; then
        echo "bugfix"
    elif [[ $labels == *"enhancement"* ]]; then
        echo "feature"
    elif [[ $labels == *"documentation"* ]]; then
        echo "doc"
    elif [[ $message == *"remove:"* || $message == *"deprecate:"* ]]; then
        # Non-Conventional Commit prefixes... Might delete these
        echo "removal"
    else
        echo "misc"
    fi
}

# Function to remove Conventional Commit prefix from message
remove_prefix() {
    local message="$1"
    echo "$message" | sed -E 's/^(feat|fix|docs|build|refactor|chore|style):\s*//'
}

# Get merged PRs from the last week
echo "Processing merged PRs from the last week..."
prs=$(gh pr list --limit 100 --state merged --json number,title,labels,mergedAt,body --search "merged:>=2024-09-01")

echo "$prs" | jq -c '.[] | select(.mergedAt >= "2024-09-01")' | while read pr; do
    number=$(echo $pr | jq -r '.number')
    title=$(echo $pr | jq -r '.title')
    labels=$(echo $pr | jq -r '.labels[].name' | tr '\n' ' ')
    body=$(echo $pr | jq -r '.body')

    change_type=$(get_change_type "$title $body" "$labels")

    # Create news fragment if not excluded
    if [[ "$change_type" != "exclude" ]]; then
        cleaned_title=$(remove_prefix "$title")
        echo "${cleaned_title^}" > "$frags/${number}.${change_type}.md"
        echo "Created news fragment for PR #$number: $cleaned_title (Type: $change_type)"
    else
        echo "Excluded PR #$number: $title"
    fi
done

# Get commits that are not associated with PRs
echo "Processing commits not associated with PRs..."
git log --since="1 week ago" --pretty=format:"%h %s" | while read -r commit_hash commit_message; do
    # Check if this commit is associated with a PR
    pr_number=$(gh pr list --state merged --search "$commit_hash" --json number --jq '.[0].number')
    if [ -z "$pr_number" ]; then
        change_type=$(get_change_type "$commit_message" "")
        if [[ "$change_type" != "exclude" ]]; then
            cleaned_message=$(remove_prefix "$commit_message")
            echo "${cleaned_message^}" > "$frags/${commit_hash}.${change_type}.md"
            echo "Created news fragment for commit ${commit_hash}: $cleaned_message (Type: ${change_type})"
        else
            echo "Excluded commit ${commit_hash}: $commit_message"
        fi
    fi
done

echo "News fragments creation complete."
