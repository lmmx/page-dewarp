#!/bin/bash

# Function to get the latest version
get_latest_version() { git describe --tags --abbrev=0 2>/dev/null || echo "0.0.0" ; }

# Function to get the commit hash for a version or commit message
get_commit_hash() {
    local version_or_message="$1"
    if [[ "$version_or_message" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        # echo "Looking up $version_or_message with git rev-list"
        local version_tag="$version_or_message"
        git rev-list -n 1 "$version_tag" 2>/dev/null
    else
        # echo "Looking up commit hash with git log --format='%H' --grep=$version_or_message -n 1"
        local commit_message="$version_or_message"
        git log --format="%H" --grep="$commit_message" -n 1
    fi
}

# Function to generate changelog for a specific range
generate_changelog() {
    local start_commit="$1"
    local end_commit="$2"
    local version="$3"

    [[ -n "$start_commit" ]] || { echo "Error: start_commit is empty" >&2; return 1; }
    [[ -n "$end_commit" ]] || { echo "Error: end_commit is empty" >&2; return 1; }
    [[ -n "$version" ]] || { echo "Error: version is empty" >&2; return 1; }

    echo "Generating changelog for version $version..."
    ./news/scripts/create_news_fragments.sh "$start_commit" "$end_commit"
    ./news/scripts/build_changelog.sh "$version"
    ./news/scripts/erase_news_fragments.sh
}

# Main script logic
latest_tag=$(get_latest_version)
latest_commit=$(git rev-parse HEAD)
initial_commit=$(git rev-list --max-parents=0 HEAD)

echo "Script running with argument: $1"
echo "Latest tag: $latest_tag"
echo "Latest commit: $latest_commit"
echo "Initial commit: $initial_commit"
echo

if [[ "$1" == "next" ]]; then
    start_commit=$(get_commit_hash "$latest_tag")
    echo "Looked up start commit from get_commit_hash $latest_tag as $start_commit"
    echo "Generating changelog for the next (unreleased) version: start_commit=$start_commit, latest_commit=$latest_commit, version=next"
    generate_changelog "$start_commit" "$latest_commit" "next"
elif [[ "$1" == "all" ]]; then
    # Generate changelog for all versions
    versions=($(git tag --sort=v:refname))
    previous_commit="$initial_commit"
    echo "Initialised previous_commit as initial commit ($initial_commit)."

    for version in "${versions[@]}"; do
        echo
        echo "Looping through versions: on version=$version"
        current_commit=$(get_commit_hash "$version")
        echo "Looked up current commit from get_commit_hash $version as $current_commit"
        if [[ -n "$previous_commit" ]]; then
            echo "Generating changelog for $version: current_commit=$current_commit, previous_commit=$previous_commit, version=$version"
            generate_changelog "$current_commit" "$previous_commit" "$version"
        fi
        previous_commit="$current_commit"
    done

    echo
    echo "Generating changelog for the initial version: initial_commit=$initial_commit, previous_commit=$previous_commit, version=${versions[-1]}"
    generate_changelog "$initial_commit" "$previous_commit" "${versions[-1]}"

    echo "Generating changelog for unreleased changes: previous_commit=$previous_commit, latest_commit=$latest_commit, version=next"
    generate_changelog "$previous_commit" "$latest_commit" "next"
else
    echo "Usage: $0 [next|all]"
    exit 1
fi
