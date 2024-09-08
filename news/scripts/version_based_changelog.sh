#!/bin/bash

# Function to get the latest version
get_latest_version() { git describe --tags --abbrev=0 2>/dev/null || echo "0.0.0" ; }

# Function to get the commit hash for a version or commit message
get_commit_hash() {
    local version_or_message="$1"
    if [[ "$version_or_message" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        git rev-list -n 1 "$version_or_message" 2>/dev/null
    else
        git log --format="%H" --grep="$version_or_message" -n 1
    fi
}

# Function to generate changelog for a specific range
generate_changelog() {
    local start_commit="$1"
    local end_commit="$2"
    local version="$3"

    echo "Generating changelog for version $version..."
    ./news/scripts/create_news_fragments.sh "$start_commit" "$end_commit"
    ./news/scripts/build_changelog.sh "$version"
    ./news/scripts/erase_news_fragments.sh
}

# Main script logic
latest_tag=$(get_latest_version)
latest_commit=$(git rev-parse HEAD)

if [[ "$1" == "next" ]]; then
    # Generate changelog for the next (unreleased) version
    start_commit=$(get_commit_hash "$latest_tag")
    generate_changelog "$start_commit" "$latest_commit" "next"
elif [[ "$1" == "all" ]]; then
    # Generate changelog for all versions
    versions=($(git tag --sort=-v:refname))
    previous_commit=""

    for version in "${versions[@]}"; do
        current_commit=$(get_commit_hash "$version")
        if [[ -n "$previous_commit" ]]; then
            generate_changelog "$current_commit" "$previous_commit" "$version"
        fi
        previous_commit="$current_commit"
    done

    # Generate changelog for the initial version
    initial_commit=$(git rev-list --max-parents=0 HEAD)
    generate_changelog "$initial_commit" "$previous_commit" "${versions[-1]}"

    # Generate changelog for unreleased changes
    generate_changelog "$previous_commit" "$latest_commit" "next"
else
    echo "Usage: $0 [next|all]"
    exit 1
fi
