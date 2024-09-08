#!/bin/bash

# Ensure the news directory exists
mkdir -p news

# Function to determine change type based on commit message or PR labels
get_change_type() {
    local message="$1"
    if [[ $message == *"feat"* || $message == *"feature"* ]]; then
        echo "feature"
    elif [[ $message == *"fix"* || $message == *"bugfix"* ]]; then
        echo "bugfix"
    elif [[ $message == *"doc"* ]]; then
        echo "doc"
    elif [[ $message == *"remove"* || $message == *"deprecate"* ]]; then
        echo "removal"
    else
        echo "misc"
    fi
}

# Get commits from the last week
echo "Processing commits from the last week..."
git log --since="1 week ago" --pretty=format:"%h %s" | while read -r commit_hash commit_message; do
    change_type=$(get_change_type "$commit_message")
    echo "$commit_message" > "news/${commit_hash}.${change_type}.md"
    echo "Created news fragment for commit ${commit_hash}: ${commit_message} (Type: ${change_type})"
done

# Get merged PRs from the last week
echo "Processing merged PRs from the last week..."
prs=$(gh pr list --limit 100 --state merged --json number,title,labels,mergedAt --search "merged:>=2024-09-01")

echo "$prs" | jq -c '.[] | select(.mergedAt >= "2024-09-01")' | while read pr; do
    number=$(echo $pr | jq -r '.number')
    title=$(echo $pr | jq -r '.title')
    labels=$(echo $pr | jq -r '.labels[].name')
    
    change_type=$(get_change_type "$labels")
    
    # Create news fragment if it doesn't already exist
    if [[ ! -f "news/${number}.${change_type}.md" ]]; then
        echo "$title" > "news/${number}.${change_type}.md"
        echo "Created news fragment for PR #$number: $title (Type: $change_type)"
    fi
done

echo "News fragments creation complete."