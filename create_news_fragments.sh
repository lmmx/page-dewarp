#!/bin/bash

# Ensure the news directory exists
mkdir -p news

# Function to determine change type based on commit message or PR labels
get_change_type() {
    local message="$1"
    local labels="$2"
    
    if [[ $labels == *"bug"* || $message == *"fix:"* || $message == *"Fix"* ]]; then
        echo "bugfix"
    elif [[ $labels == *"enhancement"* || $message == *"feat:"* || $message == *"feature"* ]]; then
        echo "feature"
    elif [[ $labels == *"documentation"* || $message == *"docs:"* ]]; then
        echo "doc"
    elif [[ $message == *"remove:"* || $message == *"deprecate:"* ]]; then
        echo "removal"
    elif [[ $message == *"build:"* || $message == *"chore:"* || $message == *"style:"* || $message == *"refactor:"* ]]; then
        echo "misc"
    else
        echo "misc"
    fi
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
    
    # Create news fragment
    echo "$title" > "news/${number}.${change_type}.md"
    echo "Created news fragment for PR #$number: $title (Type: $change_type)"
done

# Get commits that are not associated with PRs
echo "Processing commits not associated with PRs..."
git log --since="1 week ago" --pretty=format:"%h %s" | while read -r commit_hash commit_message; do
    # Check if this commit is associated with a PR
    pr_number=$(gh pr list --state merged --search "$commit_hash" --json number --jq '.[0].number')
    if [ -z "$pr_number" ]; then
        change_type=$(get_change_type "$commit_message" "")
        echo "$commit_message" > "news/${commit_hash}.${change_type}.md"
        echo "Created news fragment for commit ${commit_hash}: ${commit_message} (Type: ${change_type})"
    fi
done

echo "News fragments creation complete."