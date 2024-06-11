#!/bin/bash

# Define the list of files to keep
keep_files=(
    "cot-0shot-end.txt"
    "struct-0shot-end.txt"
    "tot-0shot-end.txt"
    "cot-wm-chat-0shot-end.txt"
    "struct-yaml-0shot-end.txt"
)

# Convert the array into a string pattern for grep
keep_pattern=$(printf "|%s" "${keep_files[@]}")
keep_pattern=${keep_pattern:1}  # Remove the leading '|'

# Iterate over all files in the current directory
for file in *; do
    # Check if the file does not match any of the keep patterns
    if ! echo "$file" | grep -E "^($keep_pattern)$" > /dev/null; then
        echo "Deleting $file"
        rm "$file"
    fi
done

